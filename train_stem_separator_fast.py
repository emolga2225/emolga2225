#!/usr/bin/env python3
"""
Fast training using preprocessed numpy arrays.

This loads pre-processed frame arrays instead of processing HDF5 on-the-fly,
resulting in much faster training (13s/iter -> <1s/iter).

First run:
    python preprocess_hdf5_to_arrays.py --data-dirs ajfa/ blackned/ --output-dir preprocessed_data/

Then train:
    python train_stem_separator_fast.py --preprocessed-dir preprocessed_data/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class PreprocessedSinusoidDataset(Dataset):
    """Fast dataset that loads preprocessed numpy arrays"""

    def __init__(self, preprocessed_dir):
        self.preprocessed_dir = Path(preprocessed_dir)

        # Load index
        index_file = self.preprocessed_dir / 'index.json'
        with open(index_file, 'r') as f:
            self.index = json.load(f)

        self.frames = self.index['frames']
        self.stem_names = self.index['stem_names']
        self.max_sinusoids = self.index['max_sinusoids']

        print(f"Loaded preprocessed dataset: {len(self.frames):,} frames")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_info = self.frames[idx]
        frame_file = self.preprocessed_dir / frame_info['file']

        # Load preprocessed arrays
        data = np.load(frame_file)
        fullmix = data['fullmix']  # (max_sinusoids, 3)
        stems = data['stems']      # (n_stems, max_sinusoids, 3)

        # Add frame dimension: (1, max_sinusoids, 3) and (n_stems, 1, max_sinusoids, 3)
        fullmix = fullmix[np.newaxis, :, :]  # (1, max_sinusoids, 3)
        stems = stems[:, np.newaxis, :, :]   # (n_stems, 1, max_sinusoids, 3)

        return (
            torch.from_numpy(fullmix),
            torch.from_numpy(stems)
        )


class TransformerStemSeparator(nn.Module):
    """
    Transformer-based stem separator that processes exact frequencies.

    MEMORY-EFFICIENT: Works with 1 frame per chunk!
    - Sequence length = max_sinusoids (e.g., 2000)
    - Attention matrix: 2000 × 2000 = 16 MB (fits easily!)

    Input: (batch, n_frames, max_sinusoids, 3) where 3 = [freq, amp, phase]
    Output: (batch, n_stems, n_frames, max_sinusoids, 3)
    """

    def __init__(self, n_stems=5, max_sinusoids=2000, d_model=128, nhead=4,
                 num_layers=4, dim_feedforward=512):
        super().__init__()
        self.n_stems = n_stems
        self.max_sinusoids = max_sinusoids
        self.d_model = d_model

        # Embed each sinusoid [freq, amp, phase] -> d_model dimensions
        self.sinusoid_embed = nn.Linear(3, d_model)

        # Positional encoding for sinusoid index
        self.sinusoid_pos_embed = nn.Parameter(torch.randn(1, max_sinusoids, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads - one per stem
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, 3)  # Predict [freq, amp, phase] for each stem
            for _ in range(n_stems)
        ])

    def forward(self, x):
        """
        x: (batch, n_frames, max_sinusoids, 3)
        returns: (batch, n_stems, n_frames, max_sinusoids, 3)
        """
        batch_size, n_frames, max_sines, _ = x.shape

        # Process each frame independently to avoid huge sequence length
        # With n_frames=1, this is just one iteration!
        all_stem_preds = []

        for frame_idx in range(n_frames):
            # Get sinusoids for this frame: (batch, max_sinusoids, 3)
            frame_sines = x[:, frame_idx, :, :]

            # Embed sinusoids: (batch, max_sinusoids, d_model)
            frame_embed = self.sinusoid_embed(frame_sines)

            # Add positional encoding
            frame_embed = frame_embed + self.sinusoid_pos_embed

            # Create padding mask (freq == 0 means padded)
            padding_mask = (frame_sines[:, :, 0] == 0)  # (batch, max_sinusoids)

            # Transformer for THIS FRAME
            # Sequence length = max_sinusoids (e.g., 2000), not n_frames * max_sinusoids!
            encoded = self.transformer(frame_embed, src_key_padding_mask=padding_mask)
            # Shape: (batch, max_sinusoids, d_model)

            # Generate predictions for each stem
            frame_stem_preds = []
            for stem_idx in range(self.n_stems):
                stem_output = self.output_heads[stem_idx](encoded)  # (batch, max_sines, 3)

                # Apply constraints:
                # - Frequency: keep positive, scale to reasonable range (0-22050 Hz)
                # - Amplitude: non-negative
                # - Phase: wrap to [-pi, pi]
                freq = F.relu(stem_output[:, :, 0]) * 22050 / 100
                amp = F.relu(stem_output[:, :, 1])
                phase = torch.atan2(torch.sin(stem_output[:, :, 2]),
                                   torch.cos(stem_output[:, :, 2]))

                stem_prediction = torch.stack([freq, amp, phase], dim=-1)
                frame_stem_preds.append(stem_prediction)

            # Stack stems: (batch, n_stems, max_sinusoids, 3)
            frame_stem_preds = torch.stack(frame_stem_preds, dim=1)
            all_stem_preds.append(frame_stem_preds)

        # Stack frames: (batch, n_stems, n_frames, max_sinusoids, 3)
        output = torch.stack(all_stem_preds, dim=2)

        return output


def train_epoch(model, dataloader, optimizer, device, gradient_accumulation_steps=1, use_amp=False):
    """Train for one epoch with gradient accumulation and optional mixed precision"""
    model.train()
    total_loss = 0

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch_idx, (fullmix, stems) in enumerate(tqdm(dataloader, desc="Training")):
        fullmix = fullmix.to(device)
        stems = stems.to(device)

        # Mixed precision context
        with torch.cuda.amp.autocast() if use_amp else torch.enable_grad():
            # Forward pass
            pred_stems = model(fullmix)

            # L1 loss
            loss = F.l1_loss(pred_stems, stems)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-dir', required=True,
                       help='Directory containing preprocessed arrays (from preprocess_hdf5_to_arrays.py)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (can be much larger with preprocessed data)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Accumulate gradients over N steps')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision (fp16) training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers (can use multiple with preprocessed data)')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataset = PreprocessedSinusoidDataset(args.preprocessed_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # Can use multiple workers with .npz files!
        pin_memory=True if device.type == 'cuda' else False
    )

    # Model
    model = TransformerStemSeparator(
        n_stems=len(dataset.stem_names),
        max_sinusoids=dataset.max_sinusoids,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Data: {len(dataset):,} frames")
    print(f"Stems: {dataset.stem_names}")
    print(f"Max sinusoids: {dataset.max_sinusoids}")

    # Show attention matrix size
    seq_len = dataset.max_sinusoids
    attn_elements = seq_len * seq_len
    attn_mb = (attn_elements * 4) / (1024**2)
    print(f"\nAttention matrix per frame:")
    print(f"  Sequence length: {seq_len:,}")
    print(f"  Memory: {attn_mb:.1f} MB")

    print(f"\nTraining configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  DataLoader workers: {args.num_workers}")

    # Training loop
    for epoch in range(args.epochs):
        loss = train_epoch(
            model, dataloader, optimizer, device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=args.mixed_precision
        )
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.6f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': {
                    'stem_names': dataset.stem_names,
                    'max_sinusoids': dataset.max_sinusoids,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_layers': args.num_layers
                }
            }
            torch.save(checkpoint, f'stem_separator_exact_freq_epoch{epoch+1}.pt')
            print(f"Saved checkpoint: stem_separator_exact_freq_epoch{epoch+1}.pt")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
