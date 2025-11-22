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
    """Fast dataset that loads preprocessed numpy arrays with multi-frame chunk support"""

    def __init__(self, preprocessed_dir, chunk_frames=1):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.chunk_frames = chunk_frames

        # Load index
        index_file = self.preprocessed_dir / 'index.json'
        with open(index_file, 'r') as f:
            self.index = json.load(f)

        self.frames = self.index['frames']
        self.stem_names = self.index['stem_names']
        self.max_sinusoids = self.index['max_sinusoids']

        # Group frames by data_dir for multi-frame chunking
        self.chunks = []
        frames_by_dir = {}
        for frame in self.frames:
            data_dir = frame['data_dir']
            if data_dir not in frames_by_dir:
                frames_by_dir[data_dir] = []
            frames_by_dir[data_dir].append(frame)

        # Create chunks from consecutive frames
        for data_dir, dir_frames in frames_by_dir.items():
            # Sort by frame index
            dir_frames.sort(key=lambda x: x['frame_idx'])

            # Create chunks of consecutive frames
            for i in range(0, len(dir_frames), chunk_frames):
                chunk_frames_list = dir_frames[i:i + chunk_frames]
                # Only include full chunks (or last partial chunk)
                if len(chunk_frames_list) == chunk_frames or i + chunk_frames >= len(dir_frames):
                    self.chunks.append(chunk_frames_list)

        print(f"Loaded preprocessed dataset: {len(self.frames):,} frames")
        print(f"Organized into {len(self.chunks):,} chunks of {chunk_frames} frame(s) each")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk_frames_list = self.chunks[idx]

        # Load all frames in this chunk
        fullmix_frames = []
        stem_frames = []

        for frame_info in chunk_frames_list:
            frame_file = self.preprocessed_dir / frame_info['file']

            # Load preprocessed arrays
            data = np.load(frame_file)
            fullmix_frames.append(data['fullmix'])  # (max_sinusoids, 3)
            stem_frames.append(data['stems'])      # (n_stems, max_sinusoids, 3)

        # Stack frames: (n_frames, max_sinusoids, 3) and (n_stems, n_frames, max_sinusoids, 3)
        fullmix = np.stack(fullmix_frames, axis=0)  # (n_frames, max_sinusoids, 3)
        stems = np.stack(stem_frames, axis=1)       # (n_stems, n_frames, max_sinusoids, 3)

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

        With multi-frame chunks, process all frames together for temporal context!
        """
        batch_size, n_frames, max_sines, _ = x.shape

        if n_frames == 1:
            # Single frame: process efficiently
            frame_sines = x[:, 0, :, :]  # (batch, max_sinusoids, 3)

            # Embed sinusoids
            frame_embed = self.sinusoid_embed(frame_sines)
            frame_embed = frame_embed + self.sinusoid_pos_embed

            # Padding mask
            padding_mask = (frame_sines[:, :, 0] == 0)

            # Transformer
            encoded = self.transformer(frame_embed, src_key_padding_mask=padding_mask)

            # Predict for each stem
            stem_preds = []
            for stem_idx in range(self.n_stems):
                stem_output = self.output_heads[stem_idx](encoded)

                freq = F.relu(stem_output[:, :, 0]) * 22050 / 100
                amp = F.relu(stem_output[:, :, 1])
                phase = torch.atan2(torch.sin(stem_output[:, :, 2]),
                                   torch.cos(stem_output[:, :, 2]))

                stem_prediction = torch.stack([freq, amp, phase], dim=-1)
                stem_preds.append(stem_prediction)

            # Stack and add frame dimension
            output = torch.stack(stem_preds, dim=1).unsqueeze(2)  # (batch, n_stems, 1, max_sines, 3)

        else:
            # Multi-frame: flatten all frames into one sequence for temporal context!
            # Reshape: (batch, n_frames, max_sinusoids, 3) -> (batch, n_frames * max_sinusoids, 3)
            x_flat = x.reshape(batch_size, n_frames * max_sines, 3)

            # Embed sinusoids
            x_embed = self.sinusoid_embed(x_flat)

            # Add positional encodings (tile for each frame)
            pos_embed = self.sinusoid_pos_embed.repeat(1, n_frames, 1)
            x_embed = x_embed + pos_embed

            # Create padding mask
            padding_mask = (x_flat[:, :, 0] == 0)

            # Transformer on full sequence (provides temporal context!)
            encoded = self.transformer(x_embed, src_key_padding_mask=padding_mask)

            # Reshape back: (batch, n_frames * max_sinusoids, d_model) -> (batch, n_frames, max_sinusoids, d_model)
            encoded = encoded.reshape(batch_size, n_frames, max_sines, self.d_model)

            # Predict for each stem
            stem_preds = []
            for stem_idx in range(self.n_stems):
                stem_output = self.output_heads[stem_idx](encoded)  # (batch, n_frames, max_sines, 3)

                freq = F.relu(stem_output[:, :, :, 0]) * 22050 / 100
                amp = F.relu(stem_output[:, :, :, 1])
                phase = torch.atan2(torch.sin(stem_output[:, :, :, 2]),
                                   torch.cos(stem_output[:, :, :, 2]))

                stem_prediction = torch.stack([freq, amp, phase], dim=-1)
                stem_preds.append(stem_prediction)

            # Stack stems: (batch, n_stems, n_frames, max_sinusoids, 3)
            output = torch.stack(stem_preds, dim=1)

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
    parser.add_argument('--chunk-duration', type=float, default=None,
                       help='Duration of each chunk in seconds (default: None = 1 frame). Try 0.046 for 4 frames.')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (reduce if using multi-frame chunks)')
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

    # Calculate chunk frames
    if args.chunk_duration is None:
        chunk_frames = 1
        print("\nUsing 1 frame per chunk")
    else:
        chunk_frames = int(args.chunk_duration * 44100 / 512)
        print(f"\nUsing {chunk_frames} frames per chunk ({args.chunk_duration}s)")

    # Dataset
    dataset = PreprocessedSinusoidDataset(args.preprocessed_dir, chunk_frames=chunk_frames)

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
    print(f"Data: {len(dataset):,} chunks ({chunk_frames} frame(s) each)")
    print(f"Stems: {dataset.stem_names}")
    print(f"Max sinusoids per frame: {dataset.max_sinusoids}")

    # Show attention matrix size
    seq_len = dataset.max_sinusoids * chunk_frames
    attn_elements = seq_len * seq_len
    attn_mb = (attn_elements * 4) / (1024**2)
    attn_gb = attn_mb / 1024
    print(f"\nAttention matrix size:")
    print(f"  Sequence length: {seq_len:,} ({chunk_frames} frames × {dataset.max_sinusoids} sinusoids)")
    if attn_gb < 1:
        print(f"  Memory: {attn_mb:.1f} MB")
    else:
        print(f"  Memory: {attn_gb:.2f} GB")

    print(f"\nTraining configuration:")
    print(f"  Chunk frames: {chunk_frames}")
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
