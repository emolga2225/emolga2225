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

    def __init__(self, preprocessed_dir, chunk_frames=1, subset=None):
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

        # Create chunks from consecutive frames with overlap
        for data_dir, dir_frames in frames_by_dir.items():
            # Sort by frame index
            dir_frames.sort(key=lambda x: x['frame_idx'])

            # Create overlapping chunks with stride=1 for temporal continuity
            # This ensures the model learns smooth evolution of freq, amp, and phase
            for i in range(len(dir_frames) - chunk_frames + 1):
                chunk_frames_list = dir_frames[i:i + chunk_frames]
                self.chunks.append(chunk_frames_list)

        print(f"Loaded preprocessed dataset: {len(self.frames):,} frames")
        print(f"Organized into {len(self.chunks):,} chunks of {chunk_frames} frame(s) each")

        # Apply subset if specified
        if subset is not None and subset < len(self.chunks):
            self.chunks = self.chunks[:subset]
            print(f"Using subset: {len(self.chunks):,} chunks (--subset {subset})")

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
            fullmix_frames.append(data['fullmix'].copy())  # (max_sinusoids, 3)
            stem_frames.append(data['stems'].copy())      # (n_stems, max_sinusoids, 3)

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

    def __init__(self, n_stems=5, max_sinusoids=900, d_model=128, nhead=4,
                 num_layers=4, dim_feedforward=512):
        super().__init__()
        self.n_stems = n_stems
        self.max_sinusoids = max_sinusoids
        self.d_model = d_model

        # Embed each sinusoid [freq, amp, phase] -> d_model dimensions
        self.sinusoid_embed = nn.Linear(3, d_model)

        # Positional encoding for sinusoid index (which sinusoid in the frame)
        self.sinusoid_pos_embed = nn.Parameter(torch.randn(1, max_sinusoids, d_model))

        # STEM CONDITIONING: Learnable embeddings for each stem type
        # This tells the model "extract vocals" vs "extract drums" etc.
        self.stem_embeddings = nn.Embedding(n_stems, d_model)

        # Frame positional embeddings for temporal context (which frame in time)
        # This is separate from sinusoid position - tells model temporal order
        self.max_frames = 10  # Support up to 10 frames per chunk
        self.frame_pos_embed = nn.Parameter(torch.randn(1, self.max_frames, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Single output head shared across stems (stem conditioning handles differentiation)
        self.output_head = nn.Linear(d_model, 3)  # Predict [freq, amp, phase]

        # Initialize weights with better variance
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with reasonable values"""
        # Initialize embedding layers
        nn.init.xavier_uniform_(self.sinusoid_embed.weight, gain=0.1)
        nn.init.zeros_(self.sinusoid_embed.bias)

        # Initialize output head
        nn.init.xavier_uniform_(self.output_head.weight, gain=0.1)
        nn.init.zeros_(self.output_head.bias)

        # Initialize positional embeddings
        nn.init.normal_(self.sinusoid_pos_embed, mean=0, std=0.02)
        nn.init.normal_(self.frame_pos_embed, mean=0, std=0.02)

        # Initialize stem embeddings with larger variance (they're important!)
        nn.init.normal_(self.stem_embeddings.weight, mean=0, std=0.1)

    def forward(self, x):
        """
        x: (batch, n_frames, max_sinusoids, 3)
        returns: (batch, n_stems, n_frames, max_sinusoids, 3)

        NEW: Process ALL stems in parallel with batched stem conditioning!
        Much faster than sequential processing - uses GPU parallelism.
        """
        batch_size, n_frames, max_sines, _ = x.shape
        seq_len = n_frames * max_sines

        # NORMALIZE INPUTS to prevent numerical instability
        # x[..., 0] = frequency (Hz), x[..., 1] = amplitude, x[..., 2] = phase (radians)
        x_norm = x.clone()

        # Normalize frequency to [0, 1] range (48kHz sample rate, Nyquist = 24kHz)
        x_norm[..., 0] = x[..., 0] / 24000.0

        # Log-scale amplitude (handles large dynamic range, prevents huge values)
        # Use log1p to handle zero amplitudes gracefully
        x_norm[..., 1] = torch.log1p(torch.abs(x[..., 1])) / 10.0  # Scale down by 10

        # Phase already in [-π, π] range - normalize to [-1, 1]
        x_norm[..., 2] = x[..., 2] / 3.14159265

        # Flatten frames into sequence: (batch, n_frames, max_sinusoids, 3) -> (batch, seq_len, 3)
        x_flat = x_norm.reshape(batch_size, seq_len, 3)

        # Embed sinusoids: (batch, seq_len, 3) -> (batch, seq_len, d_model)
        x_embed = self.sinusoid_embed(x_flat)

        # Add sinusoid positional encoding (which sinusoid within each frame)
        pos_embed = self.sinusoid_pos_embed.repeat(1, n_frames, 1)
        x_embed = x_embed + pos_embed

        # Add frame positional encoding (which frame in time)
        x_embed_frames = x_embed.reshape(batch_size, n_frames, max_sines, self.d_model)
        x_embed_frames = x_embed_frames + self.frame_pos_embed[:, :n_frames, :, :]
        x_embed = x_embed_frames.reshape(batch_size, seq_len, self.d_model)

        # BATCHED STEM CONDITIONING: Process all stems in parallel!
        # Expand input for all stems: (batch, seq_len, d_model) -> (batch, n_stems, seq_len, d_model)
        # Use repeat() instead of expand() for better numerical stability with mixed precision
        x_embed_expanded = x_embed.unsqueeze(1).repeat(1, self.n_stems, 1, 1)

        # Get all stem embeddings: (n_stems, d_model)
        stem_embeds = self.stem_embeddings.weight  # All stem embeddings at once

        # Expand stem embeddings to match input: (n_stems, d_model) -> (batch, n_stems, seq_len, d_model)
        # Use repeat() for memory allocation (better with autocast)
        stem_embeds_expanded = stem_embeds.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, seq_len, 1)

        # Add stem conditioning: (batch, n_stems, seq_len, d_model)
        x_conditioned = x_embed_expanded + stem_embeds_expanded

        # Reshape to batch all stems together: (batch, n_stems, seq_len, d_model) -> (batch * n_stems, seq_len, d_model)
        x_conditioned = x_conditioned.reshape(batch_size * self.n_stems, seq_len, self.d_model)

        # Check for NaN after conditioning (debug)
        if torch.isnan(x_conditioned).any():
            print(f"NaN in x_conditioned! x_embed range: [{x_embed.min():.4f}, {x_embed.max():.4f}]")
            print(f"stem_embeds range: [{stem_embeds.min():.4f}, {stem_embeds.max():.4f}]")

        # Create padding mask for batched input
        padding_mask = (x_flat[:, :, 0] == 0)  # (batch, seq_len)
        padding_mask_expanded = padding_mask.unsqueeze(1).expand(-1, self.n_stems, -1).reshape(batch_size * self.n_stems, seq_len)

        # Single transformer call for ALL stems at once! (batch * n_stems as batch dimension)
        encoded = self.transformer(x_conditioned, src_key_padding_mask=padding_mask_expanded)

        # Reshape back: (batch * n_stems, seq_len, d_model) -> (batch, n_stems, seq_len, d_model)
        encoded = encoded.reshape(batch_size, self.n_stems, seq_len, self.d_model)

        # Reshape to frames: (batch, n_stems, seq_len, d_model) -> (batch, n_stems, n_frames, max_sines, d_model)
        encoded = encoded.reshape(batch_size, self.n_stems, n_frames, max_sines, self.d_model)

        # Predict sinusoid parameters for all stems at once
        stem_output = self.output_head(encoded)  # (batch, n_stems, n_frames, max_sines, 3)

        # DENORMALIZE OUTPUTS back to original scale
        # Model outputs normalized values, convert back to Hz/amplitude/radians

        # Frequency: [0, 1] -> [0, 24000] Hz (48kHz sample rate)
        freq = torch.sigmoid(stem_output[:, :, :, :, 0]) * 24000.0

        # Amplitude: log-scaled -> linear scale
        # Model outputs log1p(amp)/10, so reverse: amp = expm1(output * 10)
        amp = torch.expm1(torch.relu(stem_output[:, :, :, :, 1]) * 10.0)

        # Phase: [-1, 1] -> [-π, π] radians
        phase = torch.tanh(stem_output[:, :, :, :, 2]) * 3.14159265

        # Stack into final output: (batch, n_stems, n_frames, max_sinusoids, 3)
        output = torch.stack([freq, amp, phase], dim=-1)

        return output


def train_epoch(model, dataloader, optimizer, device, gradient_accumulation_steps=1, use_amp=False, max_grad_norm=1.0):
    """Train for one epoch with gradient accumulation and optional mixed precision"""
    model.train()
    total_loss = 0

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch_idx, (fullmix, stems) in enumerate(tqdm(dataloader, desc="Training")):
        fullmix = fullmix.to(device)
        stems = stems.to(device)

        # Check for NaN/Inf in input data
        if torch.isnan(fullmix).any() or torch.isinf(fullmix).any():
            print(f"\nWARNING: NaN/Inf detected in input fullmix at batch {batch_idx}")
            continue
        if torch.isnan(stems).any() or torch.isinf(stems).any():
            print(f"\nWARNING: NaN/Inf detected in input stems at batch {batch_idx}")
            continue

        # Mixed precision context
        with torch.cuda.amp.autocast() if use_amp else torch.enable_grad():
            # Forward pass
            pred_stems = model(fullmix)

            # Check for NaN/Inf in model output
            if torch.isnan(pred_stems).any() or torch.isinf(pred_stems).any():
                print(f"\nERROR: NaN/Inf in model output at batch {batch_idx}")
                print(f"  fullmix range: [{fullmix.min():.4f}, {fullmix.max():.4f}]")
                print(f"  stems range: [{stems.min():.4f}, {stems.max():.4f}]")
                print(f"  pred_stems range: [{pred_stems.min():.4f}, {pred_stems.max():.4f}]")
                raise ValueError("NaN/Inf detected in model output - training unstable")

            # L1 loss
            loss = F.l1_loss(pred_stems, stems)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"\nERROR: NaN loss at batch {batch_idx}")
                raise ValueError("NaN loss detected")

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
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping (prevents gradient explosion)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint (path to .pt file)')
    parser.add_argument('--subset', type=int, default=None,
                       help='Train on subset of N chunks (useful for quick testing)')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Calculate chunk frames
    if args.chunk_duration is None:
        chunk_frames = 1
        print("\nUsing 1 frame per chunk")
    else:
        chunk_frames = int(args.chunk_duration * 48000 / 512)
        print(f"\nUsing {chunk_frames} frames per chunk ({args.chunk_duration}s)")

    # Dataset
    dataset = PreprocessedSinusoidDataset(args.preprocessed_dir, chunk_frames=chunk_frames, subset=args.subset)

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

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch} (loss: {checkpoint['loss']:.6f})")

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
    for epoch in range(start_epoch, args.epochs):
        loss = train_epoch(
            model, dataloader, optimizer, device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=args.mixed_precision,
            max_grad_norm=args.max_grad_norm
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
