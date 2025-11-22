#!/usr/bin/env python3
"""
Train stem separator preserving EXACT frequencies (no binning).

Key difference from 2D grid approach:
- NO frequency binning - preserve exact Hz values (50 Hz, 80 Hz, 175 Hz, etc.)
- Data organized by time frames
- At each frame: list of all active sinusoids with exact frequencies
- Representation: (n_frames, max_sinusoids, 3) where 3 = [freq, amp, phase]

This keeps 100% of frequency precision while creating fixed-size inputs for neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse


class ExactFreqSinusoidDataset(Dataset):
    """Dataset that preserves exact frequencies - no binning!"""

    def __init__(self, data_dirs, stem_names, chunk_frames=344, max_sinusoids_per_frame=500):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.stem_names = stem_names
        self.chunk_frames = chunk_frames
        self.max_sinusoids_per_frame = max_sinusoids_per_frame

        # Build list of all chunks
        self.chunks = []
        for data_dir in self.data_dirs:
            fullmix_h5 = data_dir / 'fullmix_tracks.h5'
            if not fullmix_h5.exists():
                fullmix_h5 = data_dir / 'fullmix.h5'
            if not fullmix_h5.exists():
                continue

            # Determine number of chunks
            with h5py.File(fullmix_h5, 'r') as f:
                if 'c0' in f and 'i' in f['c0']:
                    max_frame = f['c0']['i'][:].max() if len(f['c0']['i']) > 0 else 0
                else:
                    continue

            n_chunks = (max_frame // chunk_frames) + 1
            for chunk_idx in range(n_chunks):
                self.chunks.append((data_dir, chunk_idx))

        print(f"Dataset: {len(self.chunks)} chunks from {len(self.data_dirs)} directories")

    def sinusoids_to_exact_freq_array(self, h5_path, chunk_idx):
        """
        Convert sinusoids to exact frequency representation - NO BINNING!

        At each frame, store all active sinusoids with their EXACT frequencies.

        Returns:
            array of shape (chunk_frames, max_sinusoids_per_frame, 3)
            where 3 = [freq (Hz), amplitude, phase (radians)]
        """
        # Initialize array (will pad with zeros where no sinusoids)
        array = np.zeros((self.chunk_frames, self.max_sinusoids_per_frame, 3), dtype=np.float32)

        if not h5_path.exists():
            return array

        start_frame = chunk_idx * self.chunk_frames
        end_frame = start_frame + self.chunk_frames

        # Dictionary: frame_idx -> list of (freq, amp, phase)
        frame_sinusoids = {i: [] for i in range(self.chunk_frames)}

        with h5py.File(h5_path, 'r') as f:
            if 'c0' not in f:
                return array

            grp = f['c0']
            track_lens = grp['len'][:]
            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            phases = grp['p'][:] if 'p' in grp else np.zeros_like(amplitudes)
            frames = grp['i'][:]

            # Process each track
            offset = 0
            for track_len in track_lens:
                track_freqs = frequencies[offset:offset + track_len]
                track_amps = amplitudes[offset:offset + track_len]
                track_phases = phases[offset:offset + track_len]
                track_frames = frames[offset:offset + track_len]

                # Only process sinusoids in this chunk's time range
                mask = (track_frames >= start_frame) & (track_frames < end_frame)
                chunk_freqs = track_freqs[mask]
                chunk_amps = track_amps[mask]
                chunk_phases = track_phases[mask]
                chunk_frames_local = track_frames[mask] - start_frame

                # Add to frame dictionary with EXACT frequencies (no binning!)
                for freq, amp, phase, frame in zip(chunk_freqs, chunk_amps, chunk_phases, chunk_frames_local):
                    frame_idx = int(frame)
                    if 0 <= frame_idx < self.chunk_frames:
                        frame_sinusoids[frame_idx].append([freq, amp, phase])

                offset += track_len

        # Convert dictionary to fixed-size array
        for frame_idx in range(self.chunk_frames):
            sinusoids = frame_sinusoids[frame_idx]
            n_sinusoids = min(len(sinusoids), self.max_sinusoids_per_frame)

            if n_sinusoids > 0:
                # Sort by frequency for consistency (optional but helps)
                sinusoids = sorted(sinusoids, key=lambda x: x[0])

                # Fill array (truncate if too many sinusoids)
                array[frame_idx, :n_sinusoids, :] = sinusoids[:n_sinusoids]

                # If more sinusoids than max, log warning (only once)
                if len(sinusoids) > self.max_sinusoids_per_frame and frame_idx == 0:
                    print(f"Warning: Frame has {len(sinusoids)} sinusoids, truncating to {self.max_sinusoids_per_frame}")

        return array

    def find_stem_h5(self, data_dir, stem_name):
        """Find HDF5 file for a stem (handles bass/rhythm and multi-file drums)"""
        if stem_name == 'drums':
            # Merge all drum files
            drum_files = []
            for i in range(1, 10):
                drum_h5 = data_dir / f'drums_{i}_tracks.h5'
                if drum_h5.exists():
                    drum_files.append(drum_h5)
            return drum_files if drum_files else [data_dir / 'drums_tracks.h5']

        elif stem_name == 'bass':
            for name in ['bass_tracks.h5', 'rhythm_tracks.h5', 'bass.h5', 'rhythm.h5']:
                h5_path = data_dir / name
                if h5_path.exists():
                    return [h5_path]

        # Regular stems
        for suffix in ['_tracks.h5', '.h5']:
            h5_path = data_dir / f'{stem_name}{suffix}'
            if h5_path.exists():
                return [h5_path]

        return []

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        data_dir, chunk_idx = self.chunks[idx]

        # Load fullmix as input
        fullmix_h5 = data_dir / 'fullmix_tracks.h5'
        if not fullmix_h5.exists():
            fullmix_h5 = data_dir / 'fullmix.h5'

        # Shape: (chunk_frames, max_sinusoids, 3) where 3 = [freq, amp, phase]
        fullmix_array = self.sinusoids_to_exact_freq_array(fullmix_h5, chunk_idx)

        # Load each stem as target
        stem_arrays = []
        for stem_name in self.stem_names:
            h5_files = self.find_stem_h5(data_dir, stem_name)

            # Merge arrays if multiple files (e.g., drums)
            stem_array = np.zeros((self.chunk_frames, self.max_sinusoids_per_frame, 3), dtype=np.float32)

            for h5_file in h5_files:
                file_array = self.sinusoids_to_exact_freq_array(h5_file, chunk_idx)

                # Merge: combine sinusoids from multiple files
                # For each frame, append sinusoids (up to max limit)
                for frame_idx in range(self.chunk_frames):
                    # Get existing sinusoids for this frame
                    existing = stem_array[frame_idx]
                    existing_count = np.count_nonzero(existing[:, 0])  # Count non-zero frequencies

                    # Get new sinusoids to add
                    new = file_array[frame_idx]
                    new_count = np.count_nonzero(new[:, 0])

                    # Append new sinusoids after existing ones (up to max)
                    space_left = self.max_sinusoids_per_frame - existing_count
                    if space_left > 0 and new_count > 0:
                        n_to_add = min(new_count, space_left)
                        stem_array[frame_idx, existing_count:existing_count + n_to_add, :] = new[:n_to_add, :]

            stem_arrays.append(stem_array)

        # Stack stems: (n_stems, chunk_frames, max_sinusoids, 3)
        stem_arrays = np.stack(stem_arrays, axis=0)

        return (
            torch.from_numpy(fullmix_array),  # (chunk_frames, max_sinusoids, 3)
            torch.from_numpy(stem_arrays)      # (n_stems, chunk_frames, max_sinusoids, 3)
        )


class TransformerStemSeparator(nn.Module):
    """
    Transformer-based stem separator that processes exact frequencies.

    MEMORY-EFFICIENT: Works best with 1 frame per chunk!
    - With 1 frame: sequence length = max_sinusoids (e.g., 2000)
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

        # Positional encoding for sinusoid index (not needed for frequency since it's in the input!)
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
                freq = F.relu(stem_output[:, :, 0]) * 22050 / 100  # Scale from activation
                amp = F.relu(stem_output[:, :, 1])
                phase = torch.atan2(torch.sin(stem_output[:, :, 2]),
                                   torch.cos(stem_output[:, :, 2]))

                stem_prediction = torch.stack([freq, amp, phase], dim=-1)
                frame_stem_preds.append(stem_prediction)

            # Stack stems: (batch, n_stems, max_sinusoids, 3)
            frame_stem_preds = torch.stack(frame_stem_preds, dim=1)
            all_stem_preds.append(frame_stem_preds)

        # Stack frames: (batch, n_stems, n_frames, max_sinusoids, 3)
        # Note: with n_frames=1, this just adds a dimension
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

        # Free memory
        del fullmix, stems, pred_stems, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', nargs='+', required=True)
    parser.add_argument('--stem-names', nargs='+',
                       default=['vocals', 'guitar', 'bass', 'drums', 'song'])
    parser.add_argument('--chunk-duration', type=float, default=None,
                       help='Duration of each chunk in seconds (default: None = 1 frame per chunk)')
    parser.add_argument('--max-sinusoids', type=int, default=2000,
                       help='Max sinusoids per frame (will truncate/pad). Use inspect_max_sinusoids.py to find optimal value.')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (can be larger with 1-frame chunks)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Accumulate gradients over N steps (effective batch = batch-size * N)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision (fp16) training to reduce memory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model dimension (reduced for 1-frame chunks)')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of transformer layers')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use 1 frame per chunk if not specified (MUCH more memory efficient!)
    if args.chunk_duration is None:
        chunk_frames = 1
        print("\nUsing 1 frame per chunk (most memory efficient)")
    else:
        chunk_frames = int(args.chunk_duration * 44100 / 512)
        print(f"\nUsing {chunk_frames} frames per chunk ({args.chunk_duration}s)")

    # Dataset
    dataset = ExactFreqSinusoidDataset(
        args.data_dirs,
        args.stem_names,
        chunk_frames=chunk_frames,
        max_sinusoids_per_frame=args.max_sinusoids
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # HDF5 doesn't support multiprocessing
    )

    # Model
    model = TransformerStemSeparator(
        n_stems=len(args.stem_names),
        max_sinusoids=args.max_sinusoids,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Data representation: ({chunk_frames} frames, {args.max_sinusoids} sinusoids/frame, 3 features)")
    print(f"Features per sinusoid: [exact_freq_Hz, amplitude, phase_radians]")
    print(f"Stems: {args.stem_names}")

    # Show attention matrix size
    seq_len = args.max_sinusoids  # With per-frame processing
    attn_elements = seq_len * seq_len
    attn_mb = (attn_elements * 4) / (1024**2)  # fp32
    print(f"\nAttention matrix per frame:")
    print(f"  Sequence length: {seq_len:,}")
    print(f"  Attention matrix: {seq_len:,} × {seq_len:,} = {attn_elements:,} elements")
    print(f"  Memory: {attn_mb:.1f} MB (vs 881 GB with flattened approach!)")

    print(f"\nMemory optimizations:")
    print(f"  Frames per chunk: {chunk_frames}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps} steps")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Mixed precision: {args.mixed_precision}")

    # Estimate memory per sample
    bytes_per_sample = chunk_frames * args.max_sinusoids * 3 * 4  # float32
    mb_per_sample = bytes_per_sample / (1024**2)
    mb_stems = mb_per_sample * len(args.stem_names)
    print(f"\nEstimated memory per sample:")
    print(f"  Fullmix: {mb_per_sample:.1f} MB")
    print(f"  All stems: {mb_stems:.1f} MB")
    print(f"  Per batch: {(mb_per_sample + mb_stems) * args.batch_size:.1f} MB")

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
                    'stem_names': args.stem_names,
                    'chunk_frames': chunk_frames,
                    'max_sinusoids': args.max_sinusoids,
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
