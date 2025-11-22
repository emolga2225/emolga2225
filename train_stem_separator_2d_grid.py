#!/usr/bin/env python3
"""
Train stem separator using 2D time-frequency grid representation.

This approach converts sinusoidal tracks to 2D grids, keeping ALL data
while creating fixed-size inputs suitable for CNNs.

Key idea:
- Each sinusoidal track = one frequency evolving over time
- Convert to 2D grid: (frequency_bins, time_frames)
- Use U-Net architecture for source separation (proven for audio)
- Output 5 separate 2D grids (one per stem)
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


class SinusoidalGrid2DDataset(Dataset):
    """Dataset that converts HDF5 sinusoids to 2D grids on-the-fly"""

    def __init__(self, data_dirs, stem_names, n_freq_bins=512, chunk_frames=344,
                 freq_min=0, freq_max=22050):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.stem_names = stem_names
        self.n_freq_bins = n_freq_bins
        self.chunk_frames = chunk_frames
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_bin_width = (freq_max - freq_min) / n_freq_bins

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

    def sinusoids_to_grid(self, h5_path, chunk_idx):
        """Convert sinusoids in a chunk to 2D grid with magnitude and phase channels"""
        # 2 channels: magnitude and phase
        grid = np.zeros((2, self.n_freq_bins, self.chunk_frames), dtype=np.float32)

        if not h5_path.exists():
            return grid

        start_frame = chunk_idx * self.chunk_frames
        end_frame = start_frame + self.chunk_frames

        with h5py.File(h5_path, 'r') as f:
            if 'c0' not in f:
                return grid

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

                # Add to grid
                for freq, amp, phase, frame in zip(chunk_freqs, chunk_amps, chunk_phases, chunk_frames_local):
                    freq_bin = int((freq - self.freq_min) / self.freq_bin_width)
                    freq_bin = np.clip(freq_bin, 0, self.n_freq_bins - 1)
                    time_idx = int(frame)

                    if 0 <= time_idx < self.chunk_frames:
                        # Convert to complex for proper phase accumulation
                        complex_val = amp * np.exp(1j * phase)

                        # Accumulate complex values (handles phase properly)
                        existing_mag = grid[0, freq_bin, time_idx]
                        existing_phase = grid[1, freq_bin, time_idx]
                        existing_complex = existing_mag * np.exp(1j * existing_phase)

                        new_complex = existing_complex + complex_val

                        # Store magnitude and phase
                        grid[0, freq_bin, time_idx] = np.abs(new_complex)
                        grid[1, freq_bin, time_idx] = np.angle(new_complex)

                offset += track_len

        return grid

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

        # Load fullmix as input (2 channels: magnitude, phase)
        fullmix_h5 = data_dir / 'fullmix_tracks.h5'
        if not fullmix_h5.exists():
            fullmix_h5 = data_dir / 'fullmix.h5'

        fullmix_grid = self.sinusoids_to_grid(fullmix_h5, chunk_idx)  # (2, freq_bins, time_frames)

        # Load each stem as target
        stem_grids = []
        for stem_name in self.stem_names:
            h5_files = self.find_stem_h5(data_dir, stem_name)

            # Merge grids if multiple files (e.g., drums)
            # Initialize with 2 channels (magnitude, phase)
            stem_grid = np.zeros((2, self.n_freq_bins, self.chunk_frames), dtype=np.float32)
            for h5_file in h5_files:
                file_grid = self.sinusoids_to_grid(h5_file, chunk_idx)

                # Add complex values for proper accumulation
                for f in range(self.n_freq_bins):
                    for t in range(self.chunk_frames):
                        # Convert both to complex
                        existing = stem_grid[0, f, t] * np.exp(1j * stem_grid[1, f, t])
                        new = file_grid[0, f, t] * np.exp(1j * file_grid[1, f, t])

                        # Add and convert back
                        combined = existing + new
                        stem_grid[0, f, t] = np.abs(combined)
                        stem_grid[1, f, t] = np.angle(combined)

            stem_grids.append(stem_grid)

        # Stack stems: (n_stems, 2, freq_bins, time_frames)
        stem_grids = np.stack(stem_grids, axis=0)

        # fullmix_grid already has shape (2, freq_bins, time_frames)
        return (
            torch.from_numpy(fullmix_grid),
            torch.from_numpy(stem_grids)
        )


class UNetStemSeparator(nn.Module):
    """
    U-Net architecture for stem separation with magnitude and phase channels.

    U-Net is the standard architecture for source separation tasks.
    It has an encoder-decoder structure with skip connections.

    Input: (2, freq_bins, time_frames) - magnitude and phase channels
    Output: (n_stems, 2, freq_bins, time_frames) - magnitude and phase per stem
    """

    def __init__(self, n_stems=5, base_channels=32):
        super().__init__()
        self.n_stems = n_stems

        # Encoder (downsampling path) - input has 2 channels (magnitude, phase)
        self.enc1 = self.conv_block(2, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)

        # Decoder (upsampling path)
        self.dec4 = self.conv_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = self.conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self.conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self.conv_block(base_channels * 2 + base_channels, base_channels)

        # Output layer - 2 channels (magnitude, phase) per stem
        self.output = nn.Conv2d(base_channels, n_stems * 2, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_channels, out_channels):
        """Standard convolutional block: Conv -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: (B, 2, H, W) where 2 = (magnitude, phase)

        # Encoder
        e1 = self.enc1(x)  # (B, 32, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 64, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 128, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 256, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 512, H/16, W/16)

        # Decoder with skip connections
        d4 = self.upsample(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upsample(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output: (B, n_stems * 2, H, W)
        out = self.output(d1)

        # Reshape to (B, n_stems, 2, H, W)
        batch_size, _, height, width = out.shape
        out = out.view(batch_size, self.n_stems, 2, height, width)

        # Apply constraints:
        # - Magnitude (channel 0): non-negative
        # - Phase (channel 1): wrap to [-pi, pi]
        out[:, :, 0, :, :] = F.relu(out[:, :, 0, :, :])  # magnitude >= 0
        out[:, :, 1, :, :] = torch.atan2(torch.sin(out[:, :, 1, :, :]),
                                         torch.cos(out[:, :, 1, :, :]))  # phase in [-pi, pi]

        return out


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for fullmix, stems in tqdm(dataloader, desc="Training"):
        fullmix = fullmix.to(device)
        stems = stems.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_stems = model(fullmix)

        # L1 loss (better for audio than L2)
        loss = F.l1_loss(pred_stems, stems)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', nargs='+', required=True)
    parser.add_argument('--stem-names', nargs='+',
                       default=['vocals', 'guitar', 'bass', 'drums', 'song'])
    parser.add_argument('--n-freq-bins', type=int, default=512,
                       help='Number of frequency bins (height of 2D grid)')
    parser.add_argument('--chunk-duration', type=float, default=4.0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base-channels', type=int, default=32)
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    chunk_frames = int(args.chunk_duration * 44100 / 512)

    # Dataset
    dataset = SinusoidalGrid2DDataset(
        args.data_dirs,
        args.stem_names,
        n_freq_bins=args.n_freq_bins,
        chunk_frames=chunk_frames
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # HDF5 doesn't support multiprocessing
    )

    # Model
    model = UNetStemSeparator(
        n_stems=len(args.stem_names),
        base_channels=args.base_channels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"2D grid shape: ({args.n_freq_bins}, {chunk_frames})")
    print(f"Stems: {args.stem_names}")

    # Training loop
    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, optimizer, device)
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
                    'n_freq_bins': args.n_freq_bins,
                    'chunk_frames': chunk_frames,
                    'base_channels': args.base_channels
                }
            }
            torch.save(checkpoint, f'stem_separator_2d_epoch{epoch+1}.pt')
            print(f"Saved checkpoint: stem_separator_2d_epoch{epoch+1}.pt")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
