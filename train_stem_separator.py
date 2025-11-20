#!/usr/bin/env python3
"""
Model 1: Stem Separator

Takes fullmix.h5 (ultra-precise synchrosqueeze data) and separates it into
stem grids (vocals, guitar, bass, drums).

Input: fullmix.h5 grid
Output: stem grids for vocals, guitar, bass, drums
Target: stem .h5 grids
Loss: L1 loss comparing generated vs target stem grids

Uses your precise synchrosqueeze data (NO STFT).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm


class StemSeparationDataset(Dataset):
    """Dataset that loads fullmix and stem .h5 grids for separation training"""

    def __init__(self, data_dirs, stem_names, segment_length_frames=344):
        """
        Args:
            data_dirs: List of directories containing preprocessed .npy files
            stem_names: List of stem names (e.g., ['vocals', 'guitar', 'bass', 'drums'])
            segment_length_frames: Number of time frames per segment
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dirs = [Path(d) for d in data_dirs]
        self.stem_names = stem_names
        self.segment_frames = segment_length_frames

        print(f"Loading preprocessed grids from {len(self.data_dirs)} song(s)...")
        self.segments = []

        for data_dir in self.data_dirs:
            print(f"\n  Processing {data_dir.name}...")

            # Check for preprocessed fullmix grid
            fullmix_spec_path = data_dir / 'fullmix_spec.npy'
            if not fullmix_spec_path.exists():
                print(f"    ERROR: Missing {fullmix_spec_path.name}")
                print(f"           Run: python preprocess_h5_to_spectrograms.py --data-dirs {data_dir}")
                continue

            # Load fullmix grid
            fullmix_spec = np.load(fullmix_spec_path)
            print(f"    Loaded fullmix: {fullmix_spec.shape}")

            # Load stem grids
            stem_specs = {}
            all_found = True
            for stem_name in stem_names:
                stem_spec_path = data_dir / f'{stem_name}_spec.npy'
                if not stem_spec_path.exists():
                    print(f"    ERROR: Missing {stem_spec_path.name}")
                    all_found = False
                    break
                stem_specs[stem_name] = np.load(stem_spec_path)
                print(f"    Loaded {stem_name}: {stem_specs[stem_name].shape}")

            if not all_found:
                continue

            # Create segments
            n_frames = fullmix_spec.shape[1]
            n_segments = n_frames // self.segment_frames

            print(f"    Creating {n_segments} segments...")

            for seg_idx in range(n_segments):
                start_frame = seg_idx * self.segment_frames
                end_frame = start_frame + self.segment_frames

                # Extract segment from fullmix
                fullmix_segment = fullmix_spec[:, start_frame:end_frame]

                # Extract segments from stems
                stem_segments = {
                    name: spec[:, start_frame:end_frame]
                    for name, spec in stem_specs.items()
                }

                self.segments.append({
                    'fullmix': fullmix_segment,
                    'stems': stem_segments,
                    'song': data_dir.name
                })

        print(f"\n  Total segments loaded: {len(self.segments)}")
        self.total_segments = len(self.segments)
        print(f"Total segments: {self.total_segments}")

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        """Get a training segment"""
        seg = self.segments[idx]

        # Get fullmix grid
        fullmix = seg['fullmix']  # [freq_bins, time_frames]

        # Stack stem grids
        stems = np.stack([
            seg['stems'][name] for name in self.stem_names
        ], axis=0)  # [n_stems, freq_bins, time_frames]

        return {
            'fullmix': torch.from_numpy(fullmix).float().unsqueeze(0),  # [1, freq, time]
            'stems': torch.from_numpy(stems).float(),  # [n_stems, freq, time]
        }


class UNetStemSeparator(nn.Module):
    """U-Net for separating fullmix grid into stem grids"""

    def __init__(self, n_stems=4, in_channels=1, base_channels=64):
        super().__init__()

        self.n_stems = n_stems

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        # Decoder
        self.dec4 = self._conv_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)

        # Output (generate all stems at once)
        self.out = nn.Conv2d(base_channels, n_stems, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.relu = nn.ReLU(inplace=True)

    def _conv_block(self, in_ch, out_ch):
        """Convolution block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _match_size(self, x, target):
        """Match tensor size to target by padding or cropping"""
        if x.shape[2] > target.shape[2]:
            diff = x.shape[2] - target.shape[2]
            x = x[:, :, diff//2:diff//2 + target.shape[2], :]
        elif x.shape[2] < target.shape[2]:
            diff = target.shape[2] - x.shape[2]
            x = nn.functional.pad(x, (0, 0, diff//2, diff - diff//2))

        if x.shape[3] > target.shape[3]:
            diff = x.shape[3] - target.shape[3]
            x = x[:, :, :, diff//2:diff//2 + target.shape[3]]
        elif x.shape[3] < target.shape[3]:
            diff = target.shape[3] - x.shape[3]
            x = nn.functional.pad(x, (diff//2, diff - diff//2, 0, 0))

        return x

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.upsample(b)
        d4 = self._match_size(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upsample(d4)
        d3 = self._match_size(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upsample(d3)
        d2 = self._match_size(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upsample(d2)
        d1 = self._match_size(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output: [batch, n_stems, freq, time]
        output = self.out(d1)
        output = self.relu(output)

        return output


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        fullmix = batch['fullmix'].to(device)
        stems = batch['stems'].to(device)

        optimizer.zero_grad()

        # Forward pass: separate fullmix into stems
        generated_stems = model(fullmix)

        # Resize if needed
        if generated_stems.shape != stems.shape:
            generated_stems = torch.nn.functional.interpolate(
                generated_stems,
                size=stems.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Loss: L1 distance between generated and target stem grids
        loss = criterion(generated_stems, stems)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train stem separator (Model 1)')
    parser.add_argument('--data-dirs', nargs='+', default=['.'])
    parser.add_argument('--stem-names', nargs='+', default=['vocals', 'guitar', 'bass', 'drums'])
    parser.add_argument('--segment-length-frames', type=int, default=344)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_separator')

    args = parser.parse_args()

    # Config
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'stem_names': args.stem_names,
        'segment_length_frames': args.segment_length_frames,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
    }

    print("=" * 60)
    print("Model 1: Stem Separator Training")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Stems: {config['stem_names']}")
    print(f"Segment length: {config['segment_length_frames']} frames")

    # Create dataset
    print("\nCreating dataset...")
    dataset = StemSeparationDataset(
        args.data_dirs,
        config['stem_names'],
        segment_length_frames=config['segment_length_frames']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    model = UNetStemSeparator(
        n_stems=len(config['stem_names']),
        in_channels=1,
        base_channels=64
    )
    model = model.to(config['device'])

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.L1Loss()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, dataloader, optimizer, criterion, config['device'])
        print(f"Train Loss: {train_loss:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
