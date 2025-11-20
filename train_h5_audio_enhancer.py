#!/usr/bin/env python3
"""
Model 2: H5 Audio Enhancer

Takes synthesized audio from .h5 files (which has artifacts) and enhances it
to sound like the clean .ogg files.

Architecture: 1D Waveform U-Net (operates on raw audio samples)
- Input: Synthesized audio waveform from .h5
- Output: Enhanced clean audio waveform
- Target: Clean .ogg audio
- Loss: L1 waveform reconstruction loss

NO SPECTROGRAMS. Just raw waveforms.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


class H5AudioDataset(Dataset):
    """Dataset that synthesizes audio from .h5 and pairs it with clean .ogg"""

    def __init__(self, data_dirs, stem_names, sample_rate=44100, segment_length=4.0):
        """
        Args:
            data_dirs: List of directories containing .h5 and .ogg files
            stem_names: List of stem names (e.g., ['vocals', 'guitar', 'bass', 'drums'])
            sample_rate: Audio sample rate
            segment_length: Length of audio segments in seconds
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dirs = [Path(d) for d in data_dirs]
        self.stem_names = stem_names
        self.sr = sample_rate
        self.segment_samples = int(segment_length * sample_rate)

        print(f"Loading data from {len(self.data_dirs)} song(s)...")
        self.segments = []

        for data_dir in self.data_dirs:
            print(f"\n  Processing {data_dir.name}...")

            for stem_name in stem_names:
                # Handle different .h5 naming conventions
                if stem_name == 'drums':
                    # Drums are split into multiple files
                    h5_paths_list = [data_dir / f'drums_{i}_tracks.h5' for i in range(1, 5)]
                    h5_paths_list = [p for p in h5_paths_list if p.exists()]
                    # Combine drum .ogg files
                    ogg_paths = [data_dir / f'drums_{i}.ogg' for i in range(1, 5)]
                    ogg_paths = [p for p in ogg_paths if p.exists()]
                    # For drums, process each .h5 file separately
                    for drum_idx, h5_path in enumerate(h5_paths_list, 1):
                        drum_ogg = data_dir / f'drums_{drum_idx}.ogg'
                        if h5_path.exists() and drum_ogg.exists():
                            self._load_stem_data(h5_path, [drum_ogg], f'drums_{drum_idx}')
                    continue  # Skip the general processing below for drums

                elif stem_name == 'bass':
                    h5_path = data_dir / 'bass_tracks.h5'
                    bass_ogg = data_dir / 'bass.ogg'
                    rhythm_ogg = data_dir / 'rhythm.ogg'
                    ogg_paths = [bass_ogg] if bass_ogg.exists() else ([rhythm_ogg] if rhythm_ogg.exists() else [])
                else:
                    h5_path = data_dir / f'{stem_name}_tracks.h5'
                    ogg_path = data_dir / f'{stem_name}.ogg'
                    ogg_paths = [ogg_path] if ogg_path.exists() else []

                # Process non-drum stems
                if h5_path.exists() and ogg_paths:
                    self._load_stem_data(h5_path, ogg_paths, stem_name)

        print(f"\n  Total segments loaded: {len(self.segments)}")

        self.total_segments = len(self.segments)
        print(f"\nTotal segments: {self.total_segments}")

    def _load_stem_data(self, h5_path, ogg_paths, stem_name):
        """Load and segment a single stem's .h5 and .ogg data"""

        # Look for pre-synthesized file
        synth_path = h5_path.parent / h5_path.name.replace('_tracks.h5', '_synthesized.wav')

        if not synth_path.exists():
            print(f"    ERROR: Missing synthesized file: {synth_path.name}")
            print(f"           Run: python presynthesize_h5_files.py --data-dirs {h5_path.parent}")
            return

        print(f"    Found {stem_name}: {synth_path.name} + {len(ogg_paths)} .ogg file(s)")

        # Load synthesized audio
        synth_audio, sr = librosa.load(synth_path, sr=self.sr, mono=True)

        # Load clean audio
        clean_audio = None
        for ogg_path in ogg_paths:
            audio, sr = librosa.load(ogg_path, sr=self.sr, mono=True)
            if clean_audio is None:
                clean_audio = audio
            else:
                clean_audio = clean_audio + audio

        # Match lengths (use shorter of the two)
        min_len = min(len(synth_audio), len(clean_audio))
        synth_audio = synth_audio[:min_len]
        clean_audio = clean_audio[:min_len]

        # Calculate number of segments
        n_segments = min_len // self.segment_samples
        print(f"      Audio length: {min_len/self.sr:.2f}s → {n_segments} segments")

        # Store segment info
        for seg_idx in range(n_segments):
            self.segments.append({
                'synth_audio': synth_audio,
                'clean_audio': clean_audio,
                'segment_idx': seg_idx,
                'stem_name': stem_name
            })

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        """Get a training segment"""
        seg = self.segments[idx]

        # Get segment bounds
        start = seg['segment_idx'] * self.segment_samples
        end = start + self.segment_samples

        # Get segments from pre-loaded audio
        synth_segment = seg['synth_audio'][start:end]
        clean_segment = seg['clean_audio'][start:end]

        # Ensure same length (pad if needed)
        if len(synth_segment) < self.segment_samples:
            synth_segment = np.pad(synth_segment, (0, self.segment_samples - len(synth_segment)))
        if len(clean_segment) < self.segment_samples:
            clean_segment = np.pad(clean_segment, (0, self.segment_samples - len(clean_segment)))

        return {
            'synth_audio': torch.from_numpy(synth_segment).float().unsqueeze(0),  # [1, samples]
            'clean_audio': torch.from_numpy(clean_segment).float().unsqueeze(0),  # [1, samples]
        }


class WaveformUNet(nn.Module):
    """1D U-Net for waveform-to-waveform audio enhancement"""

    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels, kernel_size=15)
        self.enc2 = self._conv_block(base_channels, base_channels * 2, kernel_size=15)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4, kernel_size=15)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8, kernel_size=15)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16, kernel_size=15)

        # Decoder
        self.dec4 = self._conv_block(base_channels * 16 + base_channels * 8, base_channels * 8, kernel_size=15)
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4, kernel_size=15)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2, kernel_size=15)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels, kernel_size=15)

        # Output
        self.out = nn.Conv1d(base_channels, out_channels, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.upsample = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)

    def _conv_block(self, in_ch, out_ch, kernel_size):
        """Convolution block with BatchNorm and LeakyReLU"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _match_size(self, x, target):
        """Match tensor size to target by padding or cropping"""
        if x.shape[2] > target.shape[2]:
            diff = x.shape[2] - target.shape[2]
            x = x[:, :, diff//2:diff//2 + target.shape[2]]
        elif x.shape[2] < target.shape[2]:
            diff = target.shape[2] - x.shape[2]
            x = nn.functional.pad(x, (diff//2, diff - diff//2))
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

        # Output
        out = self.out(d1)
        return out


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        synth_audio = batch['synth_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)

        optimizer.zero_grad()

        # Forward pass: enhance synthesized audio
        enhanced_audio = model(synth_audio)

        # Ensure same size
        if enhanced_audio.shape != clean_audio.shape:
            enhanced_audio = enhanced_audio[:, :, :clean_audio.shape[2]]

        # Loss: L1 waveform reconstruction
        loss = criterion(enhanced_audio, clean_audio)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(dataloader)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train H5 audio enhancer (Model 2)')
    parser.add_argument('--data-dirs', nargs='+', default=['.'])
    parser.add_argument('--stem-names', nargs='+', default=['vocals', 'guitar', 'bass', 'drums'])
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--segment-length', type=float, default=4.0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_enhancer')

    args = parser.parse_args()

    # Config
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'stem_names': args.stem_names,
        'sample_rate': args.sample_rate,
        'segment_length': args.segment_length,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
    }

    print("=" * 60)
    print("Model 2: H5 Audio Enhancer Training")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Stems: {config['stem_names']}")
    print(f"Sample rate: {config['sample_rate']} Hz")
    print(f"Segment length: {config['segment_length']}s")

    # Create dataset
    print("\nCreating dataset...")
    dataset = H5AudioDataset(
        args.data_dirs,
        config['stem_names'],
        sample_rate=config['sample_rate'],
        segment_length=config['segment_length']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Start with 0, increase if needed
        pin_memory=True if config['device'] == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    model = WaveformUNet(in_channels=1, out_channels=1, base_channels=32)
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
        print(f"Train Loss: {train_loss:.6f}")

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
