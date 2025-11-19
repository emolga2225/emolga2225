#!/usr/bin/env python3
"""
Generative Stem Separator - Proper Architecture

Input: Fullmix spectrogram + sinusoidal conditioning
Output: Generated stem spectrograms (vocals, guitar, bass, drums)
Loss: L1 loss comparing generated vs ground truth stem spectrograms

This learns to GENERATE stems, not just classify/rearrange sinusoids.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

class StemGenerationDataset(Dataset):
    """Dataset that loads fullmix + stem audio from multiple songs"""

    def __init__(self, data_dirs, stem_names, sample_rate=44100, n_fft=2048,
                 hop_length=512, segment_length=4.0, augment=True):
        """
        Args:
            data_dirs: List of directories, each containing fullmix.ogg and stem .ogg files
            stem_names: List of stem names (e.g., ['vocals', 'guitar', 'bass', 'drums'])
            sample_rate: Audio sample rate
            n_fft: FFT size for spectrogram
            hop_length: Hop length for STFT
            segment_length: Length of audio segments in seconds
            augment: Whether to apply data augmentation to prevent overfitting
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dirs = [Path(d) for d in data_dirs]
        self.stem_names = stem_names
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_samples = int(segment_length * sample_rate)
        self.augment = augment

        # Load audio from all songs
        print(f"Loading audio files from {len(self.data_dirs)} song(s)...")
        self.songs = []

        for data_dir in self.data_dirs:
            print(f"\n  Loading from {data_dir.name}...")
            fullmix = self._load_audio(data_dir / 'fullmix.ogg')
            stems = {}

            for name in stem_names:
                if name == 'drums':
                    # Drums are split into 4 files - combine them
                    drum_files = [f'drums_{i}.ogg' for i in range(1, 5)]
                    drums_combined = None

                    for drum_file in drum_files:
                        drum_path = data_dir / drum_file
                        if drum_path.exists():
                            drum_audio = self._load_audio(drum_path)
                            if drums_combined is None:
                                drums_combined = drum_audio
                            else:
                                drums_combined = drums_combined + drum_audio

                    if drums_combined is not None:
                        stems[name] = drums_combined
                elif name == 'bass':
                    # Check for both bass.ogg and rhythm.ogg
                    bass_path = data_dir / 'bass.ogg'
                    rhythm_path = data_dir / 'rhythm.ogg'
                    if bass_path.exists():
                        stems[name] = self._load_audio(bass_path)
                    elif rhythm_path.exists():
                        stems[name] = self._load_audio(rhythm_path)
                else:
                    stem_path = data_dir / f'{name}.ogg'
                    if stem_path.exists():
                        stems[name] = self._load_audio(stem_path)

            # Calculate segments for this song
            n_segments = fullmix.shape[1] // self.segment_samples

            self.songs.append({
                'fullmix': fullmix,
                'stems': stems,
                'n_segments': n_segments,
                'dir': data_dir
            })

            print(f"    Loaded {n_segments} segments")

        # Calculate total segments across all songs
        self.total_segments = sum(song['n_segments'] for song in self.songs)
        print(f"\nTotal segments across all songs: {self.total_segments}")

    def _load_audio(self, path):
        """Load audio file as stereo"""
        audio, sr = librosa.load(path, sr=self.sr, mono=False)
        # If mono, duplicate to stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        return audio

    def _compute_spectrogram(self, audio):
        """Compute magnitude spectrogram from stereo audio"""
        # Average left and right channels for spectrogram
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        stft = librosa.stft(audio_mono, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        # Convert to log scale
        log_mag = np.log1p(magnitude)
        return log_mag

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        """Get a training segment from any song"""
        # Find which song this segment belongs to
        cumulative = 0
        for song in self.songs:
            if idx < cumulative + song['n_segments']:
                # This segment is from this song
                segment_idx = idx - cumulative
                start = segment_idx * self.segment_samples
                end = start + self.segment_samples

                # Extract audio segments (handle stereo: [2, samples])
                mix_segment = song['fullmix'][:, start:end]
                stem_segments = {
                    name: audio[:, start:end]
                    for name, audio in song['stems'].items()
                }
                break
            cumulative += song['n_segments']
        else:
            raise IndexError(f"Segment index {idx} out of range")

        # Compute spectrograms
        mix_spec = self._compute_spectrogram(mix_segment)
        stem_specs = {
            name: self._compute_spectrogram(audio)
            for name, audio in stem_segments.items()
        }

        # Stack stem spectrograms: [n_stems, freq_bins, time_frames]
        stem_specs_array = np.stack([
            stem_specs[name] for name in self.stem_names
        ], axis=0)

        return {
            'mix_spec': torch.from_numpy(mix_spec).float().unsqueeze(0),  # [1, freq, time]
            'stem_specs': torch.from_numpy(stem_specs_array).float(),  # [n_stems, freq, time]
        }


class UNetStemGenerator(nn.Module):
    """U-Net architecture for generating stem spectrograms from fullmix"""

    def __init__(self, n_stems=4, in_channels=1, base_channels=64):
        """
        Args:
            n_stems: Number of stems to generate
            in_channels: Input channels (1 for mono spectrogram)
            base_channels: Base number of channels (increased to 64 for more capacity)
        """
        super().__init__()

        self.n_stems = n_stems

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels*2)
        self.enc3 = self._conv_block(base_channels*2, base_channels*4)
        self.enc4 = self._conv_block(base_channels*4, base_channels*8)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels*8, base_channels*16)

        # Decoder (upsampling)
        self.dec4 = self._up_conv_block(base_channels*16, base_channels*8)
        self.dec3 = self._up_conv_block(base_channels*16, base_channels*4)  # *16 due to skip connection
        self.dec2 = self._up_conv_block(base_channels*8, base_channels*2)
        self.dec1 = self._up_conv_block(base_channels*4, base_channels)

        # Output layer - one spectrogram per stem
        self.output = nn.Conv2d(base_channels*2, n_stems, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def _conv_block(self, in_ch, out_ch):
        """Convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _up_conv_block(self, in_ch, out_ch):
        """Upsampling block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _match_size(self, x, target):
        """Match x size to target size by center cropping or padding"""
        _, _, h_x, w_x = x.shape
        _, _, h_t, w_t = target.shape

        # Crop or pad height
        if h_x > h_t:
            diff = h_x - h_t
            x = x[:, :, diff//2:diff//2 + h_t, :]
        elif h_x < h_t:
            diff = h_t - h_x
            x = torch.nn.functional.pad(x, (0, 0, diff//2, diff - diff//2))

        # Crop or pad width
        if w_x > w_t:
            diff = w_x - w_t
            x = x[:, :, :, diff//2:diff//2 + w_t]
        elif w_x < w_t:
            diff = w_t - w_x
            x = torch.nn.functional.pad(x, (diff//2, diff - diff//2, 0, 0))

        return x

    def forward(self, x):
        """
        Args:
            x: Input fullmix spectrogram [batch, 1, freq, time]

        Returns:
            Generated stem spectrograms [batch, n_stems, freq, time]
        """
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections (match sizes before concatenating)
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat([dec4, self._match_size(enc4, dec4)], dim=1)

        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, self._match_size(enc3, dec3)], dim=1)

        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, self._match_size(enc2, dec2)], dim=1)

        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, self._match_size(enc1, dec1)], dim=1)

        # Output layer
        output = self.output(dec1)

        # ReLU to ensure positive spectrograms
        output = self.relu(output)

        return output


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        mix_spec = batch['mix_spec'].to(device)
        stem_specs = batch['stem_specs'].to(device)

        optimizer.zero_grad()

        # Forward pass: generate stem spectrograms
        generated_stems = model(mix_spec)

        # Resize generated stems to match target size (handles slight dimension mismatches from U-Net)
        if generated_stems.shape != stem_specs.shape:
            generated_stems = torch.nn.functional.interpolate(
                generated_stems,
                size=stem_specs.shape[2:],  # Match [freq, time] dimensions
                mode='bilinear',
                align_corners=False
            )

        # Loss: L1 distance between generated and ground truth stem spectrograms
        loss = criterion(generated_stems, stem_specs)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train generative stem separator')
    parser.add_argument('--data-dirs', nargs='+', default=['.'],
                       help='Directories containing song data')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    args = parser.parse_args()

    config = {
        'stem_names': ['vocals', 'guitar', 'bass', 'drums'],
        'data_dirs': args.data_dirs,
        'sample_rate': 44100,
        'n_fft': 2048,
        'hop_length': 512,
        'segment_length': 4.0,
        'batch_size': args.batch_size,
        'learning_rate': 1e-4,
        'num_epochs': args.epochs,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'base_channels': 64  # Increased capacity
    }

    print("Generative Stem Separator Training")
    print(f"Device: {config['device']}")
    print(f"Stems: {config['stem_names']}")
    print(f"Data directories: {config['data_dirs']}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = StemGenerationDataset(
        data_dirs=config['data_dirs'],
        stem_names=config['stem_names'],
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        segment_length=config['segment_length']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Single-threaded to avoid deadlock with audio loading
        pin_memory=True if config['device'] == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    model = UNetStemGenerator(
        n_stems=len(config['stem_names']),
        in_channels=1,
        base_channels=config['base_channels']
    ).to(config['device'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss = train_epoch(model, dataloader, optimizer, criterion, config['device'])

        print(f"Train Loss: {train_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'train_loss': train_loss
            }
            torch.save(checkpoint, f'stem_generator_epoch_{epoch+1}.pt')
            print(f"Saved checkpoint: stem_generator_epoch_{epoch+1}.pt")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'stem_generator_final.pt')
    print("\nTraining complete! Saved final model: stem_generator_final.pt")


if __name__ == "__main__":
    main()
