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
    """Dataset that loads fullmix + stem audio and computes spectrograms"""

    def __init__(self, data_dir, stem_names, sample_rate=44100, n_fft=2048,
                 hop_length=512, segment_length=4.0):
        """
        Args:
            data_dir: Directory containing fullmix.ogg and stem .ogg files
            stem_names: List of stem names (e.g., ['vocals', 'guitar', 'bass', 'drums'])
            sample_rate: Audio sample rate
            n_fft: FFT size for spectrogram
            hop_length: Hop length for STFT
            segment_length: Length of audio segments in seconds
        """
        self.data_dir = Path(data_dir)
        self.stem_names = stem_names
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_samples = int(segment_length * sample_rate)

        # Load full audio files
        print("Loading audio files...")
        self.fullmix = self._load_audio(self.data_dir / 'fullmix.ogg')
        self.stems = {}

        for name in stem_names:
            if name == 'drums':
                # Drums are split into 4 files - combine them
                print(f"  Loading {name} (4 files)...")
                drum_files = [f'drums_{i}.ogg' for i in range(1, 5)]
                drums_combined = None

                for drum_file in drum_files:
                    drum_audio = self._load_audio(self.data_dir / drum_file)
                    if drums_combined is None:
                        drums_combined = drum_audio
                    else:
                        # Sum the drum tracks
                        drums_combined = drums_combined + drum_audio

                self.stems[name] = drums_combined
            else:
                print(f"  Loading {name}.ogg...")
                self.stems[name] = self._load_audio(self.data_dir / f'{name}.ogg')

        # Calculate number of segments
        self.n_segments = len(self.fullmix) // self.segment_samples
        print(f"Created {self.n_segments} segments from audio")

    def _load_audio(self, path):
        """Load audio file"""
        audio, sr = librosa.load(path, sr=self.sr, mono=True)
        return audio

    def _compute_spectrogram(self, audio):
        """Compute magnitude spectrogram"""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        # Convert to log scale
        log_mag = np.log1p(magnitude)
        return log_mag

    def __len__(self):
        return self.n_segments

    def __getitem__(self, idx):
        """Get a training segment"""
        start = idx * self.segment_samples
        end = start + self.segment_samples

        # Extract audio segments
        mix_segment = self.fullmix[start:end]
        stem_segments = {
            name: audio[start:end]
            for name, audio in self.stems.items()
        }

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

    def __init__(self, n_stems=4, in_channels=1, base_channels=32):
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

        # Loss: L1 distance between generated and ground truth stem spectrograms
        loss = criterion(generated_stems, stem_specs)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def main():
    config = {
        'stem_names': ['vocals', 'guitar', 'bass', 'drums'],
        'data_dir': '.',  # Current directory
        'sample_rate': 44100,
        'n_fft': 2048,
        'hop_length': 512,
        'segment_length': 4.0,  # 4 second segments
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("Generative Stem Separator Training")
    print(f"Device: {config['device']}")
    print(f"Stems: {config['stem_names']}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = StemGenerationDataset(
        data_dir=config['data_dir'],
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
        num_workers=2,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    model = UNetStemGenerator(
        n_stems=len(config['stem_names']),
        in_channels=1,
        base_channels=32
    ).to(config['device'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.L1Loss()  # L1 loss for spectrogram generation
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Training loop
    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
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
