#!/usr/bin/env python3
"""
Generative Stem Separator - Using Custom Synchrosqueeze Data

Input: fullmix.h5 (ultra-precise synchrosqueeze + filtered data, better than STFT)
Auxiliary: stem .h5 files (harmonic content/relationships - algorithmically perfect)
Target: stem .ogg spectrograms (what it should actually sound like)

Loss:
  - Primary: L1 loss vs stem audio spectrograms
  - Auxiliary: L1 loss vs stem .h5 harmonic data (weighted 0.3x)

This learns to GENERATE stems using precise frequency data as input and guidance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import h5py
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

        # Fixed time dimension for all segments (ensures consistent batch sizes)
        self.expected_time_frames = self.segment_samples // self.hop_length

        # Load audio from all songs
        print(f"Loading audio files from {len(self.data_dirs)} song(s)...")
        self.songs = []

        for data_dir in self.data_dirs:
            print(f"\n  Loading from {data_dir.name}...")

            # Check if fullmix.h5 exists for ultra-precise synchrosqueeze data
            fullmix_h5_path = data_dir / 'fullmix.h5'
            if not fullmix_h5_path.exists():
                fullmix_h5_path = None  # Will use fullmix.ogg spectrogram as fallback

            # Load fullmix audio (used for length calculation and as fallback input)
            fullmix = self._load_audio(data_dir / 'fullmix.ogg')
            stems = {}

            # Store paths to sinusoidal .h5 files (lazy loading for memory efficiency)
            # .h5 files are 200MB-1.2GB each, so we can't load them all upfront
            h5_paths = {}

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

                # Store .h5 file path for lazy loading
                h5_path = data_dir / f'{name}.h5'
                if h5_path.exists():
                    h5_paths[name] = h5_path

            # Calculate segments for this song
            n_segments = fullmix.shape[1] // self.segment_samples

            self.songs.append({
                'fullmix': fullmix,  # Used for length calculation and as fallback input if .h5 missing
                'fullmix_h5_path': fullmix_h5_path,  # Ultra-precise synchrosqueeze data (input, if available)
                'stems': stems,  # Target audio
                'h5_paths': h5_paths,  # Stem harmonic content (auxiliary guidance)
                'n_segments': n_segments,
                'dir': data_dir
            })

            print(f"    Loaded {n_segments} segments")

        # Calculate total segments across all songs
        self.total_segments = sum(song['n_segments'] for song in self.songs)
        print(f"\nTotal segments across all songs: {self.total_segments}")

        # Cache for open .h5 files (per worker process)
        # Keeps files open to avoid repeated open/close overhead
        self._h5_cache = {}

    def _get_h5_file(self, h5_path):
        """Get or open an .h5 file and cache it to avoid repeated open/close"""
        h5_path_str = str(h5_path)
        if h5_path_str not in self._h5_cache:
            self._h5_cache[h5_path_str] = h5py.File(h5_path, 'r')
        return self._h5_cache[h5_path_str]

    def _load_audio(self, path):
        """Load audio file as stereo"""
        audio, sr = librosa.load(path, sr=self.sr, mono=False)
        # If mono, duplicate to stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        return audio

    def _load_sinusoidal_segment_h5(self, h5_path, start_sample, end_sample):
        """Load only a specific time segment from sinusoidal .h5 file

        This provides the 'reverse engineered' precise frequency data
        that makes training more detailed than Spleeter.

        Uses lazy loading - only reads the sinusoidal tracks that overlap
        with the requested time range instead of loading the entire file.
        """
        # Calculate frame range for this segment
        start_frame = start_sample // self.hop_length
        end_frame = end_sample // self.hop_length
        n_frames = end_frame - start_frame
        n_bins = 1 + (self.n_fft // 2)

        # Initialize spectrogram for this segment only
        spec = np.zeros((n_bins, n_frames), dtype=np.float32)

        # Use cached .h5 file (keeps file open instead of repeated open/close)
        f = self._get_h5_file(h5_path)

        # Process all channels and average them
        n_channels = f.attrs.get('ch', 1)

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            # Load track metadata to find tracks in this time range
            track_lens = grp['len'][:]
            track_starts = grp['s'][:]
            track_ends = grp['e'][:]

            # Load full data arrays (still need these to index properly)
            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            frame_indices = grp['i'][:]

            # Process only tracks that overlap with our segment
            offset = 0
            for track_idx, track_len in enumerate(track_lens):
                track_start = track_starts[track_idx]
                track_end = track_ends[track_idx]

                # Check if this track overlaps with our segment
                if track_end >= start_frame and track_start < end_frame:
                    track_freqs = frequencies[offset:offset + track_len]
                    track_amps = amplitudes[offset:offset + track_len]
                    track_frames = frame_indices[offset:offset + track_len]

                    # Add sinusoids that fall within our segment
                    for freq, amp, frame in zip(track_freqs, track_amps, track_frames):
                        if start_frame <= frame < end_frame:
                            # Convert to local frame index (relative to segment start)
                            local_frame = frame - start_frame
                            # Convert frequency to bin index
                            bin_idx = int(freq * self.n_fft / self.sr)
                            if 0 <= bin_idx < n_bins and 0 <= local_frame < n_frames:
                                spec[bin_idx, local_frame] += amp

                offset += track_len

        # Average across channels
        if n_channels > 0:
            spec = spec / n_channels

        # Convert to log scale like audio spectrograms
        log_spec = np.log1p(spec)
        return log_spec

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

    def _resize_spectrogram(self, spec, target_time_frames):
        """Resize spectrogram to target time dimension by padding or cropping"""
        current_frames = spec.shape[1]

        if current_frames == target_time_frames:
            return spec
        elif current_frames < target_time_frames:
            # Pad with zeros
            pad_width = ((0, 0), (0, target_time_frames - current_frames))
            return np.pad(spec, pad_width, mode='constant', constant_values=0)
        else:
            # Crop
            return spec[:, :target_time_frames]

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

                # Load fullmix input (prefer .h5 if available, else compute from audio)
                if song['fullmix_h5_path'] is not None:
                    # Use ultra-precise synchrosqueeze data from .h5
                    mix_spec = self._load_sinusoidal_segment_h5(
                        song['fullmix_h5_path'], start, end
                    )
                else:
                    # Fallback: compute spectrogram from fullmix audio
                    fullmix_segment = song['fullmix'][:, start:end]
                    mix_spec = self._compute_spectrogram(fullmix_segment)

                # Extract stem audio segments (target output)
                stem_segments = {
                    name: audio[:, start:end]
                    for name, audio in song['stems'].items()
                }

                # Lazy load stem .h5 segments (harmonic content guidance)
                # (avoids loading 200MB-1.2GB .h5 files all at once)
                sinusoidal_segments = {}
                for name, h5_path in song['h5_paths'].items():
                    sinusoidal_segments[name] = self._load_sinusoidal_segment_h5(
                        h5_path, start, end
                    )
                break
            cumulative += song['n_segments']
        else:
            raise IndexError(f"Segment index {idx} out of range")

        # Compute spectrograms from stem audio (target output)
        # mix_spec already loaded (from fullmix.h5 if available, else from fullmix.ogg)
        stem_specs = {
            name: self._compute_spectrogram(audio)
            for name, audio in stem_segments.items()
        }

        # Ensure ALL spectrograms have the FIXED expected time dimension
        # (handles rounding differences between .h5 and audio STFT)
        # This ensures consistent batch sizes
        target_time_frames = self.expected_time_frames

        # Resize mix_spec to fixed dimension
        if mix_spec.shape[1] != target_time_frames:
            mix_spec = self._resize_spectrogram(mix_spec, target_time_frames)

        # Resize stem specs to fixed dimension
        for name in stem_specs:
            if stem_specs[name].shape[1] != target_time_frames:
                stem_specs[name] = self._resize_spectrogram(stem_specs[name], target_time_frames)

        # Resize sinusoidal specs to fixed dimension
        for name in sinusoidal_segments:
            if sinusoidal_segments[name].shape[1] != target_time_frames:
                sinusoidal_segments[name] = self._resize_spectrogram(sinusoidal_segments[name], target_time_frames)

        # Stack stem spectrograms: [n_stems, freq_bins, time_frames]
        stem_specs_array = np.stack([
            stem_specs[name] for name in self.stem_names
        ], axis=0)

        # Stack sinusoidal spectrograms (the "reverse engineered" precise data)
        sinusoidal_specs_array = np.stack([
            sinusoidal_segments.get(name, np.zeros_like(mix_spec))
            for name in self.stem_names
        ], axis=0)

        return {
            'mix_spec': torch.from_numpy(mix_spec).float().unsqueeze(0),  # [1, freq, time]
            'stem_specs': torch.from_numpy(stem_specs_array).float(),  # [n_stems, freq, time]
            'sinusoidal_specs': torch.from_numpy(sinusoidal_specs_array).float(),  # [n_stems, freq, time]
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
        sinusoidal_specs = batch['sinusoidal_specs'].to(device)

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

        # Multi-objective loss:
        # 1. Primary: Match audio spectrograms (what it should sound like)
        audio_loss = criterion(generated_stems, stem_specs)

        # 2. Auxiliary: Match sinusoidal spectrograms (precise frequency detail)
        #    This is the "reverse engineered" data that makes it better than Spleeter
        sinusoidal_loss = criterion(generated_stems, sinusoidal_specs)

        # Combined loss (weight sinusoidal loss lower since it's auxiliary guidance)
        loss = audio_loss + 0.3 * sinusoidal_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'audio': f'{audio_loss.item():.4f}', 'sin': f'{sinusoidal_loss.item():.4f}'})

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
        num_workers=4,  # Parallel data loading to keep GPU fed
        persistent_workers=True,  # Keep workers alive to avoid deadlock
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
    criterion = nn.L1Loss(reduction='mean')
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
