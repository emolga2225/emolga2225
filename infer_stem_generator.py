#!/usr/bin/env python3
"""
Inference script for the generative stem separator.

Usage:
    python infer_stem_generator.py --checkpoint stem_generator_epoch_320.pt --input ajfa/fullmix.ogg --output output/

This will generate:
    output/vocals.wav
    output/guitar.wav
    output/bass.wav
    output/drums.wav
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from tqdm import tqdm


class UNetStemGenerator(nn.Module):
    """U-Net architecture for generating stem spectrograms from fullmix"""

    def __init__(self, n_stems=4, in_channels=1, base_channels=64):
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
        self.dec3 = self._up_conv_block(base_channels*16, base_channels*4)
        self.dec2 = self._up_conv_block(base_channels*8, base_channels*2)
        self.dec1 = self._up_conv_block(base_channels*4, base_channels)

        # Output layer - one spectrogram per stem
        self.output = nn.Conv2d(base_channels*2, n_stems, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _up_conv_block(self, in_ch, out_ch):
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
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
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


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']

    model = UNetStemGenerator(
        n_stems=len(config['stem_names']),
        in_channels=1,
        base_channels=config['base_channels']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Stems: {config['stem_names']}")

    return model, config


def process_audio(audio_path, model, config, device='cuda', chunk_length=4.0):
    """Process full audio file through the model in chunks"""
    print(f"\nProcessing: {audio_path}")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=config['sample_rate'], mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    print(f"Audio length: {audio.shape[1] / sr:.2f} seconds")

    n_fft = config['n_fft']
    hop_length = config['hop_length']
    chunk_samples = int(chunk_length * sr)

    # Process in chunks to avoid memory issues
    n_chunks = int(np.ceil(audio.shape[1] / chunk_samples))

    # Initialize output stems
    stem_audio = {name: np.zeros_like(audio) for name in config['stem_names']}

    print(f"Processing {n_chunks} chunks...")

    with torch.no_grad():
        for i in tqdm(range(n_chunks)):
            start = i * chunk_samples
            end = min(start + chunk_samples, audio.shape[1])

            # Get chunk
            chunk = audio[:, start:end]

            # Pad last chunk if needed
            if chunk.shape[1] < chunk_samples:
                chunk = np.pad(chunk, ((0, 0), (0, chunk_samples - chunk.shape[1])))

            # Compute spectrogram
            chunk_mono = np.mean(chunk, axis=0)
            stft = librosa.stft(chunk_mono, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            log_mag = np.log1p(magnitude)

            # Convert to tensor
            mix_spec = torch.from_numpy(log_mag).float().unsqueeze(0).unsqueeze(0).to(device)

            # Generate stems
            generated_stems = model(mix_spec)

            # Convert back to audio for each stem
            for stem_idx, stem_name in enumerate(config['stem_names']):
                # Get stem spectrogram
                stem_spec = generated_stems[0, stem_idx].cpu().numpy()

                # Match size to original magnitude
                if stem_spec.shape != magnitude.shape:
                    # Resize using interpolation
                    from scipy.ndimage import zoom
                    zoom_factors = (magnitude.shape[0] / stem_spec.shape[0],
                                   magnitude.shape[1] / stem_spec.shape[1])
                    stem_spec = zoom(stem_spec, zoom_factors, order=1)

                # Convert from log scale back to linear
                stem_magnitude = np.expm1(stem_spec)

                # Use original phase (simple approach - could use Griffin-Lim for better quality)
                stem_stft = stem_magnitude * np.exp(1j * phase)

                # ISTFT to get audio
                stem_chunk_mono = librosa.istft(stem_stft, hop_length=hop_length, length=chunk.shape[1])

                # Convert back to stereo (simple duplication - could use more sophisticated approach)
                stem_chunk_stereo = np.stack([stem_chunk_mono, stem_chunk_mono])

                # Add to output
                stem_audio[stem_name][:, start:end] = stem_chunk_stereo[:, :end-start]

    return stem_audio


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained stem separator')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint (e.g., stem_generator_epoch_320.pt)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input audio file (e.g., ajfa/fullmix.ogg)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for separated stems')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, config = load_model(args.checkpoint, device=device)

    # Process audio
    stem_audio = process_audio(args.input, model, config, device=device)

    # Save stems
    print("\nSaving stems...")
    for stem_name, audio in stem_audio.items():
        output_path = output_dir / f'{stem_name}.wav'
        # Transpose to (samples, channels) for soundfile
        audio_t = audio.T
        sf.write(output_path, audio_t, config['sample_rate'])
        print(f"  Saved: {output_path}")

    print("\nDone! Stems saved to:", output_dir)


if __name__ == "__main__":
    main()
