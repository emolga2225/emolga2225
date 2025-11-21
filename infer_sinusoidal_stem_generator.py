#!/usr/bin/env python3
"""
Inference script for sinusoidal stem separator.

Synthesizes audio directly from generated sinusoids - no FFT/ISTFT.

Usage:
    python infer_sinusoidal_stem_generator.py --checkpoint sinusoidal_stem_generator_epoch_100.pt --input song.h5 --output output/
"""

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
from tqdm import tqdm
import h5py


class SinusoidalTransformer(nn.Module):
    """Transformer that generates stem sinusoids from fullmix sinusoids"""

    def __init__(self, n_stems=4, max_input_sinusoids=2000, max_output_per_stem=500,
                 d_model=256, nhead=8, num_layers=6):
        super().__init__()

        self.n_stems = n_stems
        self.max_input_sinusoids = max_input_sinusoids
        self.max_output_per_stem = max_output_per_stem
        self.d_model = d_model

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Positional encoding
        self.pos_encoder = nn.Embedding(max_input_sinusoids, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Stem-specific decoders
        self.stem_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, max_output_per_stem * 3)
            )
            for _ in range(n_stems)
        ])

    def forward(self, x):
        batch_size = x.shape[0]

        # Embed input sinusoids
        x_embed = self.input_embed(x)

        # Add positional encoding
        positions = torch.arange(self.max_input_sinusoids, device=x.device)
        pos_embed = self.pos_encoder(positions)
        x_embed = x_embed + pos_embed.unsqueeze(0)

        # Encode
        encoded = self.transformer_encoder(x_embed)

        # Global pooling
        pooled = encoded.mean(dim=1)

        # Generate sinusoids for each stem
        stem_outputs = []
        for decoder in self.stem_decoders:
            stem_sines = decoder(pooled)
            stem_sines = stem_sines.view(batch_size, self.max_output_per_stem, 3)
            stem_outputs.append(stem_sines)

        stems = torch.stack(stem_outputs, dim=1)

        # Apply constraints for NORMALIZED values (all features in 0-1 range):
        # - freq: 0-1 (normalized from 0-22050 Hz)
        # - amp: 0-1 (already in this range)
        # - frame: 0-1 (normalized from 0-chunk_frames)
        stems = torch.clamp(stems, min=0.0, max=1.0)

        return stems


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']

    model = SinusoidalTransformer(
        n_stems=len(config['stem_names']),
        max_input_sinusoids=config['max_input_sinusoids'],
        max_output_per_stem=config['max_output_per_stem'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Stems: {config['stem_names']}")

    return model, config


def load_sinusoids_from_h5(h5_path, max_sinusoids=2000):
    """Load all sinusoids from HDF5 file"""
    print(f"Loading sinusoids from: {h5_path}")

    sinusoids = []

    with h5py.File(h5_path, 'r') as f:
        n_channels = f.attrs.get('ch', 1)
        max_frame = 0

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            # Load all sinusoid data
            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            frame_indices = grp['i'][:]

            for freq, amp, frame in zip(frequencies, amplitudes, frame_indices):
                sinusoids.append([freq, amp, frame])
                max_frame = max(max_frame, frame)

    sinusoids = np.array(sinusoids) if len(sinusoids) > 0 else np.zeros((0, 3))
    print(f"  Loaded {len(sinusoids)} sinusoids, max frame: {max_frame}")

    return sinusoids, max_frame


def synthesize_audio_from_sinusoids(sinusoids, sr=44100, hop_length=512, duration=None):
    """
    Synthesize audio from sinusoids using additive synthesis.

    Args:
        sinusoids: [n_sinusoids, 3] - [freq, amp, frame]
        sr: sample rate
        hop_length: hop length
        duration: duration in samples (if None, determined from max frame)
    """
    if len(sinusoids) == 0 or sinusoids[:, 1].max() == 0:
        # No sinusoids or all zero amplitude
        if duration is None:
            duration = sr  # 1 second of silence
        return np.zeros(duration)

    # Determine audio length
    if duration is None:
        max_frame = int(sinusoids[:, 2].max())
        duration = (max_frame + 1) * hop_length

    audio = np.zeros(duration)

    # Synthesize each sinusoid
    for freq, amp, frame in sinusoids:
        if amp == 0 or freq == 0:
            continue

        # Calculate time position
        frame_idx = int(frame)
        start_sample = frame_idx * hop_length

        # Generate sinusoid (use a short window around the frame)
        window_samples = hop_length * 2  # Window of 2 frames
        end_sample = min(start_sample + window_samples, duration)

        if start_sample >= duration:
            continue

        # Generate samples
        n_samples = end_sample - start_sample
        t = np.arange(n_samples) / sr
        phase = 0  # Could use actual phase if available

        # Sinusoid
        sine_wave = amp * np.sin(2 * np.pi * freq * t + phase)

        # Apply window to avoid clicks
        window = np.hanning(n_samples)
        sine_wave *= window

        # Add to audio
        audio[start_sample:end_sample] += sine_wave

    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8  # Leave some headroom

    return audio


def process_sinusoids(h5_path, model, config, device, chunk_duration=4.0, hop_length=512):
    """Process sinusoids through model in chunks"""
    # Load all sinusoids
    all_sinusoids, max_frame = load_sinusoids_from_h5(h5_path, config['max_input_sinusoids'])

    if len(all_sinusoids) == 0:
        print("Warning: No sinusoids found in input!")
        return {name: np.zeros(44100) for name in config['stem_names']}

    # Use config values if available, otherwise use defaults
    chunk_duration = config.get('chunk_duration', chunk_duration)
    hop_length = config.get('hop_length', hop_length)
    chunk_frames = config.get('chunk_frames', int(chunk_duration * 44100 / hop_length))

    n_chunks = int(np.ceil(max_frame / chunk_frames))

    print(f"Processing {n_chunks} chunks...")

    # Initialize output stems
    sr = 44100
    total_duration = (max_frame + 1) * hop_length
    stem_audio = {name: np.zeros(total_duration) for name in config['stem_names']}

    with torch.no_grad():
        for chunk_idx in tqdm(range(n_chunks)):
            start_frame = chunk_idx * chunk_frames
            end_frame = start_frame + chunk_frames

            # Extract sinusoids in this chunk
            mask = (all_sinusoids[:, 2] >= start_frame) & (all_sinusoids[:, 2] < end_frame)
            chunk_sines = all_sinusoids[mask].copy()

            if len(chunk_sines) == 0:
                continue

            # Make frames chunk-relative
            chunk_sines[:, 2] -= start_frame

            # Pad/truncate to max_input_sinusoids
            if len(chunk_sines) > config['max_input_sinusoids']:
                indices = np.linspace(0, len(chunk_sines) - 1, config['max_input_sinusoids'], dtype=int)
                chunk_sines = chunk_sines[indices]

            if len(chunk_sines) < config['max_input_sinusoids']:
                padding = np.zeros((config['max_input_sinusoids'] - len(chunk_sines), 3))
                chunk_sines = np.vstack([chunk_sines, padding])

            # NORMALIZE input: Scale features to 0-1 range (same as training)
            chunk_sines_norm = chunk_sines.copy()
            chunk_sines_norm[:, 0] /= 22050.0  # normalize frequency
            chunk_sines_norm[:, 2] /= chunk_frames  # normalize frame

            # Convert to tensor
            chunk_tensor = torch.from_numpy(chunk_sines_norm).float().unsqueeze(0).to(device)

            # Generate stems
            generated_stems = model(chunk_tensor)  # [1, n_stems, max_output_per_stem, 3]

            # Synthesize audio for each stem
            for stem_idx, stem_name in enumerate(config['stem_names']):
                stem_sines = generated_stems[0, stem_idx].cpu().numpy()

                # DENORMALIZE output: Convert from 0-1 back to original ranges
                stem_sines[:, 0] *= 22050.0  # denormalize frequency
                stem_sines[:, 2] *= chunk_frames  # denormalize frame

                # Filter out NaN/inf values (model might output invalid values early in training)
                valid_mask = np.isfinite(stem_sines).all(axis=1) & (stem_sines[:, 1] > 0)
                stem_sines = stem_sines[valid_mask]

                if len(stem_sines) == 0:
                    continue  # No valid sinusoids in this chunk

                # Make frames absolute again
                stem_sines[:, 2] += start_frame

                # Synthesize this chunk
                start_sample = start_frame * hop_length
                end_sample = end_frame * hop_length
                chunk_duration_samples = end_sample - start_sample

                chunk_audio = synthesize_audio_from_sinusoids(
                    stem_sines,
                    sr=sr,
                    hop_length=hop_length,
                    duration=chunk_duration_samples
                )

                # Add to output
                actual_end = min(start_sample + len(chunk_audio), total_duration)
                stem_audio[stem_name][start_sample:actual_end] += chunk_audio[:actual_end - start_sample]

    # Normalize each stem
    for stem_name in stem_audio:
        max_val = np.abs(stem_audio[stem_name]).max()
        if max_val > 0:
            stem_audio[stem_name] = stem_audio[stem_name] / max_val * 0.8

    return stem_audio


def main():
    parser = argparse.ArgumentParser(description='Run inference with sinusoidal stem separator')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input fullmix .h5 file')
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

    # Process sinusoids
    stem_audio = process_sinusoids(args.input, model, config, device)

    # Save stems
    print("\nSaving stems...")
    sr = 44100
    for stem_name, audio in stem_audio.items():
        output_path = output_dir / f'{stem_name}.wav'
        sf.write(output_path, audio, sr)
        print(f"  Saved: {output_path}")

    print("\nDone! Stems saved to:", output_dir)


if __name__ == "__main__":
    main()
