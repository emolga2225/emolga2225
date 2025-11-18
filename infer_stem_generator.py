#!/usr/bin/env python3
"""
Inference script for generative stem separator

Takes a fullmix audio file and generates separated stem audio files.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from train_stem_generator import UNetStemGenerator


def load_model(checkpoint_path, device='cuda'):
    """Load trained model"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']

    model = UNetStemGenerator(
        n_stems=len(config['stem_names']),
        in_channels=1,
        base_channels=32
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def generate_stems(model, audio_path, config, device='cuda', output_dir='generated_stems'):
    """Generate stems from fullmix audio"""
    print(f"\nGenerating stems from {audio_path}...")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=config['sample_rate'], mono=True)
    print(f"Loaded audio: {len(audio)/sr:.2f} seconds")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Process in segments
    segment_samples = int(config['segment_length'] * config['sample_rate'])
    n_segments = len(audio) // segment_samples

    # Initialize arrays for accumulated stems
    n_stems = len(config['stem_names'])
    stems_audio = {name: np.zeros_like(audio) for name in config['stem_names']}

    print(f"Processing {n_segments} segments...")

    with torch.no_grad():
        for seg_idx in range(n_segments):
            start = seg_idx * segment_samples
            end = start + segment_samples

            # Extract segment
            segment = audio[start:end]

            # Compute spectrogram
            stft = librosa.stft(segment, n_fft=config['n_fft'], hop_length=config['hop_length'])
            magnitude = np.abs(stft)
            phase = np.angle(stft)  # Keep phase for reconstruction
            log_mag = np.log1p(magnitude)

            # Convert to tensor
            mix_spec = torch.from_numpy(log_mag).float().unsqueeze(0).unsqueeze(0).to(device)

            # Generate stem spectrograms
            generated_stems = model(mix_spec)  # [1, n_stems, freq, time]
            generated_stems = generated_stems.squeeze(0).cpu().numpy()  # [n_stems, freq, time]

            # Convert back from log scale
            generated_stems = np.expm1(generated_stems)

            # Reconstruct audio for each stem using original phase
            for stem_idx, stem_name in enumerate(config['stem_names']):
                stem_mag = generated_stems[stem_idx]

                # Use original phase (this is a simplification - ideally estimate phase)
                stem_stft = stem_mag * np.exp(1j * phase)

                # Inverse STFT
                stem_audio = librosa.istft(stem_stft, hop_length=config['hop_length'])

                # Add to accumulated stem
                stems_audio[stem_name][start:start+len(stem_audio)] += stem_audio

            if (seg_idx + 1) % 10 == 0:
                print(f"  Processed {seg_idx + 1}/{n_segments} segments")

    # Save stems
    print(f"\nSaving stems to {output_path}...")
    for stem_name, stem_audio in stems_audio.items():
        output_file = output_path / f'{stem_name}.wav'

        # Normalize
        max_val = np.abs(stem_audio).max()
        if max_val > 0:
            stem_audio = stem_audio / max_val * 0.95  # Prevent clipping

        sf.write(output_file, stem_audio, config['sample_rate'])
        print(f"  ✓ {stem_name}.wav")

    print("\nStem generation complete!")


def main():
    parser = argparse.ArgumentParser(description='Generate stems from fullmix audio')
    parser.add_argument('input', help='Input fullmix audio file')
    parser.add_argument('--checkpoint', default='stem_generator_final.pt',
                       help='Model checkpoint to use')
    parser.add_argument('--output-dir', default='generated_stems',
                       help='Output directory for separated stems')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.checkpoint, device=args.device)

    # Generate stems
    generate_stems(model, args.input, config, device=args.device, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
