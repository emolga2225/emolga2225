#!/usr/bin/env python3
"""
Pre-synthesize all .h5 files to audio for faster training.

Run this once before training to generate synthesized audio from all .h5 files.
Training will then load these pre-synthesized files instead of synthesizing on-the-fly.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from hybrid_resolution_sinusoidal_extractor import HybridResolutionSinusoidalExtractor
import argparse


def synthesize_h5_file(h5_path, output_path, synthesizer):
    """Synthesize audio from .h5 file and save to .wav"""

    if output_path.exists():
        print(f"  ✓ Already exists: {output_path.name}")
        return

    print(f"  Synthesizing {h5_path.name}...")

    # Load tracks from .h5
    all_channel_tracks, is_stereo = synthesizer.load_from_hdf5(h5_path)

    # Determine length from track data
    import h5py
    with h5py.File(h5_path, 'r') as f:
        hop_size = int(f.attrs.get('hop', 512))
        n_channels = int(f.attrs.get('ch', 1))

        max_frame = 0
        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name in f:
                track_ends = f[grp_name]['e'][:]
                if len(track_ends) > 0:
                    max_frame = max(max_frame, int(np.max(track_ends)))

        # Convert frames to samples
        n_samples = max_frame * hop_size + 10000

    # Synthesize audio
    synthesized = synthesizer.synthesize_stereo(all_channel_tracks, n_samples, is_stereo)

    # Save to .wav
    sf.write(output_path, synthesized, synthesizer.sample_rate)
    print(f"    ✓ Saved {output_path.name} ({len(synthesized)/synthesizer.sample_rate:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description='Pre-synthesize .h5 files to audio')
    parser.add_argument('--data-dirs', nargs='+', required=True, help='Song directories')
    parser.add_argument('--stem-names', nargs='+', default=['vocals', 'guitar', 'bass', 'drums'])
    parser.add_argument('--sample-rate', type=int, default=44100)

    args = parser.parse_args()

    # Initialize synthesizer
    synthesizer = HybridResolutionSinusoidalExtractor(sample_rate=args.sample_rate)

    print("=" * 60)
    print("Pre-synthesizing .h5 Files")
    print("=" * 60)
    print(f"Processing {len(args.data_dirs)} song(s)...")
    print(f"Stems: {args.stem_names}")

    for data_dir_str in args.data_dirs:
        data_dir = Path(data_dir_str)
        print(f"\n📁 {data_dir.name}")

        for stem_name in args.stem_names:
            if stem_name == 'drums':
                # Process each drum track separately
                for i in range(1, 5):
                    h5_path = data_dir / f'drums_{i}_tracks.h5'
                    if h5_path.exists():
                        output_path = data_dir / f'drums_{i}_synthesized.wav'
                        synthesize_h5_file(h5_path, output_path, synthesizer)
            else:
                h5_path = data_dir / f'{stem_name}_tracks.h5'
                if h5_path.exists():
                    output_path = data_dir / f'{stem_name}_synthesized.wav'
                    synthesize_h5_file(h5_path, output_path, synthesizer)

    print("\n" + "=" * 60)
    print("✓ Pre-synthesis complete!")
    print("=" * 60)
    print("\nYou can now run training - it will use the pre-synthesized files.")


if __name__ == '__main__':
    main()
