#!/usr/bin/env python3
"""
Preprocess HDF5 sinusoidal data to dense spectrograms for faster training.

This script converts the slow-to-load HDF5 sinusoidal tracks into
dense numpy arrays that can be loaded quickly during training.

Run this once before training:
    python preprocess_h5_to_dense.py --data-dirs ajfa/ blackned/ dyerseve/

Then training will be much faster (seconds instead of minutes per epoch).
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse


def load_full_h5_to_dense(h5_path, n_fft=2048, sr=44100, hop_length=512):
    """Load entire HDF5 sinusoidal file and convert to dense spectrogram"""
    print(f"  Processing {h5_path.name}...")

    with h5py.File(h5_path, 'r') as f:
        # Get metadata
        n_channels = f.attrs.get('ch', 1)

        # Find max frame index to determine spectrogram size
        max_frame = 0
        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name in f:
                grp = f[grp_name]
                if 'i' in grp:
                    frame_indices = grp['i'][:]
                    if len(frame_indices) > 0:
                        max_frame = max(max_frame, frame_indices.max())

        n_frames = max_frame + 1
        n_bins = 1 + (n_fft // 2)

        # Initialize spectrogram
        spec = np.zeros((n_bins, n_frames), dtype=np.float32)

        # Process all channels
        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            # Load all data
            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            frame_indices = grp['i'][:]

            # Populate spectrogram
            for freq, amp, frame in zip(frequencies, amplitudes, frame_indices):
                bin_idx = int(freq * n_fft / sr)
                if 0 <= bin_idx < n_bins and 0 <= frame < n_frames:
                    spec[bin_idx, frame] += amp

        # Average across channels
        if n_channels > 0:
            spec = spec / n_channels

        # Convert to log scale
        log_spec = np.log1p(spec)

        return log_spec


def preprocess_directory(data_dir, stem_names):
    """Preprocess all H5 files in a directory"""
    data_dir = Path(data_dir)
    print(f"\nPreprocessing {data_dir.name}...")

    # Process fullmix
    fullmix_h5 = data_dir / 'fullmix_tracks.h5'
    if not fullmix_h5.exists():
        fullmix_h5 = data_dir / 'fullmix.h5'

    if fullmix_h5.exists():
        fullmix_spec = load_full_h5_to_dense(fullmix_h5)
        output_path = data_dir / 'fullmix_spec.npy'
        np.save(output_path, fullmix_spec)
        print(f"    Saved {output_path.name} ({fullmix_spec.shape})")
    else:
        print(f"    No fullmix .h5 file found, will use audio fallback")

    # Process stems
    for stem_name in stem_names:
        h5_path = data_dir / f'{stem_name}.h5'
        if h5_path.exists():
            stem_spec = load_full_h5_to_dense(h5_path)
            output_path = data_dir / f'{stem_name}_spec.npy'
            np.save(output_path, stem_spec)
            print(f"    Saved {output_path.name} ({stem_spec.shape})")


def main():
    parser = argparse.ArgumentParser(description='Preprocess HDF5 data to dense spectrograms')
    parser.add_argument('--data-dirs', nargs='+', required=True,
                       help='Directories containing HDF5 files')
    parser.add_argument('--stem-names', nargs='+',
                       default=['vocals', 'guitar', 'bass', 'drums'],
                       help='Stem names to process')
    args = parser.parse_args()

    print("HDF5 to Dense Spectrogram Preprocessing")
    print("=" * 50)
    print(f"Processing {len(args.data_dirs)} directories...")
    print(f"Stems: {args.stem_names}")

    for data_dir in args.data_dirs:
        preprocess_directory(data_dir, args.stem_names)

    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print("\nPreprocessed files created:")
    print("  - fullmix_spec.npy (input spectrogram)")
    print("  - {stem}_spec.npy (auxiliary guidance spectrograms)")
    print("\nYou can now train faster by using these preprocessed files.")


if __name__ == "__main__":
    main()
