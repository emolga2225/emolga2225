#!/usr/bin/env python3
"""
Preprocess HDF5 sinusoids to chunked .npz files for fast training.

This converts slow HDF5 sinusoidal tracks into fast numpy chunks.

Run once:
    python preprocess_sinusoids_to_chunks.py --data-dirs ajfa/ blackned/ dyerseve/

Then training will be much faster.
"""

import numpy as np
import h5py
from pathlib import Path
import argparse
from tqdm import tqdm


def load_all_sinusoids_from_h5(h5_path):
    """Load ALL sinusoids from HDF5 file at once"""
    sinusoids = []

    with h5py.File(h5_path, 'r') as f:
        n_channels = f.attrs.get('ch', 1)

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            # Load ALL sinusoid data at once
            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            frame_indices = grp['i'][:]

            # Create sinusoid array
            for freq, amp, frame in zip(frequencies, amplitudes, frame_indices):
                sinusoids.append([freq, amp, frame])

    return np.array(sinusoids, dtype=np.float32) if len(sinusoids) > 0 else np.zeros((0, 3), dtype=np.float32)


def chunk_sinusoids(all_sinusoids, chunk_idx, chunk_frames):
    """Extract a chunk from already-loaded sinusoids"""
    start_frame = chunk_idx * chunk_frames
    end_frame = start_frame + chunk_frames

    # Filter sinusoids in this time range
    mask = (all_sinusoids[:, 2] >= start_frame) & (all_sinusoids[:, 2] < end_frame)
    chunk_sines = all_sinusoids[mask].copy()

    # Make frames chunk-relative
    if len(chunk_sines) > 0:
        chunk_sines[:, 2] -= start_frame

    return chunk_sines


def preprocess_directory(data_dir, stem_names, chunk_duration=4.0, hop_length=512):
    """Preprocess all HDF5 files in a directory to chunked .npz"""
    data_dir = Path(data_dir)
    print(f"\nPreprocessing {data_dir.name}...")

    chunk_frames = int(chunk_duration * 44100 / hop_length)

    # Find fullmix
    fullmix_h5 = data_dir / 'fullmix_tracks.h5'
    if not fullmix_h5.exists():
        fullmix_h5 = data_dir / 'fullmix.h5'
        if not fullmix_h5.exists():
            print(f"  No fullmix .h5 found, skipping")
            return

    # Determine number of chunks
    with h5py.File(fullmix_h5, 'r') as f:
        if 'c0' in f and 'i' in f['c0']:
            max_frame = f['c0']['i'][:].max() if len(f['c0']['i']) > 0 else 0
        else:
            print(f"  No sinusoids in fullmix, skipping")
            return

    n_chunks = max_frame // chunk_frames
    if n_chunks == 0:
        n_chunks = 1

    print(f"  Loading all sinusoids from .h5 files...")

    # Load fullmix sinusoids ONCE
    print(f"    Loading fullmix...")
    fullmix_all = load_all_sinusoids_from_h5(fullmix_h5)

    # Load stem sinusoids ONCE
    stem_all = {}
    for stem_name in stem_names:
        # Try _tracks.h5 suffix first, then plain .h5
        stem_h5 = data_dir / f'{stem_name}_tracks.h5'
        if not stem_h5.exists():
            stem_h5 = data_dir / f'{stem_name}.h5'

        if stem_h5.exists():
            print(f"    Loading {stem_name}...")
            stem_all[stem_name] = load_all_sinusoids_from_h5(stem_h5)

    # Create chunks directory
    chunks_dir = data_dir / 'sinusoid_chunks'
    chunks_dir.mkdir(exist_ok=True)

    # Now chunk the loaded sinusoids (fast!)
    print(f"  Creating {n_chunks} chunks...")
    for chunk_idx in tqdm(range(n_chunks), desc=f"  {data_dir.name}"):
        # Extract fullmix chunk
        fullmix_sines = chunk_sinusoids(fullmix_all, chunk_idx, chunk_frames)

        # Extract stem chunks
        stem_data = {}
        for stem_name in stem_names:
            if stem_name in stem_all:
                stem_sines = chunk_sinusoids(stem_all[stem_name], chunk_idx, chunk_frames)
                stem_data[stem_name] = stem_sines

        # Save chunk
        chunk_file = chunks_dir / f'chunk_{chunk_idx:04d}.npz'
        np.savez_compressed(chunk_file, fullmix=fullmix_sines, **stem_data)

    print(f"  Saved {n_chunks} chunks to {chunks_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess HDF5 sinusoids to chunked .npz files')
    parser.add_argument('--data-dirs', nargs='+', required=True,
                       help='Directories containing HDF5 files')
    parser.add_argument('--stem-names', nargs='+',
                       default=['vocals', 'guitar', 'bass', 'drums'],
                       help='Stem names to process')
    parser.add_argument('--chunk-duration', type=float, default=4.0,
                       help='Chunk duration in seconds')
    args = parser.parse_args()

    print("HDF5 Sinusoids to Chunked .npz Preprocessing")
    print("=" * 60)
    print(f"Processing {len(args.data_dirs)} directories...")
    print(f"Stems: {args.stem_names}")
    print(f"Chunk duration: {args.chunk_duration}s")

    for data_dir in args.data_dirs:
        preprocess_directory(data_dir, args.stem_names, args.chunk_duration)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("\nPreprocessed chunks created in sinusoid_chunks/ subdirectories")
    print("Training will now load from these fast .npz files.")


if __name__ == "__main__":
    main()
