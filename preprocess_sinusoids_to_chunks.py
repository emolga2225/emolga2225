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


def extract_sinusoids_from_h5(h5_path, chunk_frames, chunk_idx):
    """Extract sinusoids from a specific time chunk"""
    start_frame = chunk_idx * chunk_frames
    end_frame = start_frame + chunk_frames

    sinusoids = []

    with h5py.File(h5_path, 'r') as f:
        n_channels = f.attrs.get('ch', 1)

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            # Load track data
            track_lens = grp['len'][:]
            track_starts = grp['s'][:]
            track_ends = grp['e'][:]

            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            frame_indices = grp['i'][:]

            # Process tracks that overlap with this chunk
            offset = 0
            for track_idx, track_len in enumerate(track_lens):
                track_start = track_starts[track_idx]
                track_end = track_ends[track_idx]

                if track_end >= start_frame and track_start < end_frame:
                    track_freqs = frequencies[offset:offset + track_len]
                    track_amps = amplitudes[offset:offset + track_len]
                    track_frames = frame_indices[offset:offset + track_len]

                    for freq, amp, frame in zip(track_freqs, track_amps, track_frames):
                        if start_frame <= frame < end_frame:
                            local_frame = frame - start_frame
                            sinusoids.append([freq, amp, local_frame])

                offset += track_len

    return np.array(sinusoids, dtype=np.float32) if len(sinusoids) > 0 else np.zeros((0, 3), dtype=np.float32)


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

    print(f"  Processing {n_chunks} chunks...")

    # Create chunks directory
    chunks_dir = data_dir / 'sinusoid_chunks'
    chunks_dir.mkdir(exist_ok=True)

    # Process each chunk
    for chunk_idx in tqdm(range(n_chunks), desc=f"  {data_dir.name}"):
        # Extract fullmix sinusoids
        fullmix_sines = extract_sinusoids_from_h5(fullmix_h5, chunk_frames, chunk_idx)

        # Extract stem sinusoids
        stem_data = {}
        for stem_name in stem_names:
            stem_h5 = data_dir / f'{stem_name}.h5'
            if stem_h5.exists():
                stem_sines = extract_sinusoids_from_h5(stem_h5, chunk_frames, chunk_idx)
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
