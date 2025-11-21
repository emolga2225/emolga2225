#!/usr/bin/env python3
"""
Preprocess HDF5 sinusoids to chunked .npz files for fast training.

This converts slow HDF5 sinusoidal tracks into fast numpy chunks.
Memory-efficient: processes one stem at a time to avoid OOM.

Run once:
    python preprocess_sinusoids_to_chunks.py --data-dirs ajfa/ blackned/ dyerseve/

Then training will be much faster.
"""

import numpy as np
import h5py
from pathlib import Path
import argparse
from tqdm import tqdm
import gc


def load_all_sinusoids_from_h5(h5_path):
    """Load ALL sinusoids from HDF5 file at once (track-aware)"""
    sinusoids = []

    with h5py.File(h5_path, 'r') as f:
        n_channels = f.attrs.get('ch', 1)

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            # Load track structure (CRITICAL for correct amplitude loading!)
            track_lens = grp['len'][:]

            # Load concatenated sinusoid data
            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            frame_indices = grp['i'][:]

            # Process each track (respecting track boundaries)
            offset = 0
            for track_len in track_lens:
                # Extract data for this track
                track_freqs = frequencies[offset:offset + track_len]
                track_amps = amplitudes[offset:offset + track_len]
                track_frames = frame_indices[offset:offset + track_len]

                # Add sinusoids from this track
                for freq, amp, frame in zip(track_freqs, track_amps, track_frames):
                    sinusoids.append([freq, amp, frame])

                offset += track_len

    return np.array(sinusoids, dtype=np.float32) if len(sinusoids) > 0 else np.zeros((0, 3), dtype=np.float32)


def load_and_merge_drum_files(data_dir):
    """Load and merge multiple drum files (drums_1, drums_2, drums_3, etc.)"""
    all_sinusoids = []

    # Try to find all drum files
    drum_files = []
    for i in range(1, 10):  # Check drums_1 through drums_9
        drum_h5 = data_dir / f'drums_{i}_tracks.h5'
        if drum_h5.exists():
            drum_files.append(drum_h5)

    if not drum_files:
        return None

    print(f"    Found {len(drum_files)} drum files, merging...")
    for drum_file in drum_files:
        sinusoids = load_all_sinusoids_from_h5(drum_file)
        if len(sinusoids) > 0:
            all_sinusoids.append(sinusoids)

    if not all_sinusoids:
        return np.zeros((0, 3), dtype=np.float32)

    # Merge all drum sinusoids
    return np.vstack(all_sinusoids)


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
    """Preprocess all HDF5 files in a directory to chunked .npz (memory-efficient)"""
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

    # Create chunks directory
    chunks_dir = data_dir / 'sinusoid_chunks'
    chunks_dir.mkdir(exist_ok=True)

    # STEP 1: Process fullmix first (creates all chunk files with just fullmix)
    print(f"  Processing fullmix...")
    fullmix_all = load_all_sinusoids_from_h5(fullmix_h5)

    for chunk_idx in tqdm(range(n_chunks), desc=f"  Creating chunks"):
        fullmix_sines = chunk_sinusoids(fullmix_all, chunk_idx, chunk_frames)
        chunk_file = chunks_dir / f'chunk_{chunk_idx:04d}.npz'
        np.savez_compressed(chunk_file, fullmix=fullmix_sines)

    # Clear fullmix from memory
    del fullmix_all
    gc.collect()

    # STEP 2: Process each stem one at a time (updates existing chunk files)
    for stem_name in stem_names:
        stem_all = None

        # Special case: drums can be split across multiple files
        if stem_name == 'drums':
            stem_all = load_and_merge_drum_files(data_dir)
            if stem_all is None:
                print(f"  Skipping {stem_name} (not found)")
                continue

        # Special case: bass can be named "bass" or "rhythm"
        elif stem_name == 'bass':
            # Try bass_tracks.h5, then rhythm_tracks.h5
            stem_h5 = data_dir / 'bass_tracks.h5'
            if not stem_h5.exists():
                stem_h5 = data_dir / 'rhythm_tracks.h5'
            if not stem_h5.exists():
                stem_h5 = data_dir / 'bass.h5'
            if not stem_h5.exists():
                stem_h5 = data_dir / 'rhythm.h5'

            if not stem_h5.exists():
                print(f"  Skipping {stem_name} (not found)")
                continue

            print(f"  Processing {stem_name}... (using {stem_h5.name})")
            stem_all = load_all_sinusoids_from_h5(stem_h5)

        # Regular stems
        else:
            # Try _tracks.h5 suffix first, then plain .h5
            stem_h5 = data_dir / f'{stem_name}_tracks.h5'
            if not stem_h5.exists():
                stem_h5 = data_dir / f'{stem_name}.h5'

            if not stem_h5.exists():
                print(f"  Skipping {stem_name} (not found)")
                continue

            print(f"  Processing {stem_name}...")
            stem_all = load_all_sinusoids_from_h5(stem_h5)

        # Add stem data to each existing chunk
        for chunk_idx in tqdm(range(n_chunks), desc=f"  Adding {stem_name}"):
            stem_sines = chunk_sinusoids(stem_all, chunk_idx, chunk_frames)

            # Load existing chunk
            chunk_file = chunks_dir / f'chunk_{chunk_idx:04d}.npz'
            existing_data = np.load(chunk_file)

            # Combine with new stem data
            updated_data = {key: existing_data[key] for key in existing_data.files}
            updated_data[stem_name] = stem_sines

            # Save updated chunk
            np.savez_compressed(chunk_file, **updated_data)

        # Clear stem from memory before loading next one
        del stem_all
        gc.collect()

    print(f"  Saved {n_chunks} chunks to {chunks_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess HDF5 sinusoids to chunked .npz files')
    parser.add_argument('--data-dirs', nargs='+', required=True,
                       help='Directories containing HDF5 files')
    parser.add_argument('--stem-names', nargs='+',
                       default=['vocals', 'guitar', 'bass', 'drums', 'song'],
                       help='Stem names to process')
    parser.add_argument('--chunk-duration', type=float, default=4.0,
                       help='Chunk duration in seconds')
    args = parser.parse_args()

    print("HDF5 Sinusoids to Chunked .npz Preprocessing")
    print("=" * 60)
    print(f"Processing {len(args.data_dirs)} directories...")
    print(f"Stems: {args.stem_names}")
    print(f"Chunk duration: {args.chunk_duration}s")
    print(f"Memory-efficient mode: processing one stem at a time")

    for data_dir in args.data_dirs:
        preprocess_directory(data_dir, args.stem_names, args.chunk_duration)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("\nPreprocessed chunks created in sinusoid_chunks/ subdirectories")
    print("Training will now load from these fast .npz files.")


if __name__ == "__main__":
    main()
