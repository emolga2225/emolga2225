#!/usr/bin/env python3
"""
Preprocess HDF5 sinusoidal data to numpy arrays for fast training.

This converts HDF5 files to pre-computed frame-based arrays, which
dramatically speeds up training (13s/iter -> <1s/iter).

For each data directory:
1. Load fullmix and stem HDF5 files
2. Convert each frame to exact frequency array format
3. Save as compressed .npz files (one per frame)
4. Create index file for fast loading

Usage:
    python preprocess_hdf5_to_arrays.py \
        --data-dirs ajfa/ blackned/ dyerseve/ \
        --output-dir preprocessed_data/ \
        --max-sinusoids 2000
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from collections import defaultdict


def load_all_frames_from_h5(h5_path, max_sinusoids_per_frame=2000):
    """
    Load ALL frames from HDF5 file at once (much faster than per-frame loading).

    Returns:
        dict: frame_idx -> array of shape (max_sinusoids_per_frame, 3)
    """
    frame_data = {}

    if not h5_path.exists():
        return frame_data

    with h5py.File(h5_path, 'r') as f:
        if 'c0' not in f:
            return frame_data

        grp = f['c0']
        track_lens = grp['len'][:]
        frequencies = grp['f'][:]
        amplitudes = grp['a'][:]
        phases = grp['p'][:] if 'p' in grp else np.zeros_like(amplitudes)
        frames = grp['i'][:]

        # Group sinusoids by frame (much faster than processing each frame separately)
        frame_sinusoids = defaultdict(list)

        offset = 0
        for track_len in track_lens:
            track_freqs = frequencies[offset:offset + track_len]
            track_amps = amplitudes[offset:offset + track_len]
            track_phases = phases[offset:offset + track_len]
            track_frames = frames[offset:offset + track_len]

            # Add all sinusoids from this track to their respective frames
            for freq, amp, phase, frame_idx in zip(track_freqs, track_amps, track_phases, track_frames):
                frame_sinusoids[int(frame_idx)].append([freq, amp, phase])

            offset += track_len

    # Convert to arrays
    for frame_idx, sinusoids in frame_sinusoids.items():
        array = np.zeros((max_sinusoids_per_frame, 3), dtype=np.float32)

        # Sort by frequency for consistency
        sinusoids = sorted(sinusoids, key=lambda x: x[0])
        n_sinusoids = min(len(sinusoids), max_sinusoids_per_frame)
        array[:n_sinusoids, :] = sinusoids[:n_sinusoids]

        frame_data[frame_idx] = array

    return frame_data


def find_stem_h5(data_dir, stem_name):
    """Find HDF5 file(s) for a stem"""
    if stem_name == 'drums':
        # Check for multi-file drums
        drum_files = []
        for i in range(1, 10):
            drum_h5 = data_dir / f'drums_{i}_tracks.h5'
            if drum_h5.exists():
                drum_files.append(drum_h5)
        if drum_files:
            return drum_files
        # Fall back to single file
        for name in ['drums_tracks.h5', 'drums.h5']:
            h5_path = data_dir / name
            if h5_path.exists():
                return [h5_path]

    elif stem_name == 'bass':
        for name in ['bass_tracks.h5', 'rhythm_tracks.h5', 'bass.h5', 'rhythm.h5']:
            h5_path = data_dir / name
            if h5_path.exists():
                return [h5_path]

    else:
        for suffix in ['_tracks.h5', '.h5']:
            h5_path = data_dir / f'{stem_name}{suffix}'
            if h5_path.exists():
                return [h5_path]

    return []


def merge_multiple_h5_files(h5_files, max_sinusoids_per_frame):
    """
    Merge sinusoids from multiple HDF5 files (e.g., drums_1, drums_2, ...).

    Returns:
        dict: frame_idx -> merged array
    """
    # Load all files
    all_frame_data = []
    for h5_file in h5_files:
        frame_data = load_all_frames_from_h5(h5_file, max_sinusoids_per_frame)
        all_frame_data.append(frame_data)

    if not all_frame_data:
        return {}

    # Get all unique frame indices
    all_frame_indices = set()
    for frame_data in all_frame_data:
        all_frame_indices.update(frame_data.keys())

    # Merge frames
    merged_frames = {}
    for frame_idx in all_frame_indices:
        all_sinusoids = []

        # Collect sinusoids from all files for this frame
        for frame_data in all_frame_data:
            if frame_idx in frame_data:
                file_array = frame_data[frame_idx]
                # Extract non-zero sinusoids
                non_zero_mask = file_array[:, 0] > 0
                all_sinusoids.extend(file_array[non_zero_mask].tolist())

        # Create merged array
        merged_array = np.zeros((max_sinusoids_per_frame, 3), dtype=np.float32)
        if all_sinusoids:
            all_sinusoids = sorted(all_sinusoids, key=lambda x: x[0])
            n_sinusoids = min(len(all_sinusoids), max_sinusoids_per_frame)
            merged_array[:n_sinusoids, :] = all_sinusoids[:n_sinusoids]

        merged_frames[frame_idx] = merged_array

    return merged_frames


def preprocess_data_dir(data_dir, output_dir, stem_names, max_sinusoids):
    """Preprocess all frames in a data directory (loads one stem at a time to save memory)"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Find fullmix
    fullmix_h5 = data_dir / 'fullmix_tracks.h5'
    if not fullmix_h5.exists():
        fullmix_h5 = data_dir / 'fullmix.h5'

    if not fullmix_h5.exists():
        print(f"Skipping {data_dir}: no fullmix found")
        return []

    print(f"\n{data_dir.name}: Loading fullmix...")

    # Load ALL fullmix frames at once
    fullmix_frames = load_all_frames_from_h5(fullmix_h5, max_sinusoids)

    if not fullmix_frames:
        print(f"Skipping {data_dir}: no frames in fullmix")
        return []

    n_frames = max(fullmix_frames.keys()) + 1
    print(f"{data_dir.name}: {n_frames} frames")

    # Create output directory
    data_output_dir = output_dir / data_dir.name
    data_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize arrays to hold all stems for all frames
    # Shape: (n_frames, n_stems, max_sinusoids, 3)
    all_stems = np.zeros((n_frames, len(stem_names), max_sinusoids, 3), dtype=np.float32)

    # Process stems ONE AT A TIME to save memory
    for stem_idx, stem_name in enumerate(stem_names):
        print(f"{data_dir.name}: Loading {stem_name}...")

        h5_files = find_stem_h5(data_dir, stem_name)

        if len(h5_files) > 1:
            # Merge multiple files (e.g., drums)
            stem_frames = merge_multiple_h5_files(h5_files, max_sinusoids)
        elif len(h5_files) == 1:
            stem_frames = load_all_frames_from_h5(h5_files[0], max_sinusoids)
        else:
            stem_frames = {}

        # Fill in this stem's data for all frames
        for frame_idx in range(n_frames):
            all_stems[frame_idx, stem_idx, :, :] = stem_frames.get(
                frame_idx,
                np.zeros((max_sinusoids, 3), dtype=np.float32)
            )

        # Release memory for this stem
        del stem_frames
        print(f"{data_dir.name}: {stem_name} loaded")

    # Save all frames
    print(f"{data_dir.name}: Saving {n_frames} frames...")
    processed_frames = []

    for frame_idx in tqdm(range(n_frames), desc=f"Saving {data_dir.name}"):
        # Get fullmix for this frame
        fullmix_array = fullmix_frames.get(frame_idx, np.zeros((max_sinusoids, 3), dtype=np.float32))

        # Get all stems for this frame
        stem_arrays = all_stems[frame_idx]  # (n_stems, max_sinusoids, 3)

        # Save this frame
        frame_file = data_output_dir / f'frame_{frame_idx:06d}.npz'
        np.savez_compressed(
            frame_file,
            fullmix=fullmix_array,  # (max_sinusoids, 3)
            stems=stem_arrays        # (n_stems, max_sinusoids, 3)
        )

        processed_frames.append({
            'frame_idx': frame_idx,
            'file': str(frame_file.relative_to(output_dir)),
            'data_dir': data_dir.name
        })

    # Clean up
    del fullmix_frames
    del all_stems

    return processed_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', nargs='+', required=True,
                       help='Directories containing HDF5 files')
    parser.add_argument('--output-dir', default='preprocessed_data',
                       help='Output directory for preprocessed arrays')
    parser.add_argument('--stem-names', nargs='+',
                       default=['vocals', 'guitar', 'bass', 'drums', 'song'])
    parser.add_argument('--max-sinusoids', type=int, default=2000,
                       help='Max sinusoids per frame')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing HDF5 files to numpy arrays")
    print(f"Output directory: {output_dir}")
    print(f"Max sinusoids per frame: {args.max_sinusoids}")
    print(f"Stems: {args.stem_names}")

    # Process each data directory
    all_frames = []

    for data_dir in args.data_dirs:
        frames = preprocess_data_dir(
            data_dir,
            output_dir,
            args.stem_names,
            args.max_sinusoids
        )
        all_frames.extend(frames)

    # Save index
    index_file = output_dir / 'index.json'
    index_data = {
        'frames': all_frames,
        'stem_names': args.stem_names,
        'max_sinusoids': args.max_sinusoids,
        'n_frames': len(all_frames)
    }

    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)

    print(f"\nPreprocessing complete!")
    print(f"Total frames: {len(all_frames):,}")
    print(f"Index saved to: {index_file}")

    # Estimate size
    if all_frames:
        sample_file = output_dir / all_frames[0]['file']
        sample_size_mb = sample_file.stat().st_size / (1024**2)
        total_size_gb = (sample_size_mb * len(all_frames)) / 1024
        print(f"Estimated total size: {total_size_gb:.2f} GB ({sample_size_mb:.3f} MB per frame)")


if __name__ == "__main__":
    main()
