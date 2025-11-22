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


def get_sinusoids_for_frame(h5_path, frame_idx, max_sinusoids_per_frame=2000):
    """
    Extract sinusoids for a specific frame from HDF5 file.

    Returns:
        array of shape (max_sinusoids_per_frame, 3) where 3 = [freq, amp, phase]
    """
    array = np.zeros((max_sinusoids_per_frame, 3), dtype=np.float32)

    if not h5_path.exists():
        return array

    sinusoids = []

    with h5py.File(h5_path, 'r') as f:
        if 'c0' not in f:
            return array

        grp = f['c0']
        track_lens = grp['len'][:]
        frequencies = grp['f'][:]
        amplitudes = grp['a'][:]
        phases = grp['p'][:] if 'p' in grp else np.zeros_like(amplitudes)
        frames = grp['i'][:]

        # Process each track
        offset = 0
        for track_len in track_lens:
            track_frames = frames[offset:offset + track_len]

            # Find sinusoids at this specific frame
            mask = track_frames == frame_idx
            if mask.any():
                freq = frequencies[offset:offset + track_len][mask][0]
                amp = amplitudes[offset:offset + track_len][mask][0]
                phase = phases[offset:offset + track_len][mask][0]
                sinusoids.append([freq, amp, phase])

            offset += track_len

    # Fill array
    if sinusoids:
        # Sort by frequency for consistency
        sinusoids = sorted(sinusoids, key=lambda x: x[0])
        n_sinusoids = min(len(sinusoids), max_sinusoids_per_frame)
        array[:n_sinusoids, :] = sinusoids[:n_sinusoids]

    return array


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


def merge_stem_arrays(h5_files, frame_idx, max_sinusoids_per_frame):
    """Merge sinusoids from multiple HDF5 files (e.g., drums_1, drums_2, ...)"""
    merged_array = np.zeros((max_sinusoids_per_frame, 3), dtype=np.float32)

    all_sinusoids = []
    for h5_file in h5_files:
        file_array = get_sinusoids_for_frame(h5_file, frame_idx, max_sinusoids_per_frame)
        # Extract non-zero sinusoids
        non_zero_mask = file_array[:, 0] > 0
        all_sinusoids.extend(file_array[non_zero_mask].tolist())

    # Fill merged array
    if all_sinusoids:
        all_sinusoids = sorted(all_sinusoids, key=lambda x: x[0])
        n_sinusoids = min(len(all_sinusoids), max_sinusoids_per_frame)
        merged_array[:n_sinusoids, :] = all_sinusoids[:n_sinusoids]

    return merged_array


def preprocess_data_dir(data_dir, output_dir, stem_names, max_sinusoids):
    """Preprocess all frames in a data directory"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Find fullmix
    fullmix_h5 = data_dir / 'fullmix_tracks.h5'
    if not fullmix_h5.exists():
        fullmix_h5 = data_dir / 'fullmix.h5'

    if not fullmix_h5.exists():
        print(f"Skipping {data_dir}: no fullmix found")
        return []

    # Determine number of frames
    with h5py.File(fullmix_h5, 'r') as f:
        if 'c0' not in f or 'i' not in f['c0']:
            print(f"Skipping {data_dir}: invalid fullmix")
            return []
        max_frame = f['c0']['i'][:].max() if len(f['c0']['i']) > 0 else 0

    n_frames = int(max_frame) + 1
    print(f"\n{data_dir.name}: {n_frames} frames")

    # Create output directory
    data_output_dir = output_dir / data_dir.name
    data_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each frame
    processed_frames = []

    for frame_idx in tqdm(range(n_frames), desc=f"Processing {data_dir.name}"):
        # Get fullmix sinusoids for this frame
        fullmix_array = get_sinusoids_for_frame(fullmix_h5, frame_idx, max_sinusoids)

        # Get each stem's sinusoids for this frame
        stem_arrays = []
        for stem_name in stem_names:
            h5_files = find_stem_h5(data_dir, stem_name)

            if len(h5_files) > 1:
                # Merge multiple files (e.g., drums)
                stem_array = merge_stem_arrays(h5_files, frame_idx, max_sinusoids)
            elif len(h5_files) == 1:
                stem_array = get_sinusoids_for_frame(h5_files[0], frame_idx, max_sinusoids)
            else:
                stem_array = np.zeros((max_sinusoids, 3), dtype=np.float32)

            stem_arrays.append(stem_array)

        # Stack stems: (n_stems, max_sinusoids, 3)
        stem_arrays = np.stack(stem_arrays, axis=0)

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
