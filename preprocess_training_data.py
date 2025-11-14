#!/usr/bin/env python3
"""
Preprocess training data from HDF5 to memory-mapped numpy arrays.

This converts the slow HDF5-based dataset into fast .npy files that can be
loaded efficiently during training. Run this once before training.
"""

import h5py
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


def preprocess_dataset(
    mix_path='fullmix_tracks.h5',
    labels_path='stem_labels.json',
    output_dir='preprocessed_data',
    window_size=32,
    window_stride=16,
    match_threshold=0.5
):
    """
    Preprocess the entire dataset into memory-mapped arrays.

    Args:
        mix_path: Path to mix HDF5 file
        labels_path: Path to labels JSON
        output_dir: Output directory for preprocessed data
        window_size: Number of frames per window
        window_stride: Stride between windows
        match_threshold: Minimum match ratio for valid windows
    """
    print("Loading labels...")
    with open(labels_path) as f:
        labels_data = json.load(f)

    print("Opening mix HDF5 file...")
    mix_file = h5py.File(mix_path, 'r')

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # First pass: count total windows
    print("\nFirst pass: counting windows...")
    total_windows = 0
    window_info_list = []

    for track_idx_str, labels_list in tqdm(labels_data.items(), desc="Counting"):
        track_idx = int(track_idx_str)
        track_key = f'track_{track_idx}'

        if track_key not in mix_file:
            continue

        track_len = len(labels_list)

        # Skip tracks shorter than window size
        if track_len < window_size:
            continue

        # Count windows for this track
        for start_frame in range(0, track_len - window_size + 1, window_stride):
            end_frame = start_frame + window_size
            window_labels = labels_list[start_frame:end_frame]

            # Skip if window is wrong size
            if len(window_labels) != window_size:
                continue

            # Skip windows with too many unmatched frames
            matched_count = sum(1 for l in window_labels if l != -1)
            if matched_count < window_size * match_threshold:
                continue

            window_info_list.append({
                'track_idx': track_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'labels': window_labels
            })
            total_windows += 1

    print(f"\nTotal valid windows: {total_windows:,}")

    # Create memory-mapped arrays
    print("\nCreating memory-mapped arrays...")
    features_path = output_path / 'features.npy'
    labels_path_out = output_path / 'labels.npy'
    masks_path = output_path / 'masks.npy'
    bands_path = output_path / 'bands.npy'

    # Allocate arrays
    features_mmap = np.lib.format.open_memmap(
        features_path, mode='w+', dtype=np.float32,
        shape=(total_windows, window_size, 4)
    )
    labels_mmap = np.lib.format.open_memmap(
        labels_path_out, mode='w+', dtype=np.int64,
        shape=(total_windows, window_size)
    )
    masks_mmap = np.lib.format.open_memmap(
        masks_path, mode='w+', dtype=np.float32,
        shape=(total_windows, window_size)
    )
    bands_mmap = np.lib.format.open_memmap(
        bands_path, mode='w+', dtype=np.int64,
        shape=(total_windows,)
    )

    # Second pass: extract all windows
    print("\nSecond pass: extracting windows...")
    window_idx = 0

    for window_info in tqdm(window_info_list, desc="Extracting"):
        track_idx = window_info['track_idx']
        track_key = f'track_{track_idx}'
        start_frame = window_info['start_frame']
        end_frame = window_info['end_frame']

        # Read track data
        grp = mix_file[track_key]
        band = grp.attrs['band']

        # Get track lengths for offset calculation
        track_lens = grp['track_lens'][:]
        offset = sum(track_lens[:track_idx])

        # Extract window data
        start = offset + start_frame
        end = offset + end_frame

        freqs = grp['f'][start:end]
        amps = grp['a'][start:end]
        phases = grp['p'][start:end]

        # Create features
        features = np.zeros((window_size, 4), dtype=np.float32)
        actual_len = len(freqs)
        features[:actual_len, 0] = freqs / 96000.0  # Normalized frequency
        features[:actual_len, 1] = np.log1p(amps)    # Log amplitude
        features[:actual_len, 2] = np.cos(phases)    # Phase cos
        features[:actual_len, 3] = np.sin(phases)    # Phase sin

        # Create mask
        mask = np.zeros(window_size, dtype=np.float32)
        mask[:actual_len] = 1.0

        # Process labels
        labels = np.array(window_info['labels'], dtype=np.int64)
        labels[labels == -1] = 5  # Unknown class (5 stems: 0-4, unknown: 5)

        # Store in memory-mapped arrays
        features_mmap[window_idx] = features
        labels_mmap[window_idx] = labels
        masks_mmap[window_idx] = mask
        bands_mmap[window_idx] = band

        window_idx += 1

    # Flush to disk
    print("\nFlushing to disk...")
    del features_mmap, labels_mmap, masks_mmap, bands_mmap

    # Save metadata
    metadata = {
        'n_windows': total_windows,
        'window_size': window_size,
        'window_stride': window_stride,
        'match_threshold': match_threshold,
        'n_stems': 5,
        'features_shape': [total_windows, window_size, 4],
        'labels_shape': [total_windows, window_size],
        'masks_shape': [total_windows, window_size],
        'bands_shape': [total_windows],
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    mix_file.close()

    print(f"\nPreprocessing complete!")
    print(f"Output directory: {output_path}")
    print(f"Total windows: {total_windows:,}")
    print(f"Dataset size: {total_windows * window_size:,} frames")

    # Calculate file sizes
    features_size = features_path.stat().st_size / 1024**3
    labels_size = labels_path_out.stat().st_size / 1024**3
    masks_size = masks_path.stat().st_size / 1024**3
    bands_size = bands_path.stat().st_size / 1024**3
    total_size = features_size + labels_size + masks_size + bands_size

    print(f"\nFile sizes:")
    print(f"  features.npy: {features_size:.2f} GB")
    print(f"  labels.npy: {labels_size:.2f} GB")
    print(f"  masks.npy: {masks_size:.2f} GB")
    print(f"  bands.npy: {bands_size:.2f} GB")
    print(f"  Total: {total_size:.2f} GB")


if __name__ == '__main__':
    preprocess_dataset()
