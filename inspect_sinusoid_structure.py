#!/usr/bin/env python3
"""Inspect how sinusoids are organized in HDF5 files"""

import h5py
import numpy as np
from pathlib import Path

def inspect_h5_structure(h5_path):
    """Examine the sinusoidal data structure in detail"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {h5_path.name}")
    print(f"{'='*60}")

    with h5py.File(h5_path, 'r') as f:
        n_channels = f.attrs.get('ch', 1)
        print(f"Number of channels: {n_channels}")

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            print(f"\nChannel {ch_idx}:")
            print(f"  Available datasets: {list(grp.keys())}")

            if 'len' in grp:
                track_lens = grp['len'][:]
                print(f"  Number of tracks: {len(track_lens)}")
                print(f"  Track length range: {track_lens.min()} - {track_lens.max()}")
                print(f"  Total sinusoids: {track_lens.sum()}")

            if 'f' in grp:
                frequencies = grp['f'][:]
                print(f"  Frequency array shape: {frequencies.shape}")
                print(f"  Frequency range: {frequencies.min():.2f} - {frequencies.max():.2f} Hz")
                print(f"  Unique frequencies: {len(np.unique(frequencies))}")

                # Check if frequencies are sorted or grouped
                freq_diffs = np.diff(frequencies)
                print(f"  Freq diffs range: {freq_diffs.min():.2f} - {freq_diffs.max():.2f}")

            if 'a' in grp:
                amplitudes = grp['a'][:]
                print(f"  Amplitude array shape: {amplitudes.shape}")
                print(f"  Amplitude range: {amplitudes.min():.6f} - {amplitudes.max():.6f}")

            if 'i' in grp:
                frames = grp['i'][:]
                print(f"  Frame array shape: {frames.shape}")
                print(f"  Frame range: {frames.min()} - {frames.max()}")
                print(f"  Unique frames: {len(np.unique(frames))}")

                # Check if frames are sorted
                if len(frames) > 1:
                    is_sorted = np.all(frames[:-1] <= frames[1:])
                    print(f"  Frames are sorted: {is_sorted}")

            # Sample first track to understand organization
            if 'len' in grp and len(grp['len']) > 0:
                print(f"\n  First track (length={track_lens[0]}):")
                first_track_freqs = frequencies[:track_lens[0]]
                first_track_frames = frames[:track_lens[0]]

                print(f"    Frequencies: {first_track_freqs[:10]}")
                print(f"    Frames: {first_track_frames[:10]}")

                # Are sinusoids in this track sorted by frame?
                is_sorted_by_frame = np.all(first_track_frames[:-1] <= first_track_frames[1:])
                print(f"    Sorted by frame: {is_sorted_by_frame}")

                # Or sorted by frequency?
                is_sorted_by_freq = np.all(first_track_freqs[:-1] <= first_track_freqs[1:])
                print(f"    Sorted by frequency: {is_sorted_by_freq}")

                # Check if it's one frequency over time
                unique_freqs_in_track = len(np.unique(first_track_freqs))
                print(f"    Unique frequencies in track: {unique_freqs_in_track}")
                if unique_freqs_in_track <= 3:
                    print(f"    Track appears to follow ONE frequency over time!")

# Check vocals file
vocals_h5 = Path('ajfa/vocals_tracks.h5')
if vocals_h5.exists():
    inspect_h5_structure(vocals_h5)
else:
    print("vocals_tracks.h5 not found")

# Check fullmix too
fullmix_h5 = Path('ajfa/fullmix_tracks.h5')
if fullmix_h5.exists():
    inspect_h5_structure(fullmix_h5)
else:
    print("fullmix_tracks.h5 not found")
