#!/usr/bin/env python3
"""Detailed debug for preprocessing"""

import json
import h5py

# Load labels
with open('stem_labels.json') as f:
    labels_data = json.load(f)

print(f"Labels JSON: {len(labels_data)} tracks")

# Open HDF5
with h5py.File('fullmix_tracks.h5', 'r') as mix_file:
    print(f"HDF5 file: {len(mix_file.keys())} groups\n")

    # Check first 10 tracks
    checked = 0
    matched = 0

    for track_idx_str, track_data in list(labels_data.items())[:10]:
        track_idx = int(track_idx_str)
        track_key = f'track_{track_idx}'

        print(f"Track {track_idx_str}:")
        print(f"  Looking for HDF5 key: '{track_key}'")
        print(f"  Key exists in HDF5: {track_key in mix_file}")

        if track_key in mix_file:
            matched += 1
            labels_list = track_data['labels']
            band = track_data['band']
            print(f"  Labels length: {len(labels_list)}")
            print(f"  Band: {band}")

            # Check how many frames are matched (not -1)
            matched_frames = sum(1 for l in labels_list if l != -1)
            print(f"  Matched frames: {matched_frames} / {len(labels_list)} ({100*matched_frames/len(labels_list):.1f}%)")

            # Check window_size requirements
            window_size = 32
            if len(labels_list) >= window_size:
                print(f"  ✓ Long enough for windows (>= {window_size})")

                # Check first window
                first_window = labels_list[:window_size]
                matched_in_window = sum(1 for l in first_window if l != -1)
                match_ratio = matched_in_window / window_size
                print(f"  First window: {matched_in_window}/{window_size} matched ({match_ratio:.2%})")
                print(f"  Passes 50% threshold: {match_ratio >= 0.5}")
            else:
                print(f"  ✗ Too short (< {window_size})")
        else:
            print(f"  ✗ Not found in HDF5")

            # Show what keys DO exist
            if checked == 0:
                print(f"\n  Available HDF5 keys (first 10): {list(mix_file.keys())[:10]}")

        print()
        checked += 1

    print(f"\nSummary: {matched}/{checked} tracks found in HDF5")
