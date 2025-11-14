#!/usr/bin/env python3
"""Inspect HDF5 file structure"""

import h5py

with h5py.File('fullmix_tracks.h5', 'r') as f:
    print("HDF5 file structure:\n")

    for key in f.keys():
        print(f"Group: {key}")
        grp = f[key]

        print(f"  Attributes: {dict(grp.attrs)}")
        print(f"  Datasets:")
        for dataset_name in grp.keys():
            dataset = grp[dataset_name]
            print(f"    {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")

        # Show track_lens if it exists
        if 'track_lens' in grp:
            track_lens = grp['track_lens'][:]
            print(f"  track_lens: {len(track_lens)} tracks, sum={sum(track_lens)}")
            print(f"    First 10: {track_lens[:10].tolist()}")

        print()
