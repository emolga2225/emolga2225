#!/usr/bin/env python3
"""Debug preprocessing issue"""

import json
import h5py

# Check labels structure
with open('stem_labels.json') as f:
    labels = json.load(f)

print('Labels JSON structure:')
print(f'  Number of tracks: {len(labels)}')
print(f'  First few keys: {list(labels.keys())[:5]}')
if len(labels) > 0:
    first_key = list(labels.keys())[0]
    print(f'  First key: "{first_key}" (type: {type(first_key)})')
    print(f'  First value length: {len(labels[first_key])}')

# Check HDF5 structure
with h5py.File('fullmix_tracks.h5', 'r') as f:
    print(f'\nHDF5 structure:')
    print(f'  Number of groups: {len(f.keys())}')
    print(f'  First few keys: {list(f.keys())[:5]}')

    if len(f.keys()) > 0:
        first_key = list(f.keys())[0]
        print(f'\n  First group: "{first_key}"')
        print(f'  First group attrs: {dict(f[first_key].attrs)}')
        print(f'  First group datasets: {list(f[first_key].keys())}')

        # Check track_lens
        if 'track_lens' in f[first_key]:
            track_lens = f[first_key]['track_lens'][:]
            print(f'  track_lens shape: {track_lens.shape}')
            print(f'  track_lens first 5: {track_lens[:5]}')

# Check for key mismatch
print(f'\nChecking for key mismatches:')
with h5py.File('fullmix_tracks.h5', 'r') as f:
    for track_idx_str, labels_list in list(labels.items())[:5]:
        track_idx = int(track_idx_str)
        track_key = f'track_{track_idx}'
        exists = track_key in f
        print(f'  Label key "{track_idx_str}" -> HDF5 key "{track_key}": exists={exists}')
