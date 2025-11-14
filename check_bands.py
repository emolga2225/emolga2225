#!/usr/bin/env python3
"""Check band values in preprocessed data"""

import numpy as np
import json

# Load preprocessed bands
bands = np.load('preprocessed_data/bands.npy', mmap_mode='r')

print(f"Total windows: {len(bands):,}")
print(f"Unique band values: {sorted(set(bands))}")
print(f"Band value range: {bands.min()} to {bands.max()}")
print(f"\nBand value counts:")
unique, counts = np.unique(bands, return_counts=True)
for band, count in zip(unique, counts):
    print(f"  Band {band}: {count:,} windows ({100*count/len(bands):.1f}%)")

# Check metadata
with open('preprocessed_data/metadata.json') as f:
    metadata = json.load(f)
print(f"\nMetadata n_stems: {metadata['n_stems']}")
