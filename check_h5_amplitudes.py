#!/usr/bin/env python3
"""Check amplitude values in raw .h5 files"""

import h5py
import numpy as np
from pathlib import Path

def check_h5_file(h5_path):
    """Check amplitude range in an HDF5 file"""
    print(f"\nChecking: {h5_path.name}")

    with h5py.File(h5_path, 'r') as f:
        n_channels = f.attrs.get('ch', 1)
        all_amps = []

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            if 'a' in grp:
                amplitudes = grp['a'][:]
                all_amps.extend(amplitudes)

        if all_amps:
            all_amps = np.array(all_amps)
            print(f"  Amplitude range: {all_amps.min():.6f} - {all_amps.max():.6f}")
            print(f"  Non-zero amps: {(all_amps > 0).sum()} / {len(all_amps)}")
            print(f"  Mean amplitude: {all_amps.mean():.6f}")
        else:
            print(f"  No amplitude data found!")

# Check all .h5 files in ajfa directory
data_dir = Path('ajfa')
if data_dir.exists():
    print("=" * 60)
    print("Checking raw .h5 files for amplitude data")
    print("=" * 60)

    for h5_file in sorted(data_dir.glob('*_tracks.h5')):
        check_h5_file(h5_file)
else:
    print("ajfa directory not found")
