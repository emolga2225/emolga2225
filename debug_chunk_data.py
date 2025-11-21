#!/usr/bin/env python3
"""
Debug script to inspect preprocessed chunk data.
"""

import numpy as np
from pathlib import Path

def inspect_chunks(data_dir):
    """Inspect what's in the preprocessed chunks"""
    data_dir = Path(data_dir)
    chunks_dir = data_dir / 'sinusoid_chunks'

    if not chunks_dir.exists():
        print(f"No chunks directory found in {data_dir}")
        return

    chunk_files = sorted(chunks_dir.glob('chunk_*.npz'))

    print(f"\n{data_dir.name}: {len(chunk_files)} chunks")
    print("=" * 60)

    # Check first few chunks
    for i, chunk_file in enumerate(chunk_files[:5]):
        data = np.load(chunk_file)

        print(f"\nChunk {i}: {chunk_file.name}")
        print(f"  Keys: {list(data.keys())}")

        # Check fullmix
        fullmix = data['fullmix']
        print(f"  fullmix: shape={fullmix.shape}, non-zero={np.count_nonzero(fullmix[:, 1])}")
        if len(fullmix) > 0:
            print(f"    freq range: [{fullmix[:, 0].min():.1f}, {fullmix[:, 0].max():.1f}]")
            print(f"    amp range: [{fullmix[:, 1].min():.4f}, {fullmix[:, 1].max():.4f}]")
            print(f"    frame range: [{fullmix[:, 2].min():.1f}, {fullmix[:, 2].max():.1f}]")

        # Check stems
        for stem_name in ['vocals', 'guitar', 'bass', 'drums']:
            if stem_name in data:
                stem = data[stem_name]
                non_zero = np.count_nonzero(stem[:, 1]) if len(stem) > 0 else 0
                print(f"  {stem_name}: shape={stem.shape}, non-zero={non_zero}")
                if non_zero > 0:
                    print(f"    amp range: [{stem[:, 1].min():.4f}, {stem[:, 1].max():.4f}]")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        dirs = ['ajfa', 'blackned', 'dyerseve']

    for data_dir in dirs:
        inspect_chunks(data_dir)
