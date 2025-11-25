#!/usr/bin/env python3
"""
Quick script to check how many sinusoids are actually present in frames.
This helps determine if we can reduce max_sinusoids to speed up training.
"""

import numpy as np
from pathlib import Path
import json

def check_sinusoid_density(preprocessed_dir, num_samples=1000):
    """Check average number of non-zero sinusoids per frame"""

    preprocessed_dir = Path(preprocessed_dir)

    # Load index
    index_file = preprocessed_dir / 'index.json'
    with open(index_file, 'r') as f:
        index = json.load(f)

    frames = index['frames']
    max_sinusoids = index['max_sinusoids']

    print(f"Max sinusoids: {max_sinusoids}")
    print(f"Total frames: {len(frames):,}")
    print(f"Sampling {num_samples} frames...\n")

    # Sample random frames
    sample_indices = np.random.choice(len(frames), min(num_samples, len(frames)), replace=False)

    sinusoid_counts = []

    for idx in sample_indices:
        frame_info = frames[idx]
        frame_file = preprocessed_dir / frame_info['file']

        # Load frame
        data = np.load(frame_file)
        fullmix = data['fullmix']  # (max_sinusoids, 3)

        # Count non-zero sinusoids (check if frequency > 0)
        non_zero = (fullmix[:, 0] > 0).sum()
        sinusoid_counts.append(non_zero)

    # Statistics
    counts = np.array(sinusoid_counts)

    print(f"Sinusoid density statistics:")
    print(f"  Mean: {counts.mean():.1f}")
    print(f"  Median: {np.median(counts):.1f}")
    print(f"  Min: {counts.min()}")
    print(f"  Max: {counts.max()}")
    print(f"  95th percentile: {np.percentile(counts, 95):.1f}")
    print(f"  99th percentile: {np.percentile(counts, 99):.1f}")
    print(f"\nUtilization: {counts.mean() / max_sinusoids * 100:.1f}% of max_sinusoids")

    # Suggest optimal max_sinusoids
    p95 = np.percentile(counts, 95)
    p99 = np.percentile(counts, 99)

    print(f"\nRecommendations:")
    print(f"  Conservative (covers 99% of frames): max_sinusoids={int(p99)}")
    print(f"  Balanced (covers 95% of frames): max_sinusoids={int(p95)}")
    print(f"  Aggressive (median): max_sinusoids={int(np.median(counts))}")

    # Show speedup potential
    for name, val in [("99th percentile", p99), ("95th percentile", p95), ("Median", np.median(counts))]:
        speedup = (max_sinusoids / val) ** 2  # Attention is O(n^2)
        print(f"  Using {int(val)} would be ~{speedup:.1f}x faster")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-dir', default='preprocessed_data/')
    parser.add_argument('--num-samples', type=int, default=1000)
    args = parser.parse_args()

    check_sinusoid_density(args.preprocessed_dir, args.num_samples)
