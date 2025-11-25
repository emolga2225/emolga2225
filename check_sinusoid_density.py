#!/usr/bin/env python3
"""
Quick script to check how many sinusoids are actually present in frames.
This helps determine if we can reduce max_sinusoids to speed up training.
"""

import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def check_sinusoid_density(preprocessed_dir, num_samples=None):
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

    # Determine sample size
    if num_samples is None:
        num_samples = len(frames)
        sample_indices = range(len(frames))
        print(f"Analyzing ALL {len(frames):,} frames...\n")
    else:
        sample_indices = np.random.choice(len(frames), min(num_samples, len(frames)), replace=False)
        print(f"Sampling {num_samples:,} frames...\n")

    sinusoid_counts = []

    for idx in tqdm(sample_indices, desc="Analyzing frames"):
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

    print(f"\nSinusoid density statistics:")
    print(f"  Mean: {counts.mean():.1f}")
    print(f"  Median: {np.median(counts):.1f}")
    print(f"  Min: {counts.min()}")
    print(f"  Max: {counts.max()}")
    print(f"  90th percentile: {np.percentile(counts, 90):.1f}")
    print(f"  95th percentile: {np.percentile(counts, 95):.1f}")
    print(f"  99th percentile: {np.percentile(counts, 99):.1f}")
    print(f"  99.9th percentile: {np.percentile(counts, 99.9):.1f}")
    print(f"  99.99th percentile: {np.percentile(counts, 99.99):.1f}")
    print(f"\nUtilization: {counts.mean() / max_sinusoids * 100:.1f}% of max_sinusoids")

    # Suggest optimal max_sinusoids
    p90 = np.percentile(counts, 90)
    p95 = np.percentile(counts, 95)
    p99 = np.percentile(counts, 99)
    p999 = np.percentile(counts, 99.9)
    max_val = counts.max()

    print(f"\nRecommendations:")
    print(f"  Ultra-conservative (100% coverage): max_sinusoids={int(max_val)}")
    print(f"  Very conservative (99.9% coverage): max_sinusoids={int(p999)}")
    print(f"  Conservative (99% coverage): max_sinusoids={int(p99)}")
    print(f"  Balanced (95% coverage): max_sinusoids={int(p95)}")
    print(f"  Aggressive (90% coverage): max_sinusoids={int(p90)}")

    print(f"\nSpeedup analysis (attention is O(n²)):")
    for name, val in [("Max (100%)", max_val), ("99.9%", p999), ("99%", p99), ("95%", p95), ("90%", p90)]:
        speedup = (max_sinusoids / val) ** 2  # Attention is O(n^2)
        print(f"  {name:20s} -> max_sinusoids={int(val):4d} -> {speedup:.1f}x faster")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-dir', default='preprocessed_data/')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of frames to sample (default: all frames)')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all frames (same as --num-samples=None)')
    args = parser.parse_args()

    if args.all:
        num_samples = None
    else:
        num_samples = args.num_samples

    check_sinusoid_density(args.preprocessed_dir, num_samples)
