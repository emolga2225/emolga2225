#!/usr/bin/env python3
"""
Inspect HDF5 files to determine actual max sinusoids per frame.
This helps set realistic max_sinusoids_per_frame instead of guessing.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


def analyze_sinusoids_per_frame(h5_path, hop_length=512):
    """Count sinusoids per frame in an HDF5 file"""

    if not h5_path.exists():
        return {}

    frame_counts = defaultdict(int)

    with h5py.File(h5_path, 'r') as f:
        if 'c0' not in f:
            return frame_counts

        grp = f['c0']
        track_lens = grp['len'][:]
        frames = grp['i'][:]

        # Process each track
        offset = 0
        for track_len in track_lens:
            track_frames = frames[offset:offset + track_len]

            # Count sinusoids per frame
            for frame_idx in track_frames:
                frame_counts[int(frame_idx)] += 1

            offset += track_len

    return frame_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', nargs='+', required=True)
    parser.add_argument('--stem-names', nargs='+',
                       default=['vocals', 'guitar', 'bass', 'drums', 'song'])
    args = parser.parse_args()

    all_max_counts = []

    for data_dir in args.data_dirs:
        data_dir = Path(data_dir)
        print(f"\n{'='*60}")
        print(f"Analyzing: {data_dir}")
        print(f"{'='*60}")

        # Check fullmix
        fullmix_h5 = data_dir / 'fullmix_tracks.h5'
        if not fullmix_h5.exists():
            fullmix_h5 = data_dir / 'fullmix.h5'

        if fullmix_h5.exists():
            print(f"\nFullmix: {fullmix_h5.name}")
            frame_counts = analyze_sinusoids_per_frame(fullmix_h5)

            if frame_counts:
                counts = list(frame_counts.values())
                max_count = max(counts)
                avg_count = np.mean(counts)
                median_count = np.median(counts)
                p95_count = np.percentile(counts, 95)
                p99_count = np.percentile(counts, 99)

                print(f"  Total frames: {len(counts)}")
                print(f"  Sinusoids per frame:")
                print(f"    Max:    {max_count:,}")
                print(f"    99th %: {int(p99_count):,}")
                print(f"    95th %: {int(p95_count):,}")
                print(f"    Median: {int(median_count):,}")
                print(f"    Mean:   {int(avg_count):,}")

                all_max_counts.append(max_count)

        # Check each stem
        for stem_name in args.stem_names:
            stem_h5s = []

            # Handle drums (multiple files)
            if stem_name == 'drums':
                for i in range(1, 10):
                    drum_h5 = data_dir / f'drums_{i}_tracks.h5'
                    if drum_h5.exists():
                        stem_h5s.append(drum_h5)
                if not stem_h5s:
                    drum_h5 = data_dir / 'drums_tracks.h5'
                    if drum_h5.exists():
                        stem_h5s.append(drum_h5)

            # Handle bass/rhythm
            elif stem_name == 'bass':
                for name in ['bass_tracks.h5', 'rhythm_tracks.h5']:
                    stem_h5 = data_dir / name
                    if stem_h5.exists():
                        stem_h5s.append(stem_h5)
                        break

            # Regular stems
            else:
                for suffix in ['_tracks.h5', '.h5']:
                    stem_h5 = data_dir / f'{stem_name}{suffix}'
                    if stem_h5.exists():
                        stem_h5s.append(stem_h5)
                        break

            for stem_h5 in stem_h5s:
                print(f"\n{stem_name.capitalize()}: {stem_h5.name}")
                frame_counts = analyze_sinusoids_per_frame(stem_h5)

                if frame_counts:
                    counts = list(frame_counts.values())
                    max_count = max(counts)
                    avg_count = np.mean(counts)
                    median_count = np.median(counts)
                    p95_count = np.percentile(counts, 95)
                    p99_count = np.percentile(counts, 99)

                    print(f"  Total frames: {len(counts)}")
                    print(f"  Sinusoids per frame:")
                    print(f"    Max:    {max_count:,}")
                    print(f"    99th %: {int(p99_count):,}")
                    print(f"    95th %: {int(p95_count):,}")
                    print(f"    Median: {int(median_count):,}")
                    print(f"    Mean:   {int(avg_count):,}")

                    all_max_counts.append(max_count)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if all_max_counts:
        overall_max = max(all_max_counts)
        print(f"\nOverall maximum sinusoids per frame: {overall_max:,}")
        print(f"\nRecommended --max-sinusoids values:")
        print(f"  Conservative (99th percentile): Use inspection per-file")
        print(f"  Safe (covers max):              {overall_max:,}")
        print(f"  Memory-efficient:               {min(overall_max, 5000):,} (may truncate some frames)")

        # Memory estimates
        print(f"\nMemory estimates (float32, per sample in batch):")
        for chunk_frames in [43, 86, 172, 344]:
            duration = chunk_frames * 512 / 44100
            for max_sines in [1000, 2000, 5000, overall_max]:
                size_mb = (chunk_frames * max_sines * 3 * 4) / (1024**2)  # 4 bytes per float32
                size_stems = size_mb * 5  # 5 stems
                print(f"  {int(duration*1000):4d}ms ({chunk_frames:3d} frames), {max_sines:6,} sinusoids: {size_mb:7.1f} MB fullmix, {size_stems:7.1f} MB stems")


if __name__ == "__main__":
    main()
