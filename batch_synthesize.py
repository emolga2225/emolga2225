#!/usr/bin/env python3
"""
Batch synthesize all stems from HDF5 tracks.

Takes a directory of separated HDF5 files and synthesizes all to WAV.
"""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Batch synthesize all stems")
    parser.add_argument('--input-dir', required=True, help='Directory with HDF5 tracks (e.g., separated_h5/)')
    parser.add_argument('--output-dir', default='audio_output', help='Output directory for WAV files')
    parser.add_argument('--hop-length', type=int, default=512, help='Hop length (default: 512)')
    parser.add_argument('--stems', nargs='+', default=['vocals', 'guitar', 'bass', 'drums', 'song'],
                       help='Stem names to process')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Batch synthesizing stems from: {input_dir}")
    print(f"Output directory: {output_dir}\n")

    for stem_name in args.stems:
        h5_file = input_dir / f"{stem_name}_tracks.h5"

        if not h5_file.exists():
            print(f"[SKIP] {stem_name}: {h5_file} not found")
            continue

        output_file = output_dir / f"{stem_name}.wav"

        print(f"Synthesizing {stem_name}...")

        # Run synthesize_from_h5.py
        cmd = [
            'python', 'synthesize_from_h5.py',
            '--input', str(h5_file),
            '--output', str(output_file),
            '--hop-length', str(args.hop_length)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] synthesizing {stem_name}:")
            print(result.stderr)
        else:
            print(f"[OK] Saved: {output_file}\n")

    print(f"\n[DONE] All stems saved to: {output_dir}")


if __name__ == "__main__":
    main()
