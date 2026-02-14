#!/usr/bin/env python3
"""
Complete pipeline to prepare NES training data from NSF files.

This script:
1. Renders NSF channels to WAV files
2. Extracts sinusoidal features to HDF5
3. Creates training chunks with proper stem labeling

Usage:
    python prepare_nes_training_data.py game.nsf --track 0 --duration 120
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        return False

    print(f"\n[OK] {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare NES training data from NSF files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('nsf_file', type=str, help='Path to NSF file')
    parser.add_argument('-t', '--track', type=int, default=0,
                        help='Track number (default: 0)')
    parser.add_argument('-d', '--duration', type=float, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: nsf_data/<nsf_name>)')
    parser.add_argument('--sample-rate', type=int, default=48000,
                        help='Sample rate (default: 48000)')
    parser.add_argument('--chunk-duration', type=float, default=0.032,
                        help='Training chunk duration (default: 0.032s = 3 frames)')

    args = parser.parse_args()

    nsf_path = Path(args.nsf_file)
    if not nsf_path.exists():
        print(f"ERROR: NSF file not found: {args.nsf_file}")
        sys.exit(1)

    nsf_name = nsf_path.stem

    # Set up directory structure
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path('nsf_data') / nsf_name

    wav_dir = base_dir / 'wav'
    h5_dir = base_dir / 'h5'
    chunks_dir = base_dir / 'chunks'

    print(f"NES Training Data Pipeline")
    print(f"NSF: {nsf_path}")
    print(f"Track: {args.track}, Duration: {args.duration}s")
    print(f"Output: {base_dir}")

    # Step 1: Render NSF channels
    if not run_command(
        ['python', 'render_nsf_channels.py',
         str(nsf_path),
         '-o', str(wav_dir),
         '-t', str(args.track),
         '-d', str(args.duration),
         '-r', str(args.sample_rate)],
        "Step 1: Rendering NSF channels"
    ):
        sys.exit(1)

    # Step 2: Extract sinusoidal features for each channel
    wav_files = list(wav_dir.glob('*.wav'))
    if len(wav_files) != 5:
        print(f"\n[ERROR] Expected 5 WAV files, found {len(wav_files)}")
        sys.exit(1)

    h5_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Step 2: Extracting sinusoidal features")
    print(f"{'='*60}")

    for wav_file in sorted(wav_files):
        print(f"\nProcessing: {wav_file.name}")

        # Output HDF5 to h5_dir with same base name
        h5_output = h5_dir / wav_file.with_suffix('.h5').name

        result = subprocess.run(
            ['python', 'ssq_sinusoidal_extractor.py',
             str(wav_file),
             '--output', str(h5_output)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[ERROR] Failed to extract: {wav_file.name}")
            print(result.stderr)
            sys.exit(1)

        print(f"  [OK] Created: {h5_output.name}")

    print(f"\n[OK] All sinusoidal features extracted")

    # Step 3: Create training chunks
    # Map NES channels to stem names
    stem_mapping = {
        'pulse1': 'vocals',     # Melody
        'pulse2': 'guitar',     # Harmony
        'triangle': 'bass',     # Bass line
        'noise': 'drums',       # Percussion
        'mix': 'song'           # Full mix
    }

    print(f"\n{'='*60}")
    print(f"Step 3: Creating training chunks")
    print(f"{'='*60}")
    print(f"\nChannel mapping:")
    for nes_ch, stem_name in stem_mapping.items():
        print(f"  {nes_ch:10s} -> {stem_name}")

    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Find the HDF5 files and map them to stem names
    h5_args = []
    for nes_channel, stem_name in stem_mapping.items():
        h5_file = h5_dir / f"{nsf_name}_{nes_channel}.h5"
        if not h5_file.exists():
            print(f"\n[ERROR] Missing HDF5 file: {h5_file}")
            sys.exit(1)
        h5_args.extend(['--' + stem_name, str(h5_file)])

    output_file = chunks_dir / f"{nsf_name}_chunks.h5"

    if not run_command(
        ['python', 'create_training_data_optimized.py',
         *h5_args,
         '--output', str(output_file),
         '--chunk-duration', str(args.chunk_duration),
         '--max-sinusoids', '900'],
        "Creating training chunks"
    ):
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nTraining data ready: {output_file}")
    print(f"\nTo train the model:")
    print(f"  python train_stem_separator_fast.py \\")
    print(f"    --data-file {output_file} \\")
    print(f"    --max-sinusoids 900 \\")
    print(f"    --chunk-duration {args.chunk_duration} \\")
    print(f"    --batch-size 32 \\")
    print(f"    --epochs 100")
    print(f"\nExpected behavior with NES data:")
    print(f"  - Should converge MUCH faster than complex music")
    print(f"  - Loss should drop below 5 within 50-100 epochs")
    print(f"  - Each stem should have very few sinusoids (5-15)")
    print(f"  - Easy to verify stem separation is working")


if __name__ == '__main__':
    main()
