#!/usr/bin/env python3
"""
Train Both Models Overnight

Runs all preprocessing and training for both Model 1 and Model 2:
1. Preprocess .h5 → dense grids (.npy) for Model 1
2. Pre-synthesize .h5 → audio (.wav) for Model 2
3. Train Model 1 (stem separator)
4. Train Model 2 (audio enhancer)

Start this before bed and wake up to two trained models!
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(description, cmd):
    """Run a command and handle errors"""
    print("\n" + "=" * 60)
    print(f"{description}")
    print("=" * 60)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ {description} failed!")
        return False

    print(f"\n✓ {description} complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Preprocess and train both models')
    parser.add_argument('--data-dirs', nargs='+', required=True, help='Song directories')
    parser.add_argument('--stem-names', nargs='+', default=['vocals', 'guitar', 'bass', 'drums'])
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)

    args = parser.parse_args()

    print("=" * 60)
    print("OVERNIGHT TRAINING PIPELINE")
    print("=" * 60)
    print(f"Songs: {args.data_dirs}")
    print(f"Stems: {args.stem_names}")
    print(f"Epochs per model: {args.epochs}")
    print("\nThis will:")
    print("  1. Preprocess .h5 → grids for Model 1")
    print("  2. Pre-synthesize .h5 → audio for Model 2")
    print("  3. Train Model 1 (stem separator)")
    print("  4. Train Model 2 (audio enhancer)")
    print("\nGo to bed - wake up to trained models!")

    # Step 1: Preprocess .h5 to grids for Model 1
    if not run_command(
        "Step 1/4: Preprocessing .h5 → grids (Model 1)",
        [
            sys.executable,
            'preprocess_h5_to_spectrograms.py',
            '--data-dirs', *args.data_dirs,
            '--stem-names', *args.stem_names,
            '--sample-rate', str(args.sample_rate)
        ]
    ):
        sys.exit(1)

    # Step 2: Pre-synthesize .h5 to audio for Model 2
    if not run_command(
        "Step 2/4: Pre-synthesizing .h5 → audio (Model 2)",
        [
            sys.executable,
            'presynthesize_h5_files.py',
            '--data-dirs', *args.data_dirs,
            '--stem-names', *args.stem_names,
            '--sample-rate', str(args.sample_rate)
        ]
    ):
        sys.exit(1)

    # Step 3: Train Model 1 (stem separator)
    if not run_command(
        "Step 3/4: Training Model 1 (stem separator)",
        [
            sys.executable,
            'train_stem_separator.py',
            '--data-dirs', *args.data_dirs,
            '--stem-names', *args.stem_names,
            '--batch-size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--checkpoint-dir', 'checkpoints_separator'
        ]
    ):
        sys.exit(1)

    # Step 4: Train Model 2 (audio enhancer)
    if not run_command(
        "Step 4/4: Training Model 2 (audio enhancer)",
        [
            sys.executable,
            'train_h5_audio_enhancer.py',
            '--data-dirs', *args.data_dirs,
            '--stem-names', *args.stem_names,
            '--sample-rate', str(args.sample_rate),
            '--batch-size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--checkpoint-dir', 'checkpoints_enhancer'
        ]
    ):
        sys.exit(1)

    # Done!
    print("\n" + "=" * 60)
    print("✓✓✓ ALL COMPLETE! ✓✓✓")
    print("=" * 60)
    print("\nYou now have two trained models:")
    print("  1. Model 1 (stem separator): checkpoints_separator/")
    print("  2. Model 2 (audio enhancer): checkpoints_enhancer/")
    print("\nGood morning! 🌅")


if __name__ == '__main__':
    main()
