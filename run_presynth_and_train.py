#!/usr/bin/env python3
"""
Pre-synthesize .h5 files and then automatically start training.

Run this before bed and it will:
1. Synthesize all missing .h5 → .wav files
2. Automatically start training when synthesis is done
"""

import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Pre-synthesize and train audio enhancer')
    parser.add_argument('--data-dirs', nargs='+', required=True, help='Song directories')
    parser.add_argument('--stem-names', nargs='+', default=['vocals', 'guitar', 'bass', 'drums'])
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--segment-length', type=float, default=4.0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_enhancer')

    args = parser.parse_args()

    print("=" * 60)
    print("Pre-synthesis + Training Pipeline")
    print("=" * 60)
    print("\nStep 1: Pre-synthesizing .h5 files...")
    print("=" * 60)

    # Run pre-synthesis
    presynth_cmd = [
        sys.executable,
        'presynthesize_h5_files.py',
        '--data-dirs', *args.data_dirs,
        '--stem-names', *args.stem_names,
        '--sample-rate', str(args.sample_rate)
    ]

    result = subprocess.run(presynth_cmd)

    if result.returncode != 0:
        print("\n❌ Pre-synthesis failed!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Step 2: Starting training...")
    print("=" * 60)

    # Run training
    train_cmd = [
        sys.executable,
        'train_h5_audio_enhancer.py',
        '--data-dirs', *args.data_dirs,
        '--stem-names', *args.stem_names,
        '--sample-rate', str(args.sample_rate),
        '--segment-length', str(args.segment_length),
        '--batch-size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--checkpoint-dir', args.checkpoint_dir
    ]

    result = subprocess.run(train_cmd)

    if result.returncode != 0:
        print("\n❌ Training failed!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
