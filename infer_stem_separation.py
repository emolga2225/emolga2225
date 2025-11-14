#!/usr/bin/env python3
"""
Inference script for stem separation using trained model.

Takes a mix audio file, extracts sinusoidal tracks, classifies them,
and synthesizes separate stem audio files.
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Import the model architecture
import sys
sys.path.append('.')
from train_stem_separator_fast import FramewiseStemClassifier


STEM_NAMES = ['vocals', 'guitar', 'bass', 'other', 'drums']


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']

    # Create model
    model = FramewiseStemClassifier(
        n_stems=config['n_stems'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        window_size=config['window_size']
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (epoch {checkpoint['epoch']}, acc: {checkpoint['train_acc']:.2f}%)")

    return model, config


def classify_tracks(model, tracks_h5_path, config, device='cuda'):
    """
    Classify all tracks in an HDF5 file.

    Returns:
        track_predictions: dict mapping track_idx -> {
            'stem': predicted stem (0-4),
            'confidence': confidence score,
            'band': frequency band
        }
    """
    print(f"\nClassifying tracks from {tracks_h5_path}...")

    window_size = config['window_size']
    track_predictions = {}

    with h5py.File(tracks_h5_path, 'r') as f:
        # Handle both compact (c0, c1) and individual track formats
        if 'c0' in f or 'c1' in f:
            # Compact format
            all_tracks = []
            for group_name in f.keys():
                grp = f[group_name]
                n_tracks = len(grp['id'][:])

                for local_idx in range(n_tracks):
                    track_id = int(grp['id'][local_idx])
                    band = int(grp['b'][local_idx])
                    start_idx = int(grp['s'][local_idx])
                    end_idx = int(grp['e'][local_idx])

                    freqs = grp['f'][start_idx:end_idx]
                    amps = grp['a'][start_idx:end_idx]
                    phases = grp['p'][start_idx:end_idx]

                    all_tracks.append({
                        'id': track_id,
                        'band': band,
                        'freqs': freqs,
                        'amps': amps,
                        'phases': phases
                    })
        else:
            # Individual track format
            all_tracks = []
            for track_key in f.keys():
                grp = f[track_key]
                track_idx = int(track_key.split('_')[1])
                band = int(grp.attrs.get('band', 0))

                # Get all data for this track
                track_lens = grp['track_lens'][:]
                offset = sum(track_lens[:track_idx])
                track_len = track_lens[track_idx]

                freqs = grp['f'][offset:offset+track_len]
                amps = grp['a'][offset:offset+track_len]
                phases = grp['p'][offset:offset+track_len]

                all_tracks.append({
                    'id': track_idx,
                    'band': band,
                    'freqs': freqs,
                    'amps': amps,
                    'phases': phases
                })

        print(f"Found {len(all_tracks)} tracks")

        # Classify each track
        with torch.no_grad():
            for track in tqdm(all_tracks, desc="Classifying"):
                track_id = track['id']
                band = track['band']
                freqs = track['freqs']
                amps = track['amps']
                phases = track['phases']

                n_frames = len(freqs)

                # Skip very short tracks
                if n_frames < window_size:
                    track_predictions[track_id] = {
                        'stem': 5,  # Unknown
                        'confidence': 0.0,
                        'band': band
                    }
                    continue

                # Create features for all frames
                features = np.zeros((n_frames, 4), dtype=np.float32)
                features[:, 0] = freqs / 96000.0  # Normalized frequency
                features[:, 1] = np.log1p(amps)    # Log amplitude
                features[:, 2] = np.cos(phases)    # Phase cos
                features[:, 3] = np.sin(phases)    # Phase sin

                # Classify using sliding windows
                stem_votes = np.zeros(6)  # 5 stems + unknown
                total_windows = 0

                for start_frame in range(0, n_frames - window_size + 1, window_size // 2):
                    end_frame = start_frame + window_size
                    window_features = features[start_frame:end_frame]

                    # Prepare batch
                    window_tensor = torch.from_numpy(window_features).unsqueeze(0).to(device)
                    mask_tensor = torch.ones(1, window_size).to(device)
                    band_tensor = torch.tensor([[band]], dtype=torch.long).to(device)

                    # Classify
                    logits = model(window_tensor, mask_tensor, band_tensor)
                    probs = torch.softmax(logits, dim=-1)

                    # Vote across frames in window
                    window_probs = probs[0].cpu().numpy()  # [window_size, n_stems]
                    stem_votes += window_probs.sum(axis=0)
                    total_windows += window_size

                # Get final prediction
                stem_probs = stem_votes / total_windows
                predicted_stem = int(stem_probs.argmax())
                confidence = float(stem_probs[predicted_stem])

                track_predictions[track_id] = {
                    'stem': predicted_stem,
                    'confidence': confidence,
                    'band': band
                }

    return track_predictions


def synthesize_stems(tracks_h5_path, track_predictions, output_dir, sr=48000):
    """
    Synthesize separate audio files for each stem.

    Groups tracks by predicted stem and synthesizes audio.
    """
    print(f"\nSynthesizing stems to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Group tracks by stem
    stem_tracks = {i: [] for i in range(6)}  # 0-4: stems, 5: unknown

    for track_id, pred in track_predictions.items():
        stem_tracks[pred['stem']].append(track_id)

    # Print statistics
    print("\nPrediction statistics:")
    for stem_idx, track_ids in stem_tracks.items():
        stem_name = STEM_NAMES[stem_idx] if stem_idx < 5 else 'unknown'
        print(f"  {stem_name}: {len(track_ids)} tracks")

    # Synthesize each stem
    with h5py.File(tracks_h5_path, 'r') as f:
        for stem_idx, track_ids in stem_tracks.items():
            if len(track_ids) == 0:
                continue

            stem_name = STEM_NAMES[stem_idx] if stem_idx < 5 else 'unknown'
            print(f"\nSynthesizing {stem_name}...")

            # TODO: Implement actual synthesis
            # For now, just create a placeholder
            # You'll need to use your existing synthesis code from HybridResolutionSinusoidalExtractor

            print(f"  Would synthesize {len(track_ids)} tracks for {stem_name}")
            # audio = synthesize_sinusoidal_tracks(f, track_ids, sr)
            # sf.write(output_path / f'{stem_name}.wav', audio, sr)

    print("\nStem synthesis complete!")


def main():
    parser = argparse.ArgumentParser(description='Stem separation inference')
    parser.add_argument('input_h5', help='Input HDF5 file with extracted tracks')
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint_epoch_10.pt',
                        help='Model checkpoint to use')
    parser.add_argument('--output-dir', default='separated_stems',
                        help='Output directory for separated stems')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.checkpoint, device=args.device)

    # Classify tracks
    track_predictions = classify_tracks(model, args.input_h5, config, device=args.device)

    # Save predictions
    predictions_path = Path(args.output_dir) / 'track_predictions.json'
    predictions_path.parent.mkdir(exist_ok=True, parents=True)

    predictions_serializable = {
        str(track_id): {
            'stem': pred['stem'],
            'stem_name': STEM_NAMES[pred['stem']] if pred['stem'] < 5 else 'unknown',
            'confidence': pred['confidence'],
            'band': pred['band']
        }
        for track_id, pred in track_predictions.items()
    }

    with open(predictions_path, 'w') as f:
        json.dump(predictions_serializable, f, indent=2)

    print(f"\nSaved predictions to {predictions_path}")

    # Synthesize stems
    synthesize_stems(args.input_h5, track_predictions, args.output_dir)


if __name__ == '__main__':
    main()
