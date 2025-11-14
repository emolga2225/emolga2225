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

        # Classify tracks in batches for speed
        batch_size = 64
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(all_tracks), batch_size), desc="Classifying"):
                batch_tracks = all_tracks[batch_start:batch_start + batch_size]

                batch_features = []
                batch_masks = []
                batch_bands = []
                batch_track_info = []

                for track in batch_tracks:
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

                    # Use middle window only for speed (instead of sliding windows)
                    mid_frame = n_frames // 2
                    start_frame = max(0, mid_frame - window_size // 2)
                    end_frame = start_frame + window_size

                    if end_frame > n_frames:
                        end_frame = n_frames
                        start_frame = max(0, end_frame - window_size)

                    window_freqs = freqs[start_frame:end_frame]
                    window_amps = amps[start_frame:end_frame]
                    window_phases = phases[start_frame:end_frame]

                    # Create features
                    features = np.zeros((window_size, 4), dtype=np.float32)
                    actual_len = len(window_freqs)
                    features[:actual_len, 0] = window_freqs / 96000.0  # Normalized frequency
                    features[:actual_len, 1] = np.log1p(window_amps)    # Log amplitude
                    features[:actual_len, 2] = np.cos(window_phases)    # Phase cos
                    features[:actual_len, 3] = np.sin(window_phases)    # Phase sin

                    mask = np.zeros(window_size, dtype=np.float32)
                    mask[:actual_len] = 1.0

                    batch_features.append(features)
                    batch_masks.append(mask)
                    batch_bands.append(band)
                    batch_track_info.append({'id': track_id, 'band': band})

                if len(batch_features) == 0:
                    continue

                # Batch inference
                features_tensor = torch.from_numpy(np.array(batch_features)).to(device)
                masks_tensor = torch.from_numpy(np.array(batch_masks)).to(device)
                bands_tensor = torch.tensor([[b] for b in batch_bands], dtype=torch.long).to(device)

                # Classify batch
                logits = model(features_tensor, masks_tensor, bands_tensor)
                probs = torch.softmax(logits, dim=-1)  # [batch, window_size, n_stems]

                # Average across frames for each track
                for i, track_info in enumerate(batch_track_info):
                    track_id = track_info['id']
                    band = track_info['band']

                    # Average probabilities across valid frames
                    track_probs = probs[i].cpu().numpy()  # [window_size, n_stems]
                    mask = batch_masks[i]

                    # Weighted average by mask
                    stem_probs = (track_probs * mask[:, None]).sum(axis=0) / mask.sum()

                    predicted_stem = int(stem_probs.argmax())
                    confidence = float(stem_probs[predicted_stem])

                    track_predictions[track_id] = {
                        'stem': predicted_stem,
                        'confidence': confidence,
                        'band': band
                    }

    return track_predictions


def load_and_reconstruct_tracks(h5_file, track_ids):
    """
    Load tracks from HDF5 and reconstruct them for synthesis.

    Returns list of track dicts with freq_times, amp_times, etc.
    """
    reconstructed_tracks = []

    # Read global metadata
    sample_rate = int(h5_file.attrs['sr'])
    fft_size = int(h5_file.attrs['fft'])

    # Calculate band sample rate
    bandwidth = fft_size * (6000.0 / 1024.0)
    band_sr = int(bandwidth * 2)

    # Build track index for compact format
    track_data_map = {}  # maps track_id -> (group_name, local_idx)

    for group_name in h5_file.keys():
        grp = h5_file[group_name]
        if 'id' not in grp:
            continue

        track_ids_in_group = grp['id'][:]
        for local_idx, tid in enumerate(track_ids_in_group):
            track_data_map[int(tid)] = (group_name, local_idx)

    # Load requested tracks
    for track_id in track_ids:
        if track_id not in track_data_map:
            continue

        group_name, local_idx = track_data_map[track_id]
        grp = h5_file[group_name]

        # Read track metadata
        track_lens = grp['len'][:]
        track_hops = grp['h'][:]

        # Calculate offset
        offset = sum(track_lens[:local_idx])
        n = int(track_lens[local_idx])

        # Extract track data
        frequencies = grp['f'][offset:offset+n].tolist()
        phases = grp['p'][offset:offset+n].tolist()
        amplitudes = grp['a'][offset:offset+n].tolist()
        indices = grp['i'][offset:offset+n].tolist()

        # Reconstruct times from indices and hop size
        hop_size = int(track_hops[local_idx])
        times = (np.array(indices) * hop_size / band_sr).tolist()

        track = {
            'frequencies': frequencies,
            'phases': phases,
            'amplitudes': amplitudes,
            'freq_times': times,
            'amp_times': times,
        }
        reconstructed_tracks.append(track)

    return reconstructed_tracks, sample_rate


def synthesize_tracks_gpu(tracks, n_samples, sample_rate, device='cuda'):
    """GPU-accelerated sinusoidal synthesis"""
    import torch
    from scipy.interpolate import interp1d

    synthesized = torch.zeros(n_samples, dtype=torch.float32, device=device)

    for track in tqdm(tracks, desc="    Synthesizing"):
        freq_times = np.array(track['freq_times'])
        frequencies = np.array(track['frequencies'])
        amp_times = np.array(track['amp_times'])
        amplitudes = np.array(track['amplitudes'])
        phases = np.array(track['phases'])

        if len(freq_times) < 1 or len(amplitudes) < 1:
            continue

        # Use exact birth/death times
        birth_time = freq_times[0]
        death_time = freq_times[-1]

        birth_sample = int(birth_time * sample_rate)
        death_sample = int(death_time * sample_rate)

        birth_sample = max(0, birth_sample)
        death_sample = min(n_samples - 1, death_sample)

        if birth_sample >= death_sample:
            continue

        n_sinusoid_samples = death_sample - birth_sample + 1
        t = np.arange(n_sinusoid_samples) / sample_rate + birth_time

        # Interpolate frequency
        if len(freq_times) == 1:
            freq_values = np.full(n_sinusoid_samples, frequencies[0])
        else:
            freq_interp = interp1d(freq_times, frequencies, kind='nearest',
                                bounds_error=False, fill_value=(frequencies[0], frequencies[-1]))
            freq_values = freq_interp(t)

        # Interpolate amplitude
        if len(amp_times) == 1:
            amp_values = np.full(n_sinusoid_samples, amplitudes[0])
        else:
            amp_values = np.interp(t, amp_times, amplitudes, left=0, right=0)
            amp_values = np.maximum(amp_values, 0)

        # Transfer to GPU
        freq_values_gpu = torch.from_numpy(freq_values).float().to(device)
        amp_values_gpu = torch.from_numpy(amp_values).float().to(device)

        # Phase integration on GPU
        initial_phase = phases[0]
        dt = 1.0 / sample_rate
        phase = initial_phase + 2 * np.pi * torch.cumsum(freq_values_gpu * dt, dim=0)

        # Synthesize on GPU
        sinusoid = amp_values_gpu * torch.sin(phase)

        # Add to output
        synthesized[birth_sample:death_sample+1] += sinusoid

    return synthesized.cpu().numpy()


def synthesize_stems(tracks_h5_path, track_predictions, output_dir, use_gpu=True):
    """
    Synthesize separate audio files for each stem.

    Groups tracks by predicted stem and synthesizes audio.
    """
    import soundfile as sf

    print(f"\nSynthesizing stems to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

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
        # Get audio length from metadata or estimate
        sample_rate = int(f.attrs['sr'])

        # Estimate n_samples from track data
        max_end_time = 0
        for group_name in f.keys():
            if 'i' in f[group_name] and 'h' in f[group_name]:
                grp = f[group_name]
                indices = grp['i'][:]
                hops = grp['h'][:]
                if len(indices) > 0 and len(hops) > 0:
                    fft_size = int(f.attrs['fft'])
                    bandwidth = fft_size * (6000.0 / 1024.0)
                    band_sr = int(bandwidth * 2)
                    max_idx = np.max(indices)
                    max_hop = np.max(hops)
                    end_time = max_idx * max_hop / band_sr
                    max_end_time = max(max_end_time, end_time)

        n_samples = int(max_end_time * sample_rate) + sample_rate  # Add 1 second buffer
        print(f"\nAudio length: {n_samples / sample_rate:.2f} seconds ({sample_rate} Hz)")

        for stem_idx, track_ids in stem_tracks.items():
            if len(track_ids) == 0:
                continue

            stem_name = STEM_NAMES[stem_idx] if stem_idx < 5 else 'unknown'
            print(f"\n  {stem_name}: {len(track_ids)} tracks")

            # Load and reconstruct tracks
            tracks, sr = load_and_reconstruct_tracks(f, track_ids)

            if len(tracks) == 0:
                print(f"    No tracks to synthesize")
                continue

            # Synthesize
            if use_gpu and torch.cuda.is_available():
                audio = synthesize_tracks_gpu(tracks, n_samples, sr, device=device)
            else:
                # CPU fallback (implement if needed)
                print(f"    CPU synthesis not implemented, skipping...")
                continue

            # Normalize
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val * 0.95  # Prevent clipping

            # Save
            output_file = output_path / f'{stem_name}.wav'
            sf.write(output_file, audio, sr)
            print(f"    ✓ Saved to {output_file}")

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
