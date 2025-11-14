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
    Classify all tracks in an HDF5 file using sliding windows.

    With immortal tracks, each track can contain multiple stems at different
    times, so we classify frame-by-frame using sliding windows.

    Returns:
        frame_predictions: dict mapping track_idx -> list of frame-level predictions
    """
    print(f"\nClassifying tracks from {tracks_h5_path}...")

    window_size = config['window_size']
    window_stride = window_size // 2  # 50% overlap
    frame_predictions = {}  # track_id -> list of (frame_idx, stem, confidence)

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

        # Create all windows from all tracks
        all_windows = []
        for track in all_tracks:
            track_id = track['id']
            band = track['band']
            freqs = track['freqs']
            amps = track['amps']
            phases = track['phases']
            n_frames = len(freqs)

            # Skip very short tracks
            if n_frames < window_size:
                frame_predictions[track_id] = []
                continue

            # Create sliding windows
            for start_frame in range(0, n_frames - window_size + 1, window_stride):
                end_frame = start_frame + window_size

                window_freqs = freqs[start_frame:end_frame]
                window_amps = amps[start_frame:end_frame]
                window_phases = phases[start_frame:end_frame]

                all_windows.append({
                    'track_id': track_id,
                    'band': band,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'freqs': window_freqs,
                    'amps': window_amps,
                    'phases': window_phases
                })

        print(f"Created {len(all_windows):,} windows from {len(all_tracks)} tracks")

        # Classify windows in batches
        batch_size = 256
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(all_windows), batch_size), desc="Classifying"):
                batch_windows = all_windows[batch_start:batch_start + batch_size]

                batch_features = []
                batch_masks = []
                batch_bands = []

                for window in batch_windows:
                    # Create features
                    features = np.zeros((window_size, 4), dtype=np.float32)
                    actual_len = len(window['freqs'])
                    features[:actual_len, 0] = window['freqs'] / 96000.0
                    features[:actual_len, 1] = np.log1p(window['amps'])
                    features[:actual_len, 2] = np.cos(window['phases'])
                    features[:actual_len, 3] = np.sin(window['phases'])

                    mask = np.zeros(window_size, dtype=np.float32)
                    mask[:actual_len] = 1.0

                    batch_features.append(features)
                    batch_masks.append(mask)
                    batch_bands.append(window['band'])

                # Batch inference
                features_tensor = torch.from_numpy(np.array(batch_features)).to(device)
                masks_tensor = torch.from_numpy(np.array(batch_masks)).to(device)
                bands_tensor = torch.tensor([[b] for b in batch_bands], dtype=torch.long).to(device)

                # Classify batch
                logits = model(features_tensor, masks_tensor, bands_tensor)
                probs = torch.softmax(logits, dim=-1)  # [batch, window_size, n_stems]

                # Store frame-level predictions
                for i, window in enumerate(batch_windows):
                    track_id = window['track_id']
                    start_frame = window['start_frame']

                    if track_id not in frame_predictions:
                        frame_predictions[track_id] = {}

                    # Get predictions for each frame in window
                    window_probs = probs[i].cpu().numpy()  # [window_size, n_stems]

                    for frame_offset in range(window_size):
                        frame_idx = start_frame + frame_offset
                        predicted_stem = int(window_probs[frame_offset].argmax())
                        confidence = float(window_probs[frame_offset, predicted_stem])

                        # Average predictions if frame appears in multiple windows
                        if frame_idx in frame_predictions[track_id]:
                            # Average with existing prediction
                            old_stem, old_conf, count = frame_predictions[track_id][frame_idx]
                            new_count = count + 1
                            frame_predictions[track_id][frame_idx] = (predicted_stem, confidence, new_count)
                        else:
                            frame_predictions[track_id][frame_idx] = (predicted_stem, confidence, 1)

    return frame_predictions


def load_and_reconstruct_track_segments(h5_file, segments):
    """
    Load track segments from HDF5 and reconstruct them for synthesis.

    Args:
        segments: list of (track_id, start_frame, end_frame) tuples

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

    # Load requested track segments
    for track_id, start_frame, end_frame in segments:
        if track_id not in track_data_map:
            continue

        group_name, local_idx = track_data_map[track_id]
        grp = h5_file[group_name]

        # Read track metadata
        track_lens = grp['len'][:]
        track_hops = grp['h'][:]

        # Calculate offset for this track
        offset = sum(track_lens[:local_idx])
        n = int(track_lens[local_idx])

        # Extract only the specified segment
        seg_start = max(0, start_frame)
        seg_end = min(n - 1, end_frame)

        if seg_start >= seg_end:
            continue

        frequencies = grp['f'][offset + seg_start:offset + seg_end + 1].tolist()
        phases = grp['p'][offset + seg_start:offset + seg_end + 1].tolist()
        amplitudes = grp['a'][offset + seg_start:offset + seg_end + 1].tolist()
        indices = grp['i'][offset + seg_start:offset + seg_end + 1].tolist()

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


def synthesize_stems(tracks_h5_path, frame_predictions, output_dir, use_gpu=True):
    """
    Synthesize separate audio files for each stem using frame-level predictions.

    For immortal tracks, segments of tracks are assigned to different stems
    based on frame-level classifications.
    """
    import soundfile as sf

    print(f"\nSynthesizing stems to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Group track segments by stem
    # stem_segments[stem_idx] = [(track_id, start_frame, end_frame), ...]
    stem_segments = {i: [] for i in range(6)}  # 0-4: stems, 5: unknown

    # Process each track's frame predictions
    for track_id, frame_preds in frame_predictions.items():
        if len(frame_preds) == 0:
            continue

        # Convert to sorted list of (frame_idx, stem, confidence)
        frames = sorted([(idx, pred[0], pred[1]) for idx, pred in frame_preds.items()])

        # Group consecutive frames with same stem
        if len(frames) == 0:
            continue

        current_stem = frames[0][1]
        segment_start = frames[0][0]
        segment_end = frames[0][0]

        for frame_idx, stem, conf in frames[1:]:
            if stem == current_stem and frame_idx == segment_end + 1:
                # Continue current segment
                segment_end = frame_idx
            else:
                # Save current segment and start new one
                stem_segments[current_stem].append((track_id, segment_start, segment_end))
                current_stem = stem
                segment_start = frame_idx
                segment_end = frame_idx

        # Save final segment
        stem_segments[current_stem].append((track_id, segment_start, segment_end))

    # Print statistics
    print("\nPrediction statistics:")
    for stem_idx, segments in stem_segments.items():
        stem_name = STEM_NAMES[stem_idx] if stem_idx < 5 else 'unknown'
        unique_tracks = len(set(seg[0] for seg in segments))
        print(f"  {stem_name}: {len(segments)} segments from {unique_tracks} tracks")

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

        for stem_idx, segments in stem_segments.items():
            if len(segments) == 0:
                continue

            stem_name = STEM_NAMES[stem_idx] if stem_idx < 5 else 'unknown'
            unique_tracks = len(set(seg[0] for seg in segments))
            print(f"\n  {stem_name}: {len(segments)} segments from {unique_tracks} tracks")

            # Load and reconstruct track segments
            tracks, sr = load_and_reconstruct_track_segments(f, segments)

            if len(tracks) == 0:
                print(f"    No segments to synthesize")
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

    # Classify tracks (frame-level with immortal tracks support)
    frame_predictions = classify_tracks(model, args.input_h5, config, device=args.device)

    # Save predictions
    predictions_path = Path(args.output_dir) / 'frame_predictions.json'
    predictions_path.parent.mkdir(exist_ok=True, parents=True)

    predictions_serializable = {}
    for track_id, frame_preds in frame_predictions.items():
        predictions_serializable[str(track_id)] = {
            'frames': {
                str(frame_idx): {
                    'stem': int(pred[0]),
                    'stem_name': STEM_NAMES[pred[0]] if pred[0] < 5 else 'unknown',
                    'confidence': float(pred[1]),
                    'count': int(pred[2])  # How many overlapping windows predicted this frame
                }
                for frame_idx, pred in frame_preds.items()
            }
        }

    with open(predictions_path, 'w') as f:
        json.dump(predictions_serializable, f, indent=2)

    print(f"\nSaved frame-level predictions to {predictions_path}")

    # Synthesize stems
    synthesize_stems(args.input_h5, frame_predictions, args.output_dir)


if __name__ == '__main__':
    main()
