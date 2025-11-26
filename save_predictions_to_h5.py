#!/usr/bin/env python3
"""
Convert dense model predictions to sparse HDF5 track format.

Takes model output (n_frames, max_sinusoids, 3) and converts to
track-based HDF5 format matching ssq_sinusoidal_extractor.py output.
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


def dense_to_tracks(dense_sinusoids, amplitude_threshold=0.001):
    """
    Convert dense sinusoids to sparse tracks.

    A track is a sinusoid that persists over multiple frames.
    We track each sinusoid index across frames.

    Args:
        dense_sinusoids: (n_frames, max_sinusoids, 3) array [freq, amp, phase]
        amplitude_threshold: minimum amplitude to consider sinusoid active

    Returns:
        list of track dicts with:
            - frequencies: list of freqs for each frame
            - amplitudes: list of amps for each frame
            - phases: list of phases for each frame
            - freq_frame_indices: list of frame indices
            - start_frame: first frame
            - end_frame: last frame
            - hop_size: 16 (for 1500 Hz band rate)
    """
    n_frames, max_sinusoids, _ = dense_sinusoids.shape
    tracks = []
    track_id = 0

    # Track each sinusoid index across frames
    for sin_idx in tqdm(range(max_sinusoids), desc="Converting to tracks"):
        # Extract this sinusoid across all frames
        freqs = dense_sinusoids[:, sin_idx, 0]
        amps = dense_sinusoids[:, sin_idx, 1]
        phases = dense_sinusoids[:, sin_idx, 2]

        # Find active regions (where amplitude > threshold)
        active = amps > amplitude_threshold

        if not active.any():
            continue  # Skip if never active

        # Find contiguous active regions
        # Split into separate tracks when there are gaps
        in_track = False
        track_start = None
        track_freqs = []
        track_amps = []
        track_phases = []
        track_frames = []

        for frame_idx in range(n_frames):
            if active[frame_idx]:
                if not in_track:
                    # Start new track
                    in_track = True
                    track_start = frame_idx

                # Add to current track
                track_freqs.append(freqs[frame_idx])
                track_amps.append(amps[frame_idx])
                track_phases.append(phases[frame_idx])
                track_frames.append(frame_idx)

            else:
                if in_track:
                    # End current track
                    in_track = False

                    # Save track if it has points
                    if len(track_freqs) > 0:
                        tracks.append({
                            'id': track_id,
                            'frequencies': track_freqs,
                            'amplitudes': track_amps,
                            'phases': track_phases,
                            'freq_frame_indices': track_frames,
                            'start_frame': track_start,
                            'end_frame': track_frames[-1],
                            'hop_size': 16,
                            'band': 0
                        })
                        track_id += 1

                    # Reset for next track
                    track_freqs = []
                    track_amps = []
                    track_phases = []
                    track_frames = []

        # Handle track that extends to end
        if in_track and len(track_freqs) > 0:
            tracks.append({
                'id': track_id,
                'frequencies': track_freqs,
                'amplitudes': track_amps,
                'phases': track_phases,
                'freq_frame_indices': track_frames,
                'start_frame': track_start,
                'end_frame': track_frames[-1],
                'hop_size': 16,
                'band': 0
            })
            track_id += 1

    return tracks


def save_tracks_to_h5(tracks, output_file, sample_rate=48000, fft_size=128, hop_size=128):
    """
    Save tracks to HDF5 in ssq_sinusoidal_extractor format.

    Args:
        tracks: list of track dicts from dense_to_tracks()
        output_file: path to save HDF5
        sample_rate: original sample rate (48000 Hz)
        fft_size: FFT size used (128)
        hop_size: FFT hop (128)
    """
    print(f"Saving {len(tracks)} tracks to {output_file}")

    with h5py.File(output_file, 'w') as f:
        # Global metadata
        f.attrs['sr'] = np.int32(sample_rate)
        f.attrs['fft'] = np.int32(fft_size)
        f.attrs['hop'] = np.int32(hop_size)
        f.attrs['stereo'] = np.uint8(0)  # Mono
        f.attrs['ch'] = np.uint8(1)  # 1 channel

        # Pack all track data
        track_lens = []
        track_ids = []
        track_starts = []
        track_ends = []
        track_bands = []
        track_hops = []

        all_freqs = []
        all_phases = []
        all_amps = []
        all_indices = []

        for track in tqdm(tracks, desc="Packing tracks"):
            n = len(track['frequencies'])
            track_lens.append(n)
            track_ids.append(track['id'])
            track_starts.append(track['start_frame'])
            track_ends.append(track['end_frame'])
            track_bands.append(track.get('band', 0))
            track_hops.append(track['hop_size'])

            all_freqs.extend(track['frequencies'])
            all_phases.extend(track['phases'])
            all_amps.extend(track['amplitudes'])
            all_indices.extend(track['freq_frame_indices'])

        # Create channel group
        grp = f.create_group('c0')
        grp.create_dataset('len', data=np.array(track_lens, dtype=np.int32))
        grp.create_dataset('id', data=np.array(track_ids, dtype=np.int32))
        grp.create_dataset('s', data=np.array(track_starts, dtype=np.int32))
        grp.create_dataset('e', data=np.array(track_ends, dtype=np.int32))
        grp.create_dataset('b', data=np.array(track_bands, dtype=np.uint8))
        grp.create_dataset('h', data=np.array(track_hops, dtype=np.int16))

        grp.create_dataset('f', data=np.array(all_freqs, dtype=np.float32))
        grp.create_dataset('p', data=np.array(all_phases, dtype=np.float32))
        grp.create_dataset('a', data=np.array(all_amps, dtype=np.float32))
        grp.create_dataset('i', data=np.array(all_indices, dtype=np.int32))

    import os
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved {len(tracks)} tracks ({size_mb:.2f} MB)")


def convert_predictions_to_h5(pred_sinusoids, output_dir, stem_names):
    """
    Convert all stem predictions to HDF5 files.

    Args:
        pred_sinusoids: dict mapping stem_name -> (n_frames, max_sinusoids, 3) array
        output_dir: where to save HDF5 files
        stem_names: list of stem names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for stem_name in stem_names:
        print(f"\nProcessing {stem_name}...")
        dense = pred_sinusoids[stem_name]

        # Convert to tracks
        tracks = dense_to_tracks(dense, amplitude_threshold=0.001)

        # Save to HDF5
        output_file = output_dir / f'{stem_name}_tracks.h5'
        save_tracks_to_h5(tracks, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert dense predictions to HDF5")
    parser.add_argument('--input', required=True, help='Input .npz file with predictions')
    parser.add_argument('--output-dir', default='separated_h5', help='Output directory')
    parser.add_argument('--stems', nargs='+', default=['vocals', 'guitar', 'bass', 'drums', 'song'])
    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from {args.input}")
    data = np.load(args.input)

    pred_sinusoids = {}
    for stem_name in args.stems:
        if stem_name in data:
            pred_sinusoids[stem_name] = data[stem_name]

    # Convert
    convert_predictions_to_h5(pred_sinusoids, args.output_dir, args.stems)
