#!/usr/bin/env python3
"""
Synthesize audio from HDF5 sinusoidal tracks.

Takes sparse HDF5 track format (from test_stem_separator.py output)
and synthesizes audio using sinusoidal resynthesis.
"""

import numpy as np
import h5py
import soundfile as sf
from pathlib import Path
import argparse
from tqdm import tqdm


def load_tracks_from_h5(h5_file):
    """
    Load tracks from HDF5 file.

    Returns:
        dict with metadata and tracks
    """
    print(f"Loading tracks from: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        # Read metadata
        sample_rate = int(f.attrs['sr'])
        fft_size = int(f.attrs.get('fft', 128))
        hop_size = int(f.attrs.get('hop', 128))

        # Read first channel (c0)
        grp = f['c0']

        track_lens = grp['len'][:]
        all_freqs = grp['f'][:]
        all_amps = grp['a'][:]
        all_phases = grp['p'][:]
        all_indices = grp['i'][:]

        # Unpack into tracks
        tracks = []
        offset = 0

        for track_len in track_lens:
            track = {
                'frequencies': all_freqs[offset:offset + track_len],
                'amplitudes': all_amps[offset:offset + track_len],
                'phases': all_phases[offset:offset + track_len],
                'frame_indices': all_indices[offset:offset + track_len]
            }
            tracks.append(track)
            offset += track_len

        print(f"  Loaded {len(tracks)} tracks")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Total sinusoids: {offset}")

        if len(tracks) > 0:
            max_frame = max(t['frame_indices'][-1] for t in tracks if len(t['frame_indices']) > 0)
            print(f"  Max frame: {max_frame}")
        else:
            max_frame = 0

        return {
            'tracks': tracks,
            'sample_rate': sample_rate,
            'fft_size': fft_size,
            'hop_size': hop_size,
            'max_frame': max_frame
        }


def synthesize_audio(tracks_data, hop_length=512):
    """
    Synthesize audio from sinusoidal tracks.

    Args:
        tracks_data: dict from load_tracks_from_h5()
        hop_length: samples per frame (512 for 48kHz)

    Returns:
        audio: synthesized audio array
    """
    tracks = tracks_data['tracks']
    sample_rate = tracks_data['sample_rate']
    max_frame = tracks_data['max_frame']

    # Calculate audio length
    n_frames = max_frame + 1
    audio_len = n_frames * hop_length + 1024  # Extra for final frame
    audio = np.zeros(audio_len, dtype=np.float32)

    print(f"\nSynthesizing audio ({n_frames} frames, {audio_len / sample_rate:.2f}s)...")

    # Process each track
    for track in tqdm(tracks, desc="Synthesizing"):
        if len(track['frame_indices']) == 0:
            continue

        freqs = track['frequencies']
        amps = track['amplitudes']
        phases = track['phases']
        frame_indices = track['frame_indices'].astype(int)

        # Synthesize each time point in this track
        for i in range(len(frame_indices)):
            frame_idx = frame_indices[i]
            freq = freqs[i]
            amp = amps[i]
            phase = phases[i]

            # Time for this frame
            start_sample = frame_idx * hop_length
            t = np.arange(hop_length) / sample_rate

            # Generate sine wave for this frame
            frame_audio = amp * np.sin(2 * np.pi * freq * t + phase)

            # Add to output (overlap-add)
            end_sample = start_sample + hop_length
            if end_sample <= len(audio):
                audio[start_sample:end_sample] += frame_audio
            else:
                # Handle edge case at end
                valid_len = len(audio) - start_sample
                if valid_len > 0:
                    audio[start_sample:] += frame_audio[:valid_len]

    # Normalize to prevent clipping
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.99  # Leave some headroom

    return audio


def main():
    parser = argparse.ArgumentParser(description="Synthesize audio from HDF5 tracks")
    parser.add_argument('--input', required=True, help='Input HDF5 file (e.g., vocals_tracks.h5)')
    parser.add_argument('--output', required=True, help='Output WAV file (e.g., vocals.wav)')
    parser.add_argument('--hop-length', type=int, default=512,
                       help='Hop length in samples (default: 512 for 48kHz)')
    parser.add_argument('--sample-rate', type=int, default=None,
                       help='Output sample rate (default: use HDF5 metadata)')
    args = parser.parse_args()

    # Load tracks
    tracks_data = load_tracks_from_h5(args.input)

    # Use provided sample rate or HDF5 metadata
    if args.sample_rate is not None:
        tracks_data['sample_rate'] = args.sample_rate
        print(f"Using provided sample rate: {args.sample_rate} Hz")

    # Synthesize
    audio = synthesize_audio(tracks_data, hop_length=args.hop_length)

    # Save
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving audio to: {output_file}")
    sf.write(output_file, audio, tracks_data['sample_rate'])

    duration = len(audio) / tracks_data['sample_rate']
    print(f"[OK] Saved {duration:.2f}s audio ({len(audio):,} samples)")


if __name__ == "__main__":
    main()
