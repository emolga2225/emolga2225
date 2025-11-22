#!/usr/bin/env python3
"""Inspect how sinusoids are organized in HDF5 files"""

import h5py
import numpy as np
from pathlib import Path

def inspect_h5_structure(h5_path):
    """Examine the sinusoidal data structure in detail"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {h5_path.name}")
    print(f"{'='*60}")

    with h5py.File(h5_path, 'r') as f:
        n_channels = f.attrs.get('ch', 1)
        print(f"Number of channels: {n_channels}")

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            print(f"\nChannel {ch_idx}:")
            print(f"  Available datasets: {list(grp.keys())}")

            if 'len' in grp:
                track_lens = grp['len'][:]
                print(f"  Number of tracks: {len(track_lens)}")
                print(f"  Track length range: {track_lens.min()} - {track_lens.max()}")
                print(f"  Total sinusoids: {track_lens.sum()}")

            if 'f' in grp:
                frequencies = grp['f'][:]
                print(f"  Frequency array shape: {frequencies.shape}")
                print(f"  Frequency range: {frequencies.min():.2f} - {frequencies.max():.2f} Hz")
                print(f"  Unique frequencies: {len(np.unique(frequencies))}")

                # Check if frequencies are sorted or grouped
                freq_diffs = np.diff(frequencies)
                print(f"  Freq diffs range: {freq_diffs.min():.2f} - {freq_diffs.max():.2f}")

            if 'a' in grp:
                amplitudes = grp['a'][:]
                print(f"  Amplitude array shape: {amplitudes.shape}")
                print(f"  Amplitude range: {amplitudes.min():.6f} - {amplitudes.max():.6f}")

            if 'i' in grp:
                frames = grp['i'][:]
                print(f"  Frame array shape: {frames.shape}")
                print(f"  Frame range: {frames.min()} - {frames.max()}")
                print(f"  Unique frames: {len(np.unique(frames))}")

                # Check if frames are sorted
                if len(frames) > 1:
                    is_sorted = np.all(frames[:-1] <= frames[1:])
                    print(f"  Frames are sorted: {is_sorted}")

            # Sample first few tracks to understand organization
            if 'len' in grp and len(grp['len']) > 0:
                # Also check for phases
                has_phases = 'p' in grp
                if has_phases:
                    phases = grp['p'][:]
                    print(f"  Phase array shape: {phases.shape}")

                # Analyze first 5 tracks
                offset = 0
                for track_idx in range(min(5, len(track_lens))):
                    track_len = track_lens[track_idx]
                    track_freqs = frequencies[offset:offset + track_len]
                    track_frames = frames[offset:offset + track_len]
                    track_amps = amplitudes[offset:offset + track_len]

                    print(f"\n  Track {track_idx} (length={track_len}):")
                    print(f"    Frame range: {track_frames.min()} - {track_frames.max()}")
                    print(f"    Freq range: {track_freqs.min():.1f} - {track_freqs.max():.1f} Hz")
                    print(f"    Freq std dev: {track_freqs.std():.2f} Hz")
                    print(f"    Freq mean: {track_freqs.mean():.1f} Hz")
                    print(f"    Amp range: {track_amps.min():.6f} - {track_amps.max():.6f}")

                    # Check if frequency is stable or varying
                    if track_freqs.std() < 5.0:
                        print(f"    -> STABLE frequency track (~{track_freqs.mean():.1f} Hz)")
                    else:
                        print(f"    -> VARYING frequency track ({track_freqs.std():.1f} Hz variation)")

                    # Check frame ordering
                    is_sorted_by_frame = np.all(track_frames[:-1] <= track_frames[1:])
                    print(f"    Sorted by frame: {is_sorted_by_frame}")

                    offset += track_len

                    if track_idx == 0:
                        print(f"    First 10 freqs: {track_freqs[:10]}")
                        print(f"    First 10 frames: {track_frames[:10]}")

# Check vocals file
vocals_h5 = Path('ajfa/vocals_tracks.h5')
if vocals_h5.exists():
    inspect_h5_structure(vocals_h5)
else:
    print("vocals_tracks.h5 not found")

# Check fullmix too
fullmix_h5 = Path('ajfa/fullmix_tracks.h5')
if fullmix_h5.exists():
    inspect_h5_structure(fullmix_h5)
else:
    print("fullmix_tracks.h5 not found")
