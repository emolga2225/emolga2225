#!/usr/bin/env python3
"""Test track-aware loading to debug amplitude issue"""

import h5py
import numpy as np
from pathlib import Path

def load_direct_no_tracks(h5_path):
    """Load WITHOUT respecting tracks (old broken way)"""
    sinusoids = []
    with h5py.File(h5_path, 'r') as f:
        n_channels = f.attrs.get('ch', 1)
        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue
            grp = f[grp_name]

            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            frame_indices = grp['i'][:]

            for freq, amp, frame in zip(frequencies, amplitudes, frame_indices):
                sinusoids.append([freq, amp, frame])

    return np.array(sinusoids, dtype=np.float32) if len(sinusoids) > 0 else np.zeros((0, 3), dtype=np.float32)

def load_track_aware(h5_path):
    """Load WITH track awareness (new way)"""
    sinusoids = []
    with h5py.File(h5_path, 'r') as f:
        n_channels = f.attrs.get('ch', 1)
        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue
            grp = f[grp_name]

            # Load track structure
            track_lens = grp['len'][:]

            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            frame_indices = grp['i'][:]

            # Process each track
            offset = 0
            for track_len in track_lens:
                track_freqs = frequencies[offset:offset + track_len]
                track_amps = amplitudes[offset:offset + track_len]
                track_frames = frame_indices[offset:offset + track_len]

                for freq, amp, frame in zip(track_freqs, track_amps, track_frames):
                    sinusoids.append([freq, amp, frame])

                offset += track_len

    return np.array(sinusoids, dtype=np.float32) if len(sinusoids) > 0 else np.zeros((0, 3), dtype=np.float32)

# Test on vocals
vocals_h5 = Path('ajfa/vocals_tracks.h5')
if vocals_h5.exists():
    print("Testing vocals_tracks.h5...")
    print("\n1. Direct loading (no tracks):")
    data_notracks = load_direct_no_tracks(vocals_h5)
    print(f"   Count: {len(data_notracks)}")
    print(f"   Amp range: {data_notracks[:, 1].min():.6f} - {data_notracks[:, 1].max():.6f}")
    print(f"   Non-zero: {(data_notracks[:, 1] > 0).sum()}")

    print("\n2. Track-aware loading:")
    data_tracks = load_track_aware(vocals_h5)
    print(f"   Count: {len(data_tracks)}")
    print(f"   Amp range: {data_tracks[:, 1].min():.6f} - {data_tracks[:, 1].max():.6f}")
    print(f"   Non-zero: {(data_tracks[:, 1] > 0).sum()}")

    print("\n3. First 10 sinusoids comparison:")
    print("   No tracks  : amp =", data_notracks[:10, 1])
    print("   Track-aware: amp =", data_tracks[:10, 1])
else:
    print("vocals_tracks.h5 not found")
