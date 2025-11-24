#!/usr/bin/env python3
"""
Inspect HDF5 file to see what's stored and calculate correct parameters.
"""

import h5py
import numpy as np

def inspect_h5(filename):
    """Inspect HDF5 file and show metadata"""
    print(f"📊 Inspecting: {filename}\n")

    with h5py.File(filename, 'r') as f:
        print("=== METADATA (Global Attributes) ===")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        # Extract metadata
        if 'sr' in f.attrs:
            sr = int(f.attrs['sr'])
            print(f"\n  → Sample rate: {sr} Hz")

        if 'fft' in f.attrs:
            fft_size = int(f.attrs['fft'])
            bandwidth = fft_size * (6000.0 / 1024.0)
            band_sr = int(bandwidth * 2)
            print(f"  → FFT size: {fft_size}")
            print(f"  → Bandwidth per band: {bandwidth:.1f} Hz")
            print(f"  → Band sample rate: {band_sr} Hz")

        if 'hop' in f.attrs:
            hop = int(f.attrs['hop'])
            print(f"  → Hop size (metadata): {hop}")

        print("\n=== CHANNELS ===")
        channels = [k for k in f.keys() if k.startswith('c')]
        print(f"  Found {len(channels)} channel(s): {channels}")

        for ch in channels:
            print(f"\n=== CHANNEL: {ch} ===")
            grp = f[ch]

            print("  Datasets:")
            for key in grp.keys():
                dset = grp[key]
                print(f"    {key}: shape={dset.shape}, dtype={dset.dtype}")

            # Read track data
            if 'len' in grp and 's' in grp and 'h' in grp:
                track_lens = grp['len'][:]
                track_starts = grp['s'][:]
                track_hops = grp['h'][:]

                n_tracks = len(track_lens)
                print(f"\n  Tracks: {n_tracks}")
                print(f"  Total sinusoids: {np.sum(track_lens):,}")
                print(f"  Average sinusoids per track: {np.mean(track_lens):.1f}")

                # Analyze hop sizes
                unique_hops = np.unique(track_hops)
                print(f"  Unique hop sizes: {unique_hops}")

                # Find actual number of frames
                if 'i' in grp:
                    all_indices = grp['i'][:]
                    max_frame_idx = np.max(all_indices)
                    print(f"  Max frame index: {max_frame_idx}")
                    print(f"  Total frames: ~{max_frame_idx + 1}")

                # Calculate timing info
                if 'sr' in f.attrs and len(unique_hops) > 0:
                    hop_size = int(unique_hops[0])
                    if 'fft' in f.attrs:
                        fft_size = int(f.attrs['fft'])
                        bandwidth = fft_size * (6000.0 / 1024.0)
                        band_sr = int(bandwidth * 2)

                        frame_duration_ms = (hop_size / band_sr) * 1000
                        total_duration_s = (max_frame_idx * hop_size) / band_sr

                        print(f"\n  ⏱️  TIMING INFO:")
                        print(f"    Frame duration: {frame_duration_ms:.2f} ms")
                        print(f"    Total duration: {total_duration_s:.2f} seconds ({total_duration_s/60:.2f} minutes)")
                        print(f"    For audio reconstruction, use:")
                        print(f"      --hop-length {hop_size} (at {band_sr} Hz)")
                        print(f"    Or calculate from original sample rate:")
                        sr = int(f.attrs['sr'])
                        hop_at_original_sr = int((hop_size / band_sr) * sr)
                        print(f"      --hop-length {hop_at_original_sr} (at {sr} Hz)")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <file.h5>")
        sys.exit(1)

    inspect_h5(sys.argv[1])
