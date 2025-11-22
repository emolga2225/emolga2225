#!/usr/bin/env python3
"""
Convert sinusoidal tracks to 2D time-frequency grid.

Key insight: Each TRACK in the HDF5 file represents ONE frequency component
evolving over time. This is perfect for a 2D representation!

Proposed 2D structure:
- Dimension 1 (frequency): Frequency bins (e.g., 0-22050 Hz in N bins)
- Dimension 2 (time): Time frames
- Values: Amplitude at that (frequency, time) location

This converts variable-length sinusoid lists into fixed-size 2D grids
that can be processed by CNNs or transformers.
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def analyze_track_structure(h5_path):
    """Analyze how tracks are organized to understand the data"""
    print(f"\nAnalyzing track structure in {h5_path.name}...")

    with h5py.File(h5_path, 'r') as f:
        grp = f['c0']  # Channel 0

        track_lens = grp['len'][:]
        frequencies = grp['f'][:]
        amplitudes = grp['a'][:]
        frames = grp['i'][:]

        print(f"Total tracks: {len(track_lens)}")
        print(f"Total sinusoids: {len(frequencies)}")

        # Analyze first few tracks
        offset = 0
        for track_idx in range(min(5, len(track_lens))):
            track_len = track_lens[track_idx]
            track_freqs = frequencies[offset:offset + track_len]
            track_frames = frames[offset:offset + track_len]
            track_amps = amplitudes[offset:offset + track_len]

            freq_std = track_freqs.std()
            print(f"\nTrack {track_idx}: len={track_len}")
            print(f"  Freq range: {track_freqs.min():.1f} - {track_freqs.max():.1f} Hz (std={freq_std:.2f})")
            print(f"  Frame range: {track_frames.min()} - {track_frames.max()}")
            print(f"  Amp range: {track_amps.min():.6f} - {track_amps.max():.6f}")

            if freq_std < 10:  # Frequency is relatively stable
                print(f"  -> Track follows ~{track_freqs.mean():.1f} Hz over time")

            offset += track_len

def sinusoids_to_2d_grid(h5_path, n_freq_bins=512, n_time_frames=None,
                         freq_min=0, freq_max=22050):
    """
    Convert sinusoidal tracks to 2D time-frequency grid.

    Each track represents one frequency over time, so we bin them into
    a fixed-size 2D array suitable for neural networks.

    Args:
        h5_path: Path to HDF5 file with sinusoidal tracks
        n_freq_bins: Number of frequency bins (height of 2D grid)
        n_time_frames: Number of time frames (width of 2D grid), None = auto-detect
        freq_min: Minimum frequency in Hz
        freq_max: Maximum frequency in Hz

    Returns:
        grid: 2D array of shape (n_freq_bins, n_time_frames) with amplitudes
    """
    with h5py.File(h5_path, 'r') as f:
        grp = f['c0']

        # Auto-detect number of time frames if not specified
        if n_time_frames is None:
            max_frame = grp['i'][:].max()
            n_time_frames = max_frame + 1

        # Initialize 2D grid
        grid = np.zeros((n_freq_bins, n_time_frames), dtype=np.float32)

        # Load track data
        track_lens = grp['len'][:]
        frequencies = grp['f'][:]
        amplitudes = grp['a'][:]
        frames = grp['i'][:]

        # Frequency bin width
        freq_bin_width = (freq_max - freq_min) / n_freq_bins

        # Process each track
        offset = 0
        for track_len in track_lens:
            track_freqs = frequencies[offset:offset + track_len]
            track_amps = amplitudes[offset:offset + track_len]
            track_frames = frames[offset:offset + track_len]

            # For each sinusoid in this track
            for freq, amp, frame in zip(track_freqs, track_amps, track_frames):
                # Convert frequency to bin index
                freq_bin = int((freq - freq_min) / freq_bin_width)
                freq_bin = np.clip(freq_bin, 0, n_freq_bins - 1)

                # Convert frame to time index
                time_idx = int(frame)
                if time_idx >= n_time_frames:
                    continue

                # Add amplitude to grid (accumulate if multiple sinusoids in same bin)
                grid[freq_bin, time_idx] += amp

            offset += track_len

        return grid


def create_2d_chunks_from_h5(h5_path, output_dir, chunk_duration=4.0,
                              hop_length=512, sample_rate=44100,
                              n_freq_bins=512):
    """
    Convert HDF5 sinusoids to 2D grid chunks for training.

    This creates fixed-size 2D representations that keep ALL sinusoidal data
    but in a format that CNNs/transformers can handle.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    chunk_frames = int(chunk_duration * sample_rate / hop_length)

    # First determine total number of frames
    with h5py.File(h5_path, 'r') as f:
        max_frame = f['c0']['i'][:].max()

    n_chunks = (max_frame // chunk_frames) + 1

    print(f"Converting {h5_path.name} to 2D grids...")
    print(f"  Frequency bins: {n_freq_bins}")
    print(f"  Chunk frames: {chunk_frames}")
    print(f"  Total chunks: {n_chunks}")

    # Convert entire file to 2D grid
    full_grid = sinusoids_to_2d_grid(h5_path, n_freq_bins=n_freq_bins)

    print(f"  Full grid shape: {full_grid.shape}")
    print(f"  Grid memory: {full_grid.nbytes / 1e6:.1f} MB")

    # Split into chunks
    for chunk_idx in tqdm(range(n_chunks), desc="Creating chunks"):
        start_frame = chunk_idx * chunk_frames
        end_frame = min(start_frame + chunk_frames, full_grid.shape[1])

        # Extract chunk from grid
        chunk_grid = full_grid[:, start_frame:end_frame]

        # Pad if necessary
        if chunk_grid.shape[1] < chunk_frames:
            padding = np.zeros((n_freq_bins, chunk_frames - chunk_grid.shape[1]),
                             dtype=np.float32)
            chunk_grid = np.hstack([chunk_grid, padding])

        # Save chunk
        chunk_file = output_dir / f'chunk_{chunk_idx:04d}.npy'
        np.save(chunk_file, chunk_grid)

    print(f"  Saved {n_chunks} 2D grid chunks to {output_dir}")


def demo_2d_conversion():
    """Demonstrate the 2D conversion approach"""
    print("=" * 60)
    print("Sinusoids to 2D Grid Conversion")
    print("=" * 60)

    # Example with synthetic data
    print("\nExample with synthetic sinusoid tracks:")
    print("\nSuppose we have 3 tracks:")
    print("  Track 0: 440 Hz over frames 0-100 with varying amplitude")
    print("  Track 1: 880 Hz over frames 50-150")
    print("  Track 2: 1320 Hz over frames 0-80")

    print("\nTraditional representation (variable-length list):")
    print("  [freq, amp, frame] × N sinusoids")
    print("  Total: ~250 sinusoids")
    print("  Problem: Can't process with standard neural networks")

    print("\n2D Grid representation (fixed-size array):")
    print("  Shape: (n_freq_bins, n_time_frames)")
    print("  Example: (512 bins, 150 frames)")
    print("  Values: Amplitude at each (frequency, time) location")
    print("  Memory: 512 × 150 × 4 bytes = 300 KB")
    print("  Benefits:")
    print("    - Fixed size, works with CNNs/transformers")
    print("    - Keeps ALL sinusoidal information")
    print("    - Natural 2D structure for convolutions")
    print("    - Similar to spectrogram but built from sinusoids")

    print("\nFor full song (52M sinusoids):")
    print("  Traditional: 52M × 3 floats = 624 MB of variable-length data")
    print("  2D Grid: 512 freq bins × 40000 frames × 4 bytes = 80 MB fixed-size")
    print("  Reduction: 8× smaller AND fixed-size for neural networks!")


if __name__ == "__main__":
    demo_2d_conversion()

    # Try to find and analyze actual data
    data_dirs = ['ajfa', 'blackned', 'dyerseve']
    for data_dir in data_dirs:
        vocals_h5 = Path(data_dir) / 'vocals_tracks.h5'
        if vocals_h5.exists():
            analyze_track_structure(vocals_h5)

            # Create 2D grid chunks
            output_dir = Path(data_dir) / 'grid_chunks'
            create_2d_chunks_from_h5(vocals_h5, output_dir)
            break
    else:
        print("\n" + "=" * 60)
        print("No data files found. Run this script when data is available.")
        print("It will convert sinusoidal tracks to 2D grids.")
