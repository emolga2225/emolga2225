#!/usr/bin/env python3
"""
Inference for 2D grid-based stem separator.

Converts fullmix HDF5 sinusoids -> 2D grid -> model prediction -> 2D grids -> back to sinusoids -> audio
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import argparse
from scipy.signal import resample
from scipy.io import wavfile


def sinusoids_to_grid(h5_path, n_freq_bins=512, chunk_frames=344,
                     freq_min=0, freq_max=22050):
    """Convert HDF5 sinusoids to 2D grid"""
    grid = np.zeros((n_freq_bins, chunk_frames), dtype=np.float32)
    freq_bin_width = (freq_max - freq_min) / n_freq_bins

    if not h5_path.exists():
        return grid

    with h5py.File(h5_path, 'r') as f:
        if 'c0' not in f:
            return grid

        grp = f['c0']
        track_lens = grp['len'][:]
        frequencies = grp['f'][:]
        amplitudes = grp['a'][:]
        frames = grp['i'][:]

        # Only take first chunk_frames
        offset = 0
        for track_len in track_lens:
            track_freqs = frequencies[offset:offset + track_len]
            track_amps = amplitudes[offset:offset + track_len]
            track_frames = frames[offset:offset + track_len]

            # Filter to chunk range
            mask = track_frames < chunk_frames
            chunk_freqs = track_freqs[mask]
            chunk_amps = track_amps[mask]
            chunk_frames_local = track_frames[mask]

            # Add to grid
            for freq, amp, frame in zip(chunk_freqs, chunk_amps, chunk_frames_local):
                freq_bin = int((freq - freq_min) / freq_bin_width)
                freq_bin = np.clip(freq_bin, 0, n_freq_bins - 1)
                time_idx = int(frame)

                if 0 <= time_idx < chunk_frames:
                    grid[freq_bin, time_idx] += amp

            offset += track_len

    return grid


def grid_to_sinusoids(grid, freq_min=0, freq_max=22050):
    """
    Convert 2D grid back to sinusoids.

    For each non-zero grid cell, create a sinusoid with:
    - freq: center frequency of that bin
    - amp: grid value
    - frame: time index
    """
    n_freq_bins, n_time_frames = grid.shape
    freq_bin_width = (freq_max - freq_min) / n_freq_bins

    sinusoids = []

    for freq_bin in range(n_freq_bins):
        for time_idx in range(n_time_frames):
            amp = grid[freq_bin, time_idx]

            # Only keep non-zero amplitudes
            if amp > 1e-10:
                freq = freq_min + (freq_bin + 0.5) * freq_bin_width
                sinusoids.append([freq, amp, time_idx])

    return np.array(sinusoids, dtype=np.float32) if sinusoids else np.zeros((0, 3), dtype=np.float32)


def synthesize_audio_from_sinusoids(sinusoids, hop_length=512, sample_rate=44100,
                                   duration=None):
    """
    Synthesize audio from sinusoids using additive synthesis.

    Args:
        sinusoids: Array of [freq, amp, frame] sinusoids
        hop_length: Hop length in samples
        sample_rate: Audio sample rate
        duration: Duration in samples (auto-detect if None)
    """
    if len(sinusoids) == 0:
        return np.zeros(44100, dtype=np.float32)  # 1 second of silence

    # Determine audio length
    if duration is None:
        max_frame = int(sinusoids[:, 2].max())
        duration = (max_frame + 1) * hop_length

    audio = np.zeros(duration, dtype=np.float32)

    # Group sinusoids by frame for efficiency
    for frame_idx in range(int(sinusoids[:, 2].max()) + 1):
        frame_mask = sinusoids[:, 2] == frame_idx
        frame_sines = sinusoids[frame_mask]

        if len(frame_sines) == 0:
            continue

        # Time range for this frame
        start_sample = frame_idx * hop_length
        end_sample = min(start_sample + hop_length, duration)
        n_samples = end_sample - start_sample

        # Time vector for this frame
        t = np.arange(n_samples) / sample_rate

        # Add all sinusoids in this frame
        for freq, amp, _ in frame_sines:
            # Generate sine wave
            sine_wave = amp * np.sin(2 * np.pi * freq * t)
            audio[start_sample:end_sample] += sine_wave

    # Normalize to prevent clipping
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    from train_stem_separator_2d_grid import UNetStemSeparator

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = UNetStemSeparator(
        n_stems=len(config['stem_names']),
        base_channels=config['base_channels']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def separate_stems(fullmix_h5, checkpoint_path, output_dir):
    """Separate stems from fullmix using trained 2D grid model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print("Loading model...")
    model, config = load_model(checkpoint_path, device)

    stem_names = config['stem_names']
    n_freq_bins = config['n_freq_bins']
    chunk_frames = config['chunk_frames']

    print(f"Model: {len(stem_names)} stems, {n_freq_bins} freq bins, {chunk_frames} time frames")
    print(f"Stems: {stem_names}")

    # Convert fullmix to 2D grid
    print("\nConverting fullmix to 2D grid...")
    fullmix_grid = sinusoids_to_grid(Path(fullmix_h5), n_freq_bins, chunk_frames)
    print(f"Fullmix grid shape: {fullmix_grid.shape}")
    print(f"Fullmix grid range: {fullmix_grid.min():.6f} - {fullmix_grid.max():.6f}")

    # Run model
    print("\nRunning model inference...")
    fullmix_tensor = torch.from_numpy(fullmix_grid[np.newaxis, np.newaxis, :, :]).to(device)

    with torch.no_grad():
        pred_stems = model(fullmix_tensor)

    pred_stems = pred_stems.cpu().numpy()[0]  # (n_stems, freq_bins, time_frames)
    print(f"Predicted stems shape: {pred_stems.shape}")

    # Convert each stem grid back to sinusoids and synthesize audio
    for stem_idx, stem_name in enumerate(stem_names):
        print(f"\nProcessing {stem_name}...")
        stem_grid = pred_stems[stem_idx]

        print(f"  Grid range: {stem_grid.min():.6f} - {stem_grid.max():.6f}")
        print(f"  Non-zero cells: {(stem_grid > 1e-10).sum()} / {stem_grid.size}")

        # Convert to sinusoids
        stem_sinusoids = grid_to_sinusoids(stem_grid)
        print(f"  Sinusoids: {len(stem_sinusoids)}")

        if len(stem_sinusoids) > 0:
            print(f"  Freq range: {stem_sinusoids[:, 0].min():.1f} - {stem_sinusoids[:, 0].max():.1f} Hz")
            print(f"  Amp range: {stem_sinusoids[:, 1].min():.6f} - {stem_sinusoids[:, 1].max():.6f}")

        # Synthesize audio
        audio = synthesize_audio_from_sinusoids(stem_sinusoids, hop_length=512, sample_rate=44100)
        print(f"  Audio shape: {audio.shape}, range: {audio.min():.3f} - {audio.max():.3f}")

        # Save audio
        output_file = output_dir / f'{stem_name}.wav'
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(output_file, 44100, audio_int16)
        print(f"  Saved: {output_file}")

    print(f"\nAll stems saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fullmix-h5', required=True, help='Path to fullmix HDF5 file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', default='output_stems', help='Output directory')
    args = parser.parse_args()

    separate_stems(args.fullmix_h5, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
