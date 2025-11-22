#!/usr/bin/env python3
"""
Inference for exact frequency stem separator (NO frequency binning).

Converts fullmix HDF5 -> exact freq array -> model prediction -> stem arrays -> audio
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import argparse
from scipy.io import wavfile


def sinusoids_to_exact_freq_array(h5_path, chunk_frames=344, max_sinusoids_per_frame=500):
    """
    Convert HDF5 sinusoids to exact frequency array - NO BINNING!

    Returns:
        array of shape (chunk_frames, max_sinusoids_per_frame, 3)
        where 3 = [exact_freq_Hz, amplitude, phase_radians]
    """
    array = np.zeros((chunk_frames, max_sinusoids_per_frame, 3), dtype=np.float32)

    if not h5_path.exists():
        return array

    # Dictionary: frame_idx -> list of (freq, amp, phase)
    frame_sinusoids = {i: [] for i in range(chunk_frames)}

    with h5py.File(h5_path, 'r') as f:
        if 'c0' not in f:
            return array

        grp = f['c0']
        track_lens = grp['len'][:]
        frequencies = grp['f'][:]
        amplitudes = grp['a'][:]
        phases = grp['p'][:] if 'p' in grp else np.zeros_like(amplitudes)
        frames = grp['i'][:]

        # Only take first chunk_frames
        offset = 0
        for track_len in track_lens:
            track_freqs = frequencies[offset:offset + track_len]
            track_amps = amplitudes[offset:offset + track_len]
            track_phases = phases[offset:offset + track_len]
            track_frames = frames[offset:offset + track_len]

            # Filter to chunk range
            mask = track_frames < chunk_frames
            chunk_freqs = track_freqs[mask]
            chunk_amps = track_amps[mask]
            chunk_phases = track_phases[mask]
            chunk_frames_local = track_frames[mask]

            # Add to frame dictionary with EXACT frequencies
            for freq, amp, phase, frame in zip(chunk_freqs, chunk_amps, chunk_phases, chunk_frames_local):
                frame_idx = int(frame)
                if 0 <= frame_idx < chunk_frames:
                    frame_sinusoids[frame_idx].append([freq, amp, phase])

            offset += track_len

    # Convert dictionary to fixed-size array
    for frame_idx in range(chunk_frames):
        sinusoids = frame_sinusoids[frame_idx]
        n_sinusoids = min(len(sinusoids), max_sinusoids_per_frame)

        if n_sinusoids > 0:
            # Sort by frequency for consistency
            sinusoids = sorted(sinusoids, key=lambda x: x[0])
            array[frame_idx, :n_sinusoids, :] = sinusoids[:n_sinusoids]

    return array


def exact_freq_array_to_sinusoids(array):
    """
    Convert exact frequency array back to list of sinusoids.

    Args:
        array: (n_frames, max_sinusoids, 3) where 3 = [freq, amp, phase]

    Returns:
        List of [freq, amp, phase, frame] sinusoids
    """
    n_frames, max_sinusoids, _ = array.shape
    sinusoids = []

    for frame_idx in range(n_frames):
        for sine_idx in range(max_sinusoids):
            freq = array[frame_idx, sine_idx, 0]
            amp = array[frame_idx, sine_idx, 1]
            phase = array[frame_idx, sine_idx, 2]

            # Only keep non-zero frequencies (zeros are padding)
            if freq > 0.1:  # Small threshold to avoid numerical errors
                sinusoids.append([freq, amp, phase, frame_idx])

    return np.array(sinusoids, dtype=np.float32) if sinusoids else np.zeros((0, 4), dtype=np.float32)


def synthesize_audio_from_sinusoids(sinusoids, hop_length=512, sample_rate=44100,
                                   duration=None):
    """
    Synthesize audio from sinusoids using additive synthesis with phase.

    Args:
        sinusoids: Array of [freq, amp, phase, frame] sinusoids
        hop_length: Hop length in samples
        sample_rate: Audio sample rate
        duration: Duration in samples (auto-detect if None)
    """
    if len(sinusoids) == 0:
        return np.zeros(44100, dtype=np.float32)  # 1 second of silence

    # Determine audio length
    if duration is None:
        max_frame = int(sinusoids[:, 3].max())
        duration = (max_frame + 1) * hop_length

    audio = np.zeros(duration, dtype=np.float32)

    # Group sinusoids by frame for efficiency
    for frame_idx in range(int(sinusoids[:, 3].max()) + 1):
        frame_mask = sinusoids[:, 3] == frame_idx
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
        for freq, amp, phase, _ in frame_sines:
            # Generate sine wave with phase
            sine_wave = amp * np.sin(2 * np.pi * freq * t + phase)
            audio[start_sample:end_sample] += sine_wave

    # Normalize to prevent clipping
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    from train_stem_separator_exact_freq import TransformerStemSeparator

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = TransformerStemSeparator(
        n_stems=len(config['stem_names']),
        max_sinusoids=config['max_sinusoids'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def separate_stems(fullmix_h5, checkpoint_path, output_dir):
    """Separate stems from fullmix using trained exact frequency model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print("Loading model...")
    model, config = load_model(checkpoint_path, device)

    stem_names = config['stem_names']
    chunk_frames = config['chunk_frames']
    max_sinusoids = config['max_sinusoids']

    print(f"Model: {len(stem_names)} stems, {max_sinusoids} max sinusoids/frame, {chunk_frames} time frames")
    print(f"Stems: {stem_names}")

    # Convert fullmix to exact frequency array
    print("\nConverting fullmix to exact frequency array (NO binning)...")
    fullmix_array = sinusoids_to_exact_freq_array(
        Path(fullmix_h5),
        chunk_frames=chunk_frames,
        max_sinusoids_per_frame=max_sinusoids
    )
    print(f"Fullmix array shape: {fullmix_array.shape}")  # (chunk_frames, max_sinusoids, 3)

    # Count non-zero sinusoids per frame
    non_zero_per_frame = []
    for frame_idx in range(chunk_frames):
        n_active = np.count_nonzero(fullmix_array[frame_idx, :, 0])
        non_zero_per_frame.append(n_active)

    print(f"Active sinusoids per frame: min={min(non_zero_per_frame)}, max={max(non_zero_per_frame)}, avg={np.mean(non_zero_per_frame):.1f}")

    # Show sample frequencies from first frame
    first_frame_freqs = fullmix_array[0, :, 0]
    active_freqs = first_frame_freqs[first_frame_freqs > 0][:10]
    if len(active_freqs) > 0:
        print(f"Sample frequencies from frame 0: {active_freqs.tolist()}")

    # Run model
    print("\nRunning model inference...")
    fullmix_tensor = torch.from_numpy(fullmix_array[np.newaxis, :, :, :]).to(device)  # (1, n_frames, max_sines, 3)

    with torch.no_grad():
        pred_stems = model(fullmix_tensor)

    pred_stems = pred_stems.cpu().numpy()[0]  # (n_stems, n_frames, max_sinusoids, 3)
    print(f"Predicted stems shape: {pred_stems.shape}")

    # Convert each stem array back to sinusoids and synthesize audio
    for stem_idx, stem_name in enumerate(stem_names):
        print(f"\nProcessing {stem_name}...")
        stem_array = pred_stems[stem_idx]  # (n_frames, max_sinusoids, 3)

        # Count active sinusoids
        active_per_frame = [np.count_nonzero(stem_array[f, :, 0]) for f in range(chunk_frames)]
        print(f"  Active sinusoids per frame: min={min(active_per_frame)}, max={max(active_per_frame)}, avg={np.mean(active_per_frame):.1f}")

        # Convert to sinusoids
        stem_sinusoids = exact_freq_array_to_sinusoids(stem_array)
        print(f"  Total sinusoids: {len(stem_sinusoids)}")

        if len(stem_sinusoids) > 0:
            print(f"  Freq range: {stem_sinusoids[:, 0].min():.1f} - {stem_sinusoids[:, 0].max():.1f} Hz")
            print(f"  Amp range: {stem_sinusoids[:, 1].min():.6f} - {stem_sinusoids[:, 1].max():.6f}")
            print(f"  Phase range: {stem_sinusoids[:, 2].min():.6f} - {stem_sinusoids[:, 2].max():.6f}")

            # Show sample exact frequencies
            sample_freqs = np.unique(stem_sinusoids[:, 0])[:10]
            print(f"  Sample exact frequencies: {sample_freqs.tolist()}")

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
    parser.add_argument('--output-dir', default='output_stems_exact_freq', help='Output directory')
    args = parser.parse_args()

    separate_stems(args.fullmix_h5, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
