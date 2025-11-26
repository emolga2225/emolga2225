#!/usr/bin/env python3
"""
Synthesize audio from HDF5 using ssq_sinusoidal_extractor synthesis engine.

Uses the exact same synthesis that created the training data.
"""

import numpy as np
import h5py
import soundfile as sf
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy.interpolate import interp1d


def load_from_hdf5(filename):
    """Load tracks from HDF5 file (from ssq_sinusoidal_extractor)"""
    print(f"Loading from HDF5: {filename}")

    all_channel_tracks = []

    with h5py.File(filename, 'r') as f:
        # Read global metadata
        sample_rate = int(f.attrs['sr'])
        fft_size = int(f.attrs['fft'])
        is_stereo = bool(f.attrs.get('stereo', 0))
        n_channels = int(f.attrs.get('ch', 1))

        # Calculate bandwidth and band sample rate from FFT size
        bandwidth = fft_size * (375.0 / 64.0)  # Match extractor calculation
        band_sr = int(bandwidth * 2)

        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {n_channels}")
        print(f"  Band SR: {band_sr} Hz")

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                all_channel_tracks.append([])
                continue

            grp = f[grp_name]

            # Read packed arrays
            track_lens = grp['len'][:]
            track_ids = grp['id'][:]
            track_starts = grp['s'][:]
            track_ends = grp['e'][:]
            track_bands = grp['b'][:]
            track_hops = grp['h'][:]

            all_freqs = grp['f'][:]
            all_phases = grp['p'][:]
            all_amps = grp['a'][:]
            all_indices = grp['i'][:]

            # Unpack into individual tracks
            tracks = []
            offset = 0
            for i, n in enumerate(track_lens):
                frequencies = all_freqs[offset:offset+n].tolist()
                phases = all_phases[offset:offset+n].tolist()
                amplitudes = all_amps[offset:offset+n].tolist()
                indices = all_indices[offset:offset+n].tolist()

                # Reconstruct times from indices and hop size
                hop_size = int(track_hops[i])
                times = (np.array(indices) * hop_size / band_sr).tolist()

                track = {
                    'id': int(track_ids[i]),
                    'start_frame': int(track_starts[i]),
                    'end_frame': int(track_ends[i]),
                    'band': int(track_bands[i]),
                    'frequencies': frequencies,
                    'phases': phases,
                    'freq_frame_indices': indices,
                    'freq_times': times,
                    'amplitudes': amplitudes,
                    'amp_times': times,
                    'hop_size': hop_size
                }
                tracks.append(track)
                offset += n

            all_channel_tracks.append(tracks)
            print(f"  Channel {ch_idx}: {len(tracks)} tracks")

    return all_channel_tracks, is_stereo, sample_rate


def synthesize_channel(tracks, n_samples, sample_rate, use_gpu=False):
    """Synthesize one channel (CPU/GPU)"""
    print(f"  Synthesizing {len(tracks)} tracks...")

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        synthesized_gpu = torch.zeros(n_samples, dtype=torch.float32, device=device)

        for track in tqdm(tracks, desc="    Progress"):
            freq_times = np.array(track['freq_times'])
            frequencies = np.array(track['frequencies'])
            amp_times = np.array(track['amp_times'])
            amplitudes = np.array(track['amplitudes'])
            phases = np.array(track['phases'])

            if len(freq_times) < 1 or len(amplitudes) < 1:
                continue

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
            synthesized_gpu[birth_sample:death_sample+1] += sinusoid

        print(f"    [OK] Complete")
        return synthesized_gpu.cpu().numpy()

    else:
        # CPU synthesis
        synthesized = np.zeros(n_samples)

        for track in tqdm(tracks, desc="    Progress"):
            freq_times = np.array(track['freq_times'])
            frequencies = np.array(track['frequencies'])
            amp_times = np.array(track['amp_times'])
            amplitudes = np.array(track['amplitudes'])
            phases = np.array(track['phases'])

            if len(freq_times) < 1 or len(amplitudes) < 1:
                continue

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

            # Phase integration
            initial_phase = phases[0]
            dt = 1.0 / sample_rate
            phase = initial_phase + 2 * np.pi * np.cumsum(freq_values * dt)

            # Synthesize
            sinusoid = amp_values * np.sin(phase)

            # Add to output
            synthesized[birth_sample:death_sample+1] += sinusoid

        print(f"    [OK] Complete")
        return synthesized


def main():
    parser = argparse.ArgumentParser(description="Synthesize audio from HDF5 (ssq_sinusoidal_extractor engine)")
    parser.add_argument('--input', required=True, help='Input HDF5 file')
    parser.add_argument('--output', required=True, help='Output WAV file')
    parser.add_argument('--duration', type=float, default=None, help='Duration in seconds (default: auto from max frame)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    args = parser.parse_args()

    # Load tracks
    all_channel_tracks, is_stereo, sample_rate = load_from_hdf5(args.input)

    # Calculate duration
    if args.duration is not None:
        n_samples = int(args.duration * sample_rate)
    else:
        # Auto-detect from max frame time
        max_time = 0
        for tracks in all_channel_tracks:
            for track in tracks:
                if len(track['freq_times']) > 0:
                    max_time = max(max_time, track['freq_times'][-1])
        n_samples = int((max_time + 1.0) * sample_rate)  # Add 1s padding

    print(f"\nSynthesizing {n_samples / sample_rate:.2f}s audio...")

    # Synthesize
    n_channels = len(all_channel_tracks)
    if n_channels == 0:
        print("[ERROR] No channels found")
        return

    if is_stereo and n_channels == 2:
        print("Stereo mode")
        left = synthesize_channel(all_channel_tracks[0], n_samples, sample_rate, args.gpu)
        right = synthesize_channel(all_channel_tracks[1], n_samples, sample_rate, args.gpu)
        synthesized = np.column_stack([left, right])
    else:
        print("Mono mode")
        synthesized = synthesize_channel(all_channel_tracks[0], n_samples, sample_rate, args.gpu)

    # Normalize
    max_val = np.abs(synthesized).max()
    if max_val > 0:
        synthesized = synthesized / max_val * 0.99

    # Save
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_file}")
    sf.write(output_file, synthesized, sample_rate)

    duration = len(synthesized) / sample_rate
    print(f"[OK] Saved {duration:.2f}s audio")


if __name__ == "__main__":
    main()
