#!/usr/bin/env python3
"""
Test different SSQ extraction parameters via phase cancellation.

Measures reconstruction quality by:
1. Extracting sinusoids with different SSQ configs
2. Synthesizing them back
3. Phase canceling: original - reconstructed
4. Computing SNR to find optimal parameters

Lower residual = better sinusoidal representation
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

try:
    from ssqueezepy import ssq_cwt
    from ssqueezepy.wavelets import Wavelet
except ImportError:
    print("ERROR: ssqueezepy not installed")
    print("Install with: pip install ssqueezepy")
    exit(1)


def extract_sinusoids_ssq(audio, sr, fft_size=128, hop=16, gamma=3.0):
    """
    Extract sinusoids using Synchrosqueezed CWT.

    Returns tracks in the same format as ssq_sinusoidal_extractor.py
    """
    # Calculate bandwidth and band sample rate
    bandwidth = fft_size * (375.0 / 64.0)
    band_sr = int(bandwidth * 2)

    # Create wavelet
    wavelet = Wavelet(('morlet', {'mu': 5.0}))

    # Perform SSQ-CWT
    # This gives time-frequency representation
    Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(
        audio,
        wavelet,
        fs=band_sr,
        hop_len=hop,
        padtype='zero'
    )

    # Extract ridges (sinusoidal tracks)
    # Simple peak detection for now - find local maxima in each time frame
    tracks = []
    magnitude = np.abs(Tx)

    # For each time frame, find peaks
    for t_idx in range(magnitude.shape[1]):
        frame_mag = magnitude[:, t_idx]

        # Find peaks above threshold
        threshold = np.max(frame_mag) * 0.1  # 10% of max

        # Simple peak detection
        for f_idx in range(1, len(frame_mag) - 1):
            if (frame_mag[f_idx] > threshold and
                frame_mag[f_idx] > frame_mag[f_idx-1] and
                frame_mag[f_idx] > frame_mag[f_idx+1]):

                # Extract sinusoid parameters
                freq = ssq_freqs[f_idx]
                amp = frame_mag[f_idx]
                phase = np.angle(Tx[f_idx, t_idx])
                time = t_idx * hop / band_sr

                # Create a simple track (single point for now)
                track = {
                    'frequencies': [freq],
                    'amplitudes': [amp],
                    'phases': [phase],
                    'freq_times': [time],
                    'amp_times': [time],
                }
                tracks.append(track)

    return tracks, band_sr


def synthesize_tracks(tracks, n_samples, sample_rate):
    """
    Synthesize audio from sinusoidal tracks.

    Uses the same synthesis logic as synthesize_from_h5.py
    """
    synthesized = np.zeros(n_samples)

    for track in tracks:
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

        # Interpolate frequency (use first value for single-point tracks)
        if len(freq_times) == 1:
            freq_values = np.full(n_sinusoid_samples, frequencies[0])
        else:
            freq_values = np.interp(t, freq_times, frequencies)

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

    return synthesized


def compute_snr(signal, noise):
    """Compute SNR in dB: 10 * log10(signal_power / noise_power)"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def test_config(audio, sr, fft_size, hop, gamma=3.0, save_residual=None):
    """
    Test one SSQ configuration.

    Returns:
        - SNR (dB)
        - Number of tracks extracted
        - Residual audio (if save_residual is True)
    """
    print(f"\n  Testing FFT={fft_size}, Hop={hop}, Gamma={gamma}")

    # Extract sinusoids
    print(f"    Extracting sinusoids...")
    tracks, band_sr = extract_sinusoids_ssq(audio, sr, fft_size, hop, gamma)
    print(f"    Extracted {len(tracks)} tracks")

    # Synthesize
    print(f"    Synthesizing...")
    reconstructed = synthesize_tracks(tracks, len(audio), sr)

    # Phase cancel
    residual = audio - reconstructed

    # Compute SNR (original vs residual)
    snr = compute_snr(reconstructed, residual)

    print(f"    SNR: {snr:.2f} dB")
    print(f"    Tracks: {len(tracks)}")

    result = {
        'fft_size': fft_size,
        'hop': hop,
        'gamma': gamma,
        'snr_db': float(snr),
        'num_tracks': len(tracks),
        'reconstructed_rms': float(np.sqrt(np.mean(reconstructed ** 2))),
        'residual_rms': float(np.sqrt(np.mean(residual ** 2))),
    }

    if save_residual:
        result['residual'] = residual
        result['reconstructed'] = reconstructed

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Test SSQ parameters via phase cancellation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default configs
  python test_ssq_parameters.py input.wav

  # Test specific FFT sizes
  python test_ssq_parameters.py input.wav --fft 64 128 256

  # Test specific hop lengths
  python test_ssq_parameters.py input.wav --hop 8 16 32

  # Full grid search
  python test_ssq_parameters.py input.wav --fft 64 128 256 --hop 8 16 32

  # Save best result
  python test_ssq_parameters.py input.wav --save-best results/
        """
    )

    parser.add_argument('audio_file', help='Input audio file (WAV, MP3, etc.)')
    parser.add_argument('--fft', nargs='+', type=int, default=[64, 128, 256],
                        help='FFT sizes to test (default: 64 128 256)')
    parser.add_argument('--hop', nargs='+', type=int, default=[8, 16, 32],
                        help='Hop lengths to test (default: 8 16 32)')
    parser.add_argument('--gamma', type=float, default=3.0,
                        help='SSQ gamma parameter (default: 3.0)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Test duration in seconds (default: full file)')
    parser.add_argument('--save-best', type=str, default=None,
                        help='Save best config residual to directory')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Load audio
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {args.audio_file}")
        return

    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Trim to duration if specified
    if args.duration is not None:
        n_samples = int(args.duration * sr)
        audio = audio[:n_samples]

    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(audio) / sr:.2f}s")
    print(f"Samples: {len(audio):,}")

    # Test all configurations
    print(f"\n{'='*60}")
    print(f"Testing {len(args.fft)} FFT sizes × {len(args.hop)} hop lengths = {len(args.fft) * len(args.hop)} configs")
    print(f"{'='*60}")

    results = []
    best_result = None
    best_snr = -float('inf')

    for fft_size in args.fft:
        for hop in args.hop:
            result = test_config(
                audio, sr, fft_size, hop, args.gamma,
                save_residual=(args.save_best is not None)
            )
            results.append(result)

            # Track best
            if result['snr_db'] > best_snr:
                best_snr = result['snr_db']
                best_result = result

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'FFT':>6} {'Hop':>6} {'SNR (dB)':>12} {'Tracks':>10} {'Recon RMS':>12} {'Resid RMS':>12}")
    print(f"{'-'*60}")

    # Sort by SNR (best first)
    results.sort(key=lambda x: x['snr_db'], reverse=True)

    for i, r in enumerate(results):
        marker = " 🏆 BEST" if i == 0 else ""
        print(f"{r['fft_size']:>6} {r['hop']:>6} {r['snr_db']:>12.2f} {r['num_tracks']:>10} "
              f"{r['reconstructed_rms']:>12.6f} {r['residual_rms']:>12.6f}{marker}")

    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"FFT size: {best_result['fft_size']}")
    print(f"Hop length: {best_result['hop']}")
    print(f"Gamma: {best_result['gamma']}")
    print(f"SNR: {best_result['snr_db']:.2f} dB")
    print(f"Tracks: {best_result['num_tracks']}")

    # Interpret SNR
    print(f"\nInterpretation:")
    if best_result['snr_db'] > 0:
        print(f"  ✓ Positive SNR = reconstructed signal stronger than residual")
        print(f"  ✓ Sinusoidal representation captures most of the content")
    else:
        print(f"  ⚠ Negative SNR = residual stronger than reconstructed signal")
        print(f"  ⚠ Significant content lost in sinusoidal representation")

    # Save results
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'audio_file': str(audio_path),
            'sample_rate': int(sr),
            'duration': float(len(audio) / sr),
            'best_config': {
                'fft_size': int(best_result['fft_size']),
                'hop': int(best_result['hop']),
                'gamma': float(best_result['gamma']),
                'snr_db': float(best_result['snr_db']),
                'num_tracks': int(best_result['num_tracks']),
            },
            'all_results': [
                {k: v for k, v in r.items() if k not in ['residual', 'reconstructed']}
                for r in results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n[OK] Results saved to: {output_path}")

    # Save best residual
    if args.save_best and best_result:
        output_dir = Path(args.save_best)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = audio_path.stem

        # Save residual
        residual_path = output_dir / f"{base_name}_residual_fft{best_result['fft_size']}_hop{best_result['hop']}.wav"
        sf.write(residual_path, best_result['residual'], sr)
        print(f"[OK] Residual saved to: {residual_path}")

        # Save reconstructed
        recon_path = output_dir / f"{base_name}_reconstructed_fft{best_result['fft_size']}_hop{best_result['hop']}.wav"
        sf.write(recon_path, best_result['reconstructed'], sr)
        print(f"[OK] Reconstructed saved to: {recon_path}")


if __name__ == '__main__':
    main()
