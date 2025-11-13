import numpy as np
import soundfile as sf
import h5py
from scipy.signal import get_window, find_peaks, savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

class HybridResolutionSinusoidalExtractor:

    def __init__(self, sample_rate=192000):
        self.sample_rate = sample_rate

        # Configuration for frequency tracking (large FFT)
        self.freq_fft_size = 1024
        self.freq_hop_size = self.freq_fft_size
        self.freq_window = get_window('blackmanharris', self.freq_fft_size)
        self.freq_scale = 2.0 / np.sum(self.freq_window)

        # Tracking parameters
        self.max_peaks = 9999999

        print("Hybrid Resolution Sinusoidal Extractor")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Frequency tracking: FFT={self.freq_fft_size}, Hop={self.freq_hop_size}")

    def analyze(self, audio_file, use_synchrosqueeze=True):
        """Analyze audio"""
        print(f"\n📊 Analyzing: {audio_file}")

        audio, sr = sf.read(audio_file)

        # Check if stereo or mono
        if len(audio.shape) > 1:
            is_stereo = True
            n_channels = audio.shape[1]
            print(f"   Stereo audio detected ({n_channels} channels)")
        else:
            is_stereo = False
            n_channels = 1
            print(f"   Mono audio detected")
            audio = audio.reshape(-1, 1)

        duration = len(audio) / self.sample_rate
        print(f"   Duration: {duration:.2f} seconds")

        all_channel_tracks = []

        for ch in range(n_channels):
            print(f"\n   Processing channel {ch + 1}/{n_channels}...")
            audio_mono = audio[:, ch]

            tracks = self.extract_with_overlapping_bands(audio_mono)

            all_channel_tracks.append(tracks)

        return audio, all_channel_tracks, is_stereo

    def compute_stft(self, audio, fft_size, hop_size, window):
        """Compute STFT with given parameters"""
        n_frames = (len(audio) - fft_size) // hop_size + 1
        stft = np.zeros((fft_size // 2 + 1, n_frames), dtype=complex)

        for i in range(n_frames):
            start = i * hop_size
            if start + fft_size > len(audio):
                break
            frame = audio[start:start + fft_size] * window
            stft[:, i] = np.fft.rfft(frame)[:fft_size // 2 + 1]

        return stft

    def detect_transients(self, audio_mono, sensitivity=0.1):
        """Detect regions with sharp changes in audio"""

        print("   Detecting transients...")

        # Simple derivative (detects amplitude changes)
        diff = np.abs(np.diff(audio_mono))

        # Pad to match original length
        diff = np.pad(diff, (0, 1), mode='edge')

        # Smooth to avoid noise
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(diff, window_length=51, polyorder=3)

        # Threshold - anything above this is a "transient"
        threshold = np.percentile(smoothed, (1 - sensitivity) * 100)
        transient_mask = smoothed > threshold

        # Expand mask around transients (catch before/after)
        expansion = int(0.005 * self.sample_rate)  # 5ms buffer
        expanded_mask = np.zeros(len(audio_mono), dtype=bool)

        for i in np.where(transient_mask)[0]:
            start = max(0, i - expansion)
            end = min(len(expanded_mask), i + expansion)
            expanded_mask[start:end] = True

        print(f"   Transient regions: {np.sum(expanded_mask) / len(expanded_mask) * 100:.1f}% of audio")

        return expanded_mask

    def extract_with_overlapping_bands(self, audio_mono):
        """
        Extract sinusoidal tracks using multi-band analysis with exact decimation.
        """
        from scipy.signal import resample_poly
        from ssqueezepy import ssq_stft
        from scipy.optimize import linear_sum_assignment

        print("   Computing multi-band analysis...")

        nyquist = self.sample_rate / 2

        # Fixed number of bands: 16 for 192kHz (scales with sample rate)
        # This gives ~6kHz bandwidth per band for 192kHz
        n_bands = 16
        bandwidth = nyquist / n_bands

        bands = []
        for i in range(n_bands):
            low_freq = i * bandwidth
            high_freq = (i + 1) * bandwidth
            bands.append((low_freq, high_freq))

        print(f"   Created {len(bands)} non-overlapping bands ({bandwidth/1000:.1f} kHz width)")

        # Collect all unique edge frequencies
        edge_freqs = sorted(set([low for low, high in bands] + [high for low, high in bands]))
        print(f"   Pre-computing {len(edge_freqs)} downsampled versions for phase-coherent bands...")

        # Pre-compute downsampled versions at each edge frequency
        # This ensures phase coherence when we subtract to create bands
        from math import gcd
        downsampled_versions = {}

        for edge_freq in edge_freqs:
            if edge_freq == 0:
                downsampled_versions[edge_freq] = np.zeros_like(audio_mono)
            elif edge_freq >= nyquist:
                downsampled_versions[edge_freq] = audio_mono.copy()
            else:
                target_sample_rate = edge_freq * 2

                # Compute exact integer ratio using GCD to avoid rounding errors
                # This ensures precise cutoff frequencies without aliasing
                g = gcd(int(self.sample_rate), int(target_sample_rate))
                down = int(self.sample_rate) // g
                up = int(target_sample_rate) // g

                # Downsample to target rate (lowpass filter)
                downsampled = resample_poly(audio_mono, up, down)

                # Upsample back to original rate (preserves lowpass filtering)
                upsampled = resample_poly(downsampled, down, up)

                if len(upsampled) > len(audio_mono):
                    upsampled = upsampled[:len(audio_mono)]
                elif len(upsampled) < len(audio_mono):
                    upsampled = np.pad(upsampled, (0, len(audio_mono) - len(upsampled)))

                downsampled_versions[edge_freq] = upsampled

        # Create bands by subtracting pre-computed versions
        # This preserves phase relationships and avoids cancellation
        band_signals = []

        for low_freq, high_freq in bands:
            band_signal = downsampled_versions[high_freq] - downsampled_versions[low_freq]

            rms = np.sqrt(np.mean(band_signal**2))
            print(f"   Band {low_freq/1000:.1f}-{high_freq/1000:.1f} kHz: RMS={rms:.6f}")
            band_signals.append((low_freq, high_freq, band_signal))

        # Process each band independently with synchrosqueezed STFT
        print("   Processing bands with synchrosqueezed STFT...")

        all_band_tracks = []
        global_track_id = 0
        ssq_hop = 16

        for band_idx, (low_freq, high_freq, band_signal) in enumerate(tqdm(band_signals, desc="   Analyzing bands")):

            # Run synchrosqueezed STFT on this band
            Tx, Sx, ssq_freqs, Sfs = ssq_stft(
                band_signal,
                window='blackmanharris',
                n_fft=self.freq_fft_size,
                hop_len=ssq_hop,
                fs=self.sample_rate,
                modulated=True,
                dtype='float64'
            )

            tx_mag = np.abs(Tx)
            sx_mag = np.abs(Sx) * self.freq_scale
            sx_phase = np.angle(Sx)

            n_freq_bins, n_time_frames = Sx.shape

            # Extract peaks from each frame
            ssq_peaks = []

            for frame_idx in range(n_time_frames):
                frame_tx = tx_mag[:, frame_idx]
                frame_sx = sx_mag[:, frame_idx]
                frame_phase = sx_phase[:, frame_idx]

                peaks_idx, _ = find_peaks(frame_tx, distance=1)

                peaks = []

                for idx in peaks_idx[:self.max_peaks]:
                    if ssq_freqs.ndim == 1:
                        freq = ssq_freqs[idx]
                    else:
                        freq = ssq_freqs[idx, 0]

                    # Filter to band range
                    if freq < low_freq or freq > high_freq:
                        continue

                    if np.isnan(freq) or freq <= 0 or freq >= self.sample_rate / 2:
                        continue

                    amp = frame_sx[idx]
                    phase = frame_phase[idx]

                    peaks.append({
                        'frequency': freq,
                        'amplitude': amp,
                        'phase': phase,
                        'bin': idx
                    })

                peaks.sort(key=lambda x: x['amplitude'], reverse=True)
                ssq_peaks.append(peaks)

            # Track peaks over time using Hungarian algorithm
            active_tracks = []

            for frame_idx, frame_peaks in enumerate(ssq_peaks):
                if len(active_tracks) == 0:
                    for peak in frame_peaks[:self.max_peaks]:
                        new_track = {
                            'id': global_track_id,
                            'band': band_idx,
                            'start_frame': frame_idx,
                            'end_frame': frame_idx,
                            'frequencies': [peak['frequency']],
                            'amplitudes': [peak['amplitude']],
                            'phases': [peak['phase']],
                            'freq_frame_indices': [frame_idx]
                        }
                        active_tracks.append(new_track)
                        global_track_id += 1
                    continue

                if len(frame_peaks) == 0:
                    for track in active_tracks:
                        track['frequencies'].append(track['frequencies'][-1])
                        track['amplitudes'].append(0.0)
                        track['phases'].append(track['phases'][-1])
                        track['freq_frame_indices'].append(frame_idx)
                        track['end_frame'] = frame_idx
                    continue

                # Hungarian matching
                n_tracks = len(active_tracks)
                n_peaks = len(frame_peaks)

                LARGE_COST = 1e10
                cost_matrix = np.full((n_tracks, n_peaks), LARGE_COST)

                for i, track in enumerate(active_tracks):
                    freq_pred = track['frequencies'][-1]
                    for j, peak in enumerate(frame_peaks):
                        cost_matrix[i, j] = abs(peak['frequency'] - freq_pred)

                track_indices, peak_indices = linear_sum_assignment(cost_matrix)

                matched_peaks = set()
                matched_tracks = set()

                for track_idx, peak_idx in zip(track_indices, peak_indices):
                    if cost_matrix[track_idx, peak_idx] < LARGE_COST:
                        track = active_tracks[track_idx]
                        peak = frame_peaks[peak_idx]

                        track['frequencies'].append(peak['frequency'])
                        track['amplitudes'].append(peak['amplitude'])
                        track['phases'].append(peak['phase'])
                        track['freq_frame_indices'].append(frame_idx)
                        track['end_frame'] = frame_idx

                        matched_peaks.add(peak_idx)
                        matched_tracks.add(track_idx)

                for i, track in enumerate(active_tracks):
                    if i not in matched_tracks:
                        track['frequencies'].append(track['frequencies'][-1])
                        track['amplitudes'].append(0.0)
                        track['phases'].append(track['phases'][-1])
                        track['freq_frame_indices'].append(frame_idx)
                        track['end_frame'] = frame_idx

                for j, peak in enumerate(frame_peaks):
                    if j not in matched_peaks and len(active_tracks) < self.max_peaks:
                        new_track = {
                            'id': global_track_id,
                            'start_frame': frame_idx,
                            'end_frame': frame_idx,
                            'band': band_idx,
                            'frequencies': [peak['frequency']],
                            'amplitudes': [peak['amplitude']],
                            'phases': [peak['phase']],
                            'freq_frame_indices': [frame_idx]
                        }
                        active_tracks.append(new_track)
                        global_track_id += 1

            all_band_tracks.extend(active_tracks)

        print(f"   Total tracks: {len(all_band_tracks)}")

        # Post-process
        processed_tracks = []
        for track in all_band_tracks:
            freq_times = np.array(track['freq_frame_indices']) * ssq_hop / self.sample_rate

            processed_track = {
                'id': track['id'],
                'band': track['band'],
                'start_frame': track['start_frame'],
                'end_frame': track['end_frame'],
                'frequencies': track['frequencies'],
                'phases': track['phases'],
                'freq_frame_indices': track['freq_frame_indices'],
                'freq_times': freq_times.tolist(),
                'amplitudes': track['amplitudes'],
                'amp_times': freq_times.tolist(),
                'hop_size': ssq_hop
            }
            processed_tracks.append(processed_track)

        print(f"   ✓ Created {len(processed_tracks)} tracks (multi-band + synchrosqueeze)")

        return processed_tracks

    def save_to_hdf5(self, filename, all_channel_tracks, is_stereo, audio_file=None):
        """Save extracted tracks to HDF5 with progress bar"""
        print(f"\n💾 Saving to HDF5: {filename}")

        total_tracks = sum(len(tracks) for tracks in all_channel_tracks)

        with h5py.File(filename, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['sample_rate'] = self.sample_rate
            meta.attrs['freq_fft_size'] = self.freq_fft_size
            meta.attrs['freq_hop_size'] = self.freq_hop_size
            meta.attrs['is_stereo'] = is_stereo
            meta.attrs['n_channels'] = len(all_channel_tracks)
            if audio_file:
                meta.attrs['source_file'] = audio_file

            pbar = tqdm(total=total_tracks, desc="   Saving tracks")
            for ch_idx, tracks in enumerate(all_channel_tracks):
                channel_group = f.create_group(f'channel_{ch_idx}')
                channel_group.attrs['n_tracks'] = len(tracks)

                for track_idx, track in enumerate(tracks):
                    track_group = channel_group.create_group(f'track_{track_idx}')

                    track_group.create_dataset('id', data=track['id'])
                    track_group.create_dataset('start_frame', data=track['start_frame'])
                    track_group.create_dataset('end_frame', data=track['end_frame'])
                    track_group.create_dataset('frequencies', data=track['frequencies'])
                    track_group.create_dataset('phases', data=track['phases'])
                    track_group.create_dataset('freq_frame_indices', data=track['freq_frame_indices'])
                    track_group.create_dataset('freq_times', data=track['freq_times'])
                    track_group.create_dataset('amplitudes', data=track['amplitudes'])
                    track_group.create_dataset('amp_times', data=track['amp_times'])
                    track_group.attrs['hop_size'] = track['hop_size']

                    pbar.update(1)

            pbar.close()

        print(f"   ✓ Saved {total_tracks} total tracks")

    def load_from_hdf5(self, filename):
        """Load tracks from HDF5 file"""
        print(f"\n📂 Loading from HDF5: {filename}")

        all_channel_tracks = []

        with h5py.File(filename, 'r') as f:
            meta = f['metadata']
            is_stereo = meta.attrs['is_stereo']
            n_channels = meta.attrs['n_channels']

            print(f"   Sample rate: {meta.attrs['sample_rate']} Hz")
            print(f"   Channels: {n_channels}")

            for ch_idx in range(n_channels):
                channel_group = f[f'channel_{ch_idx}']
                n_tracks = channel_group.attrs['n_tracks']

                tracks = []
                for track_idx in range(n_tracks):
                    track_group = channel_group[f'track_{track_idx}']

                    track = {
                        'id': int(track_group['id'][()]),
                        'start_frame': int(track_group['start_frame'][()]),
                        'end_frame': int(track_group['end_frame'][()]),
                        'frequencies': track_group['frequencies'][:].tolist(),
                        'phases': track_group['phases'][:].tolist(),
                        'freq_frame_indices': track_group['freq_frame_indices'][:].tolist(),
                        'freq_times': track_group['freq_times'][:].tolist(),
                        'amplitudes': track_group['amplitudes'][:].tolist(),
                        'amp_times': track_group['amp_times'][:].tolist(),
                        'hop_size': track_group.attrs['hop_size']
                    }
                    tracks.append(track)

                all_channel_tracks.append(tracks)
                print(f"   Channel {ch_idx}: {len(tracks)} tracks")

        return all_channel_tracks, is_stereo

    def synthesize_stereo(self, all_channel_tracks, n_samples, is_stereo):
        """Synthesize stereo or mono output"""
        n_channels = len(all_channel_tracks)

        if is_stereo and n_channels == 2:
            print(f"\n🎼 Synthesizing stereo...")
            left = self.synthesize_channel(all_channel_tracks[0], n_samples)
            right = self.synthesize_channel(all_channel_tracks[1], n_samples)
            synthesized = np.column_stack([left, right])
        elif n_channels == 1:
            print(f"\n🎼 Synthesizing mono...")
            synthesized = self.synthesize_channel(all_channel_tracks[0], n_samples)
        else:
            print(f"\n🎼 Synthesizing {n_channels} channels...")
            channels = [self.synthesize_channel(tracks, n_samples) for tracks in all_channel_tracks]
            synthesized = np.column_stack(channels)

        return synthesized

    def synthesize_channel(self, tracks, n_samples):
        print(f"   Synthesizing {len(tracks)} tracks...")

        synthesized = np.zeros(n_samples)

        for track in tqdm(tracks, desc="   Progress"):
            freq_times = np.array(track['freq_times'])
            frequencies = np.array(track['frequencies'])
            amp_times = np.array(track['amp_times'])
            amplitudes = np.array(track['amplitudes'])
            phases = np.array(track['phases'])

            if len(freq_times) < 1 or len(amplitudes) < 1:
                continue

            # Use exact birth/death times
            birth_time = freq_times[0]
            death_time = freq_times[-1]

            birth_sample = int(birth_time * self.sample_rate)
            death_sample = int(death_time * self.sample_rate)

            birth_sample = max(0, birth_sample)
            death_sample = min(n_samples - 1, death_sample)

            if birth_sample >= death_sample:
                continue

            n_sinusoid_samples = death_sample - birth_sample + 1
            t = np.arange(n_sinusoid_samples) / self.sample_rate + birth_time

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
            dt = 1.0 / self.sample_rate
            phase = initial_phase + 2 * np.pi * np.cumsum(freq_values * dt)

            # Synthesize (no fades!)
            sinusoid = amp_values * np.sin(phase)

            # Add to output
            synthesized[birth_sample:death_sample+1] += sinusoid

        print(f"   ✓ Complete")
        return synthesized

# Usage
if __name__ == "__main__":
    import os

    audio_file = "Reconstruct_short.wav"

    # Read sample rate from file
    print(f"📂 Reading: {audio_file}")
    _, sr_detected = sf.read(audio_file, frames=1)
    print(f"   Detected sample rate: {sr_detected} Hz")

    # Create extractor with correct sample rate
    extractor = HybridResolutionSinusoidalExtractor(sample_rate=sr_detected)

    original, all_channel_tracks, is_stereo = extractor.analyze(audio_file)

    # Save to HDF5
    h5_file = audio_file.replace('.wav', '_tracks_multiband.h5')
    extractor.save_to_hdf5(h5_file, all_channel_tracks, is_stereo, audio_file)

    # Synthesize
    n_samples = original.shape[0]
    synthesized = extractor.synthesize_stereo(all_channel_tracks, n_samples, is_stereo)

    # Metrics
    print("\n📊 Reconstruction Metrics:")

    if is_stereo and len(original.shape) > 1:
        original_mono = np.mean(original, axis=1)
        synth_mono = np.mean(synthesized, axis=1) if len(synthesized.shape) > 1 else synthesized
    else:
        original_mono = original.flatten()
        synth_mono = synthesized.flatten()

    original_rms = np.sqrt(np.mean(original_mono**2))
    synth_rms = np.sqrt(np.mean(synth_mono**2))
    print(f"   Original RMS: {original_rms:.6f}")
    print(f"   Synthesized RMS: {synth_rms:.6f}")
    print(f"   Amplitude ratio: {synth_rms/original_rms:.2f}x")

    residual = original_mono - synth_mono
    residual_power = np.mean(residual**2)
    signal_power = np.mean(original_mono**2)
    if residual_power > 0 and signal_power > 0:
        snr_db = 10 * np.log10(signal_power / residual_power)
        print(f"   SNR: {snr_db:.1f} dB")
        print(f"   Reconstruction: {(1 - residual_power/signal_power)*100:.1f}%")

    # Save
    max_val = np.max(np.abs(synthesized))
    if max_val > 0:
        synthesized = synthesized / max_val * 0.95

    output_file = audio_file.replace('.wav', '_synthesized_multiband.wav')
    sf.write(output_file, synthesized, extractor.sample_rate)
    print(f"\n✓ Saved synthesized audio to {output_file}")
    print(f"✓ Saved track data to {h5_file}")
