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

            tracks = self.extract_with_bands(audio_mono)

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

    def extract_with_bands(self, audio_mono):
        """Extract bands based on sample rate (6kHz bandwidth per band)"""
        from scipy.signal import resample_poly
        from math import gcd

        print("   Computing multi-band extraction...")

        nyquist = self.sample_rate / 2

        # Calculate number of bands based on sample rate
        # Each band has 6kHz bandwidth
        bandwidth = 6000  # Hz
        n_bands = int(nyquist / bandwidth)

        print(f"   Sample rate: {self.sample_rate} Hz")
        print(f"   Nyquist: {nyquist/1000:.1f} kHz")
        print(f"   Bandwidth per band: {bandwidth/1000:.1f} kHz")
        print(f"   Number of bands: {n_bands}")

        # Create band definitions
        bands = []
        for i in range(n_bands):
            low_freq = i * bandwidth
            high_freq = (i + 1) * bandwidth
            bands.append((low_freq, high_freq))

        print(f"   Created {len(bands)} non-overlapping bands")

        # Collect all unique edge frequencies
        edge_freqs = sorted(set([low for low, high in bands] + [high for low, high in bands]))
        print(f"   Pre-computing {len(edge_freqs)} downsampled versions...")

        # Pre-compute downsampled versions at each edge frequency
        downsampled_versions = {}

        for edge_freq in edge_freqs:
            if edge_freq == 0:
                downsampled_versions[edge_freq] = np.zeros_like(audio_mono)
            elif edge_freq >= nyquist:
                downsampled_versions[edge_freq] = audio_mono.copy()
            else:
                target_sample_rate = edge_freq * 2

                # Compute exact integer ratio using GCD
                g = gcd(int(self.sample_rate), int(target_sample_rate))
                down = int(self.sample_rate) // g
                up = int(target_sample_rate) // g

                # Downsample to target rate (lowpass filter)
                downsampled = resample_poly(audio_mono, up, down)

                # Upsample back to original rate
                upsampled = resample_poly(downsampled, down, up)

                if len(upsampled) > len(audio_mono):
                    upsampled = upsampled[:len(audio_mono)]
                elif len(upsampled) < len(audio_mono):
                    upsampled = np.pad(upsampled, (0, len(audio_mono) - len(upsampled)))

                downsampled_versions[edge_freq] = upsampled

        # Create bands by subtracting pre-computed versions and export as WAV
        print("   Creating and exporting bands...")

        import os
        band_dir = "bands"
        os.makedirs(band_dir, exist_ok=True)

        band_signals = []
        band_sample_rate = int(bandwidth * 2)  # 12kHz for 6kHz bandwidth

        for band_idx, (low_freq, high_freq) in enumerate(bands):
            band_signal = downsampled_versions[high_freq] - downsampled_versions[low_freq]

            rms_original = np.sqrt(np.mean(band_signal**2))

            # Step 2: Frequency-shift to 0-6kHz window (skip band 0, already at baseband)
            if low_freq == 0:
                # Band 0 already starts at 0 Hz, no shift needed
                band_baseband = band_signal
                rms_baseband = rms_original
            else:
                # Shift down by low_freq so band starts at 0 Hz
                from scipy.signal import hilbert

                # Create analytic signal (removes negative frequencies)
                analytic = hilbert(band_signal)

                t = np.arange(len(band_signal)) / self.sample_rate

                # Frequency shift the analytic signal
                shifted = analytic * np.exp(-1j * 2 * np.pi * low_freq * t)

                # Take real part (baseband signal at original sample rate)
                band_baseband = np.real(shifted)

                rms_baseband = np.sqrt(np.mean(band_baseband**2))

            # Step 3: Downsample to bandwidth sample rate
            g = gcd(int(self.sample_rate), band_sample_rate)
            up = band_sample_rate // g
            down = int(self.sample_rate) // g

            band_downsampled = resample_poly(band_baseband, up, down)
            rms_downsampled = np.sqrt(np.mean(band_downsampled**2))

            # Step 4: Reverse process - upsample and shift back to original frequency
            # Upsample back to original sample rate
            band_upsampled = resample_poly(band_downsampled, down, up)

            # Match length to original
            if len(band_upsampled) > len(band_signal):
                band_upsampled = band_upsampled[:len(band_signal)]
            elif len(band_upsampled) < len(band_signal):
                band_upsampled = np.pad(band_upsampled, (0, len(band_signal) - len(band_upsampled)))

            # Frequency-shift back up (skip band 0)
            if low_freq == 0:
                band_restored = band_upsampled
            else:
                # Create analytic signal for upshift
                analytic_up = hilbert(band_upsampled)

                t_up = np.arange(len(band_upsampled)) / self.sample_rate

                # Shift back up by low_freq using analytic signal
                shifted_up = analytic_up * np.exp(1j * 2 * np.pi * low_freq * t_up)
                band_restored = np.real(shifted_up)

            rms_restored = np.sqrt(np.mean(band_restored**2))

            print(f"   Band {band_idx}: {low_freq/1000:.1f}-{high_freq/1000:.1f} kHz, RMS={rms_original:.6f} -> {rms_downsampled:.6f} @ {band_sample_rate}Hz -> {rms_restored:.6f} (restored)")

            # Export original band at full sample rate
            band_filename = os.path.join(band_dir, f"band_{band_idx:02d}_{int(low_freq/1000):02d}-{int(high_freq/1000):02d}kHz_orig.wav")
            sf.write(band_filename, band_signal, self.sample_rate)

            # Export downsampled shifted band
            band_baseband_filename = os.path.join(band_dir, f"band_{band_idx:02d}_{int(low_freq/1000):02d}-{int(high_freq/1000):02d}kHz_shifted.wav")
            sf.write(band_baseband_filename, band_downsampled, band_sample_rate)

            # Export restored band (pitched back up at original sample rate)
            band_restored_filename = os.path.join(band_dir, f"band_{band_idx:02d}_{int(low_freq/1000):02d}-{int(high_freq/1000):02d}kHz_restored.wav")
            sf.write(band_restored_filename, band_restored, self.sample_rate)

            band_signals.append((low_freq, high_freq, band_signal, band_downsampled, band_sample_rate))

        print(f"   ✓ Created {len(band_signals)} bands")
        print(f"   ✓ Exported {len(band_signals)} band WAV files to {band_dir}/")

        # Step 2: Process each downsampled band with synchrosqueeze
        print("   Processing bands with synchrosqueeze STFT...")
        from ssqueezepy import ssq_stft

        all_tracks = []
        global_track_id = 0
        ssq_hop = 16

        for band_idx, (low_freq, high_freq, band_signal, band_downsampled, band_sr) in enumerate(tqdm(band_signals, desc="   Analyzing bands")):

            # Run synchrosqueezed STFT on downsampled band at band sample rate
            Tx, Sx, ssq_freqs, Sfs = ssq_stft(
                band_downsampled,
                window='blackmanharris',
                n_fft=self.freq_fft_size,
                hop_len=ssq_hop,
                fs=band_sr,  # Use band sample rate (12kHz)
                modulated=True,
                dtype='float64'
            )

            tx_mag = np.abs(Tx)
            sx_mag = np.abs(Sx) * self.freq_scale
            sx_phase = np.angle(Sx)

            n_freq_bins, n_time_frames = Sx.shape

            # Extract peaks from each frame
            ssq_peaks = []

            for frame_idx in tqdm(range(n_time_frames), desc=f"     Band {band_idx} peak detection", leave=False):
                frame_tx = tx_mag[:, frame_idx]
                frame_sx = sx_mag[:, frame_idx]
                frame_phase = sx_phase[:, frame_idx]

                peaks_idx, _ = find_peaks(frame_tx, distance=1)

                peaks = []

                for idx in peaks_idx[:self.max_peaks]:
                    if ssq_freqs.ndim == 1:
                        freq_baseband = ssq_freqs[idx]
                    else:
                        freq_baseband = ssq_freqs[idx, 0]

                    # Filter to baseband range (0 to bandwidth)
                    if freq_baseband < 0 or freq_baseband > bandwidth:
                        continue

                    if np.isnan(freq_baseband) or freq_baseband >= band_sr / 2:
                        continue

                    # Shift frequency back to original band range
                    freq_original = freq_baseband + low_freq

                    amp = frame_sx[idx]
                    phase = frame_phase[idx]

                    peaks.append({
                        'frequency': freq_original,  # Store in original frequency range
                        'amplitude': amp,
                        'phase': phase,
                        'bin': idx
                    })

                peaks.sort(key=lambda x: x['amplitude'], reverse=True)
                ssq_peaks.append(peaks)

            # Track peaks over time using Hungarian algorithm
            active_tracks = []

            for frame_idx, frame_peaks in enumerate(tqdm(ssq_peaks, desc=f"     Band {band_idx} tracking", leave=False)):
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

            all_tracks.extend(active_tracks)

        print(f"   Total tracks: {len(all_tracks)}")

        # Post-process tracks
        processed_tracks = []
        for track in all_tracks:
            # Time calculation uses ORIGINAL sample rate (not band sample rate)
            freq_times = np.array(track['freq_frame_indices']) * ssq_hop / band_sample_rate

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

        print(f"   ✓ Created {len(processed_tracks)} tracks")

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

    audio_file = "Oooo_1sec_left.wav"

    # Read sample rate from file
    print(f"📂 Reading: {audio_file}")
    _, sr_detected = sf.read(audio_file, frames=1)
    print(f"   Detected sample rate: {sr_detected} Hz")

    # Create extractor with correct sample rate
    extractor = HybridResolutionSinusoidalExtractor(sample_rate=sr_detected)

    original, all_channel_tracks, is_stereo = extractor.analyze(audio_file)

    print(f"\n✓ Band extraction complete")
