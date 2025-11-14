import h5py
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict

class BandFilteredFramewiseMatcher:
    """Frame matcher using band filtering for massive speedup"""

    def __init__(self, freq_threshold=100.0, time_window=0.05, frame_sample_rate=8):
        """
        Args:
            freq_threshold: Maximum frequency difference in Hz for matching
            time_window: Time window in seconds for matching
            frame_sample_rate: Only match every Nth frame (default: 8)
        """
        self.freq_threshold = freq_threshold
        self.time_window = time_window
        self.frame_sample_rate = frame_sample_rate

    def load_tracks_with_frames(self, h5_file, channel=0):
        """Load all tracks with frame-level data"""
        tracks = []

        with h5py.File(h5_file, 'r') as f:
            grp = f[f'c{channel}']
            sample_rate = f.attrs['sr']
            fft_size = f.attrs['fft']

            bandwidth = fft_size * (6000.0 / 1024.0)
            band_sr = int(bandwidth * 2)

            track_lens = grp['len'][:]
            track_ids = grp['id'][:]
            track_bands = grp['b'][:]
            track_hops = grp['h'][:]

            all_freqs = grp['f'][:]
            all_amps = grp['a'][:]
            all_indices = grp['i'][:]

            offset = 0
            for i, n in enumerate(track_lens):
                freqs = all_freqs[offset:offset+n]
                amps = all_amps[offset:offset+n]
                indices = all_indices[offset:offset+n]

                hop_size = int(track_hops[i])
                times = indices * hop_size / band_sr

                tracks.append({
                    'id': int(track_ids[i]),
                    'band': int(track_bands[i]),
                    'frequencies': freqs,
                    'amplitudes': amps,
                    'times': times,
                    'n_frames': n
                })

                offset += n

        return tracks

    def build_band_filtered_index(self, stem_tracks_list, stem_names):
        """Build index organized by band, then by time bin"""
        print("Building band-filtered index...")

        # Index structure: band -> time_bin -> list of frames
        bin_size = 0.1  # 100ms bins (coarser than before)
        band_index = defaultdict(lambda: defaultdict(list))

        for stem_tracks, stem_idx in tqdm(stem_tracks_list, desc="Indexing stems"):
            for track in stem_tracks:
                band = track['band']

                for frame_idx in range(len(track['times'])):
                    time = track['times'][frame_idx]
                    freq = track['frequencies'][frame_idx]
                    amp = track['amplitudes'][frame_idx]

                    time_bin = int(time / bin_size)

                    band_index[band][time_bin].append({
                        'stem_idx': stem_idx,
                        'freq': freq,
                        'amp': amp,
                        'time': time
                    })

        # Convert lists to numpy arrays for faster access
        print("Converting to numpy arrays...")
        for band in tqdm(band_index.keys(), desc="Vectorizing bands"):
            for time_bin in band_index[band].keys():
                frames = band_index[band][time_bin]
                if len(frames) > 0:
                    band_index[band][time_bin] = {
                        'stem_idx': np.array([f['stem_idx'] for f in frames], dtype=np.int32),
                        'freq': np.array([f['freq'] for f in frames], dtype=np.float32),
                        'amp': np.array([f['amp'] for f in frames], dtype=np.float32),
                        'time': np.array([f['time'] for f in frames], dtype=np.float32)
                    }

        total_bins = sum(len(bins) for bins in band_index.values())
        total_frames = sum(
            len(band_index[band][bin_idx]['freq'])
            for band in band_index
            for bin_idx in band_index[band]
        )
        print(f"  Indexed {total_frames:,} frames across {len(band_index)} bands, {total_bins} time bins")

        return band_index, bin_size

    def match_track_fast(self, mix_track, band_index, bin_size):
        """Match a single track using band filtering"""
        n_frames = mix_track['n_frames']
        labels = np.full(n_frames, -1, dtype=np.int32)
        mix_band = mix_track['band']

        # Check if this band exists in index
        if mix_band not in band_index:
            return labels.tolist()

        # Sample frames
        sample_indices = np.arange(0, n_frames, self.frame_sample_rate)

        for idx in sample_indices:
            mix_freq = mix_track['frequencies'][idx]
            mix_time = mix_track['times'][idx]
            mix_amp = mix_track['amplitudes'][idx]

            # Get time bin and neighbors (only search within same band!)
            time_bin = int(mix_time / bin_size)

            best_stem = -1
            best_score = 0.0

            # Check this bin and neighbors
            for bin_offset in [-1, 0, 1]:
                check_bin = time_bin + bin_offset

                if check_bin not in band_index[mix_band]:
                    continue

                bin_data = band_index[mix_band][check_bin]

                # Vectorized matching within this bin
                time_diff = np.abs(bin_data['time'] - mix_time)
                time_mask = time_diff < self.time_window

                if not np.any(time_mask):
                    continue

                freq_diff = np.abs(bin_data['freq'][time_mask] - mix_freq)
                freq_mask = freq_diff < self.freq_threshold

                if not np.any(freq_mask):
                    continue

                # Compute scores for valid candidates
                valid_freqs = bin_data['freq'][time_mask][freq_mask]
                valid_amps = bin_data['amp'][time_mask][freq_mask]
                valid_stems = bin_data['stem_idx'][time_mask][freq_mask]
                valid_freq_diffs = freq_diff[freq_mask]

                freq_similarity = 1.0 - (valid_freq_diffs / self.freq_threshold)
                amp_similarity = np.minimum(mix_amp, valid_amps) / (np.maximum(mix_amp, valid_amps) + 1e-8)
                scores = freq_similarity * amp_similarity

                # Best match
                best_idx = np.argmax(scores)
                if scores[best_idx] > best_score:
                    best_score = scores[best_idx]
                    best_stem = valid_stems[best_idx]

            if best_score > 0.1:
                labels[idx] = best_stem

        # Forward fill
        for i in range(n_frames):
            if labels[i] == -1 and i > 0:
                labels[i] = labels[i-1]

        return labels.tolist()

    def match_stems_framewise(self, mix_h5, stem_h5_files, stem_names, channel=0):
        """Match mix tracks to stems (band-filtered)"""
        print(f"Loading mix tracks from {mix_h5}...")
        mix_tracks = self.load_tracks_with_frames(mix_h5, channel)
        print(f"  Found {len(mix_tracks)} mix tracks")

        # Load stem tracks
        stem_tracks_list = []
        for stem_idx, (stem_files, stem_name) in enumerate(zip(stem_h5_files, stem_names)):
            if isinstance(stem_files, str):
                stem_files = [stem_files]

            combined_stem_tracks = []
            for stem_file in stem_files:
                print(f"Loading {stem_name} tracks from {stem_file}...")
                stem_tracks = self.load_tracks_with_frames(stem_file, channel)
                print(f"  Found {len(stem_tracks)} tracks")
                combined_stem_tracks.extend(stem_tracks)

            print(f"  Total {stem_name} tracks: {len(combined_stem_tracks)}")
            stem_tracks_list.append((combined_stem_tracks, stem_idx))

        # Build band-filtered index
        band_index, bin_size = self.build_band_filtered_index(stem_tracks_list, stem_names)

        # Match tracks
        print(f"\nMatching frames (band-filtered, sampling every {self.frame_sample_rate} frames)...")
        frame_labels = {}

        for mix_track in tqdm(mix_tracks, desc="Processing mix tracks"):
            labels = self.match_track_fast(mix_track, band_index, bin_size)

            frame_labels[mix_track['id']] = {
                'labels': labels,
                'band': int(mix_track['band']),
                'n_frames': mix_track['n_frames']
            }

        # Statistics
        print("\nFrame-level matching statistics:")
        total_frames = sum(len(data['labels']) for data in frame_labels.values())

        stem_frame_counts = {name: 0 for name in stem_names}
        stem_frame_counts['unmatched'] = 0

        for data in frame_labels.values():
            for label in data['labels']:
                if label == -1:
                    stem_frame_counts['unmatched'] += 1
                else:
                    stem_frame_counts[stem_names[label]] += 1

        print(f"Total frames: {total_frames:,}")
        for stem_name, count in stem_frame_counts.items():
            pct = 100.0 * count / total_frames if total_frames > 0 else 0
            print(f"  {stem_name}: {count:,} ({pct:.1f}%)")

        return frame_labels


def main():
    mix_h5 = "fullmix_tracks.h5"
    stem_h5_files = [
        "vocals_tracks.h5",
        "guitar_tracks.h5",
        "bass_tracks.h5",
        "song_tracks.h5",
        ["drums_1_tracks.h5", "drums_2_tracks.h5", "drums_3_tracks.h5", "drums_4_tracks.h5"]
    ]
    stem_names = ["vocals", "guitar", "bass", "other", "drums"]

    print("Band-Filtered Frame Matcher")
    print("=" * 60)

    matcher = BandFilteredFramewiseMatcher(
        freq_threshold=100.0,
        time_window=0.05,
        frame_sample_rate=8  # Sample every 8th frame
    )

    frame_labels = matcher.match_stems_framewise(
        mix_h5=mix_h5,
        stem_h5_files=stem_h5_files,
        stem_names=stem_names,
        channel=0
    )

    output_file = "frame_labels.json"
    frame_labels_serializable = {}
    for track_id, data in frame_labels.items():
        frame_labels_serializable[str(track_id)] = {
            'labels': [int(x) for x in data['labels']],
            'band': int(data['band']),
            'n_frames': int(data['n_frames'])
        }

    with open(output_file, 'w') as f:
        json.dump(frame_labels_serializable, f, indent=2)

    print(f"\nSaved frame-level labels to {output_file}")


if __name__ == "__main__":
    main()
