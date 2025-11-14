import h5py
import numpy as np
from tqdm import tqdm
import json

class UltraFastFramewiseStemMatcher:
    """Ultra-fast frame-level stem matching using vectorization"""

    def __init__(self, freq_threshold=100.0, time_window=0.05, frame_sample_rate=4):
        """
        Args:
            freq_threshold: Maximum frequency difference in Hz for matching
            time_window: Time window in seconds for matching
            frame_sample_rate: Only match every Nth frame (default: 4)
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

    def build_stem_arrays(self, stem_tracks_list, stem_names):
        """Build vectorized arrays for all stem frames"""
        print("Building vectorized stem arrays...")

        all_stem_freqs = []
        all_stem_times = []
        all_stem_amps = []
        all_stem_labels = []

        for stem_tracks, stem_idx in tqdm(stem_tracks_list, desc="Vectorizing stems"):
            for track in stem_tracks:
                all_stem_freqs.extend(track['frequencies'])
                all_stem_times.extend(track['times'])
                all_stem_amps.extend(track['amplitudes'])
                all_stem_labels.extend([stem_idx] * len(track['frequencies']))

        stem_freqs = np.array(all_stem_freqs, dtype=np.float32)
        stem_times = np.array(all_stem_times, dtype=np.float32)
        stem_amps = np.array(all_stem_amps, dtype=np.float32)
        stem_labels = np.array(all_stem_labels, dtype=np.int32)

        print(f"  Vectorized {len(stem_freqs):,} stem frames")

        return stem_freqs, stem_times, stem_amps, stem_labels

    def match_track_vectorized(self, mix_track, stem_freqs, stem_times, stem_amps, stem_labels):
        """Match a single mix track using vectorized operations"""
        n_frames = mix_track['n_frames']
        labels = np.full(n_frames, -1, dtype=np.int32)

        # Sample frames (match every Nth frame, interpolate the rest)
        sample_indices = np.arange(0, n_frames, self.frame_sample_rate)

        for idx in sample_indices:
            mix_freq = mix_track['frequencies'][idx]
            mix_time = mix_track['times'][idx]
            mix_amp = mix_track['amplitudes'][idx]

            # Vectorized matching: find all stem frames within thresholds
            time_mask = np.abs(stem_times - mix_time) < self.time_window
            freq_mask = np.abs(stem_freqs - mix_freq) < self.freq_threshold
            valid_mask = time_mask & freq_mask

            if not np.any(valid_mask):
                continue

            # Get matching candidates
            candidate_freqs = stem_freqs[valid_mask]
            candidate_amps = stem_amps[valid_mask]
            candidate_labels = stem_labels[valid_mask]

            # Compute scores
            freq_diff = np.abs(candidate_freqs - mix_freq)
            freq_similarity = 1.0 - (freq_diff / self.freq_threshold)
            amp_similarity = np.minimum(mix_amp, candidate_amps) / (np.maximum(mix_amp, candidate_amps) + 1e-8)
            scores = freq_similarity * amp_similarity

            # Best match
            best_idx = np.argmax(scores)
            if scores[best_idx] > 0.1:
                labels[idx] = candidate_labels[best_idx]

        # Fill in non-sampled frames (forward fill)
        for i in range(n_frames):
            if labels[i] == -1 and i > 0:
                labels[i] = labels[i-1]

        return labels.tolist()

    def match_stems_framewise(self, mix_h5, stem_h5_files, stem_names, channel=0):
        """
        Match mix tracks to stems at the frame level (ultra-fast)

        Args:
            mix_h5: Path to mix HDF5 file
            stem_h5_files: List of (stem files or list of files) for each stem
            stem_names: List of stem names
            channel: Channel index to process

        Returns:
            Dictionary with frame-level labels for each mix track
        """
        print(f"Loading mix tracks from {mix_h5}...")
        mix_tracks = self.load_tracks_with_frames(mix_h5, channel)
        print(f"  Found {len(mix_tracks)} mix tracks")

        # Load all stem tracks
        stem_tracks_list = []
        for stem_idx, (stem_files, stem_name) in enumerate(zip(stem_h5_files, stem_names)):
            # Handle both single file and list of files
            if isinstance(stem_files, str):
                stem_files = [stem_files]

            # Load and combine tracks from all files for this stem
            combined_stem_tracks = []
            for stem_file in stem_files:
                print(f"Loading {stem_name} tracks from {stem_file}...")
                stem_tracks = self.load_tracks_with_frames(stem_file, channel)
                print(f"  Found {len(stem_tracks)} tracks")
                combined_stem_tracks.extend(stem_tracks)

            print(f"  Total {stem_name} tracks: {len(combined_stem_tracks)}")
            stem_tracks_list.append((combined_stem_tracks, stem_idx))

        # Build vectorized arrays
        stem_freqs, stem_times, stem_amps, stem_labels = self.build_stem_arrays(stem_tracks_list, stem_names)

        # Match each mix track using vectorization
        print(f"\nMatching frames to stems (ultra-fast, sampling every {self.frame_sample_rate} frames)...")
        frame_labels = {}

        for mix_track in tqdm(mix_tracks, desc="Processing mix tracks"):
            track_id = mix_track['id']

            # Vectorized matching
            labels = self.match_track_vectorized(
                mix_track, stem_freqs, stem_times, stem_amps, stem_labels
            )

            frame_labels[track_id] = {
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
    """Example usage"""
    # Configuration
    mix_h5 = "fullmix_tracks.h5"  # Complete mix
    stem_h5_files = [
        "vocals_tracks.h5",
        "guitar_tracks.h5",
        "bass_tracks.h5",
        "song_tracks.h5",  # Everything else (keyboards, rhythm guitar, etc.)
        # Drums: combine all 4 drum tracks into single "drums" stem
        ["drums_1_tracks.h5", "drums_2_tracks.h5", "drums_3_tracks.h5", "drums_4_tracks.h5"]
    ]
    stem_names = ["vocals", "guitar", "bass", "other", "drums"]

    print("Ultra-Fast Frame-wise Stem Matcher")
    print("=" * 60)

    # Create matcher (sample every 4th frame for speed)
    matcher = UltraFastFramewiseStemMatcher(
        freq_threshold=100.0,
        time_window=0.05,
        frame_sample_rate=4
    )

    # Match stems frame-wise
    frame_labels = matcher.match_stems_framewise(
        mix_h5=mix_h5,
        stem_h5_files=stem_h5_files,
        stem_names=stem_names,
        channel=0
    )

    # Save labels
    output_file = "frame_labels.json"

    # Convert numpy types to Python types for JSON serialization
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
