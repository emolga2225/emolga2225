import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import json

class FramewiseStemMatcher:
    """Match sinusoidal tracks between mix and stems at the frame level"""

    def __init__(self, freq_threshold=100.0):
        """
        Args:
            freq_threshold: Maximum frequency difference in Hz for matching
        """
        self.freq_threshold = freq_threshold

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
                    'frame_indices': indices,
                    'n_frames': n
                })

                offset += n

        return tracks

    def match_frame_to_stems(self, mix_freq, mix_time, mix_amp, stem_tracks_list, stem_names):
        """
        Match a single frame from a mix track to stem tracks

        Args:
            mix_freq: Frequency at this frame
            mix_time: Time at this frame
            mix_amp: Amplitude at this frame
            stem_tracks_list: List of (stem_tracks, stem_idx) tuples
            stem_names: List of stem names

        Returns:
            Best matching stem index, or -1 if no match
        """
        best_stem = -1
        best_score = 0.0

        for stem_tracks, stem_idx in stem_tracks_list:
            for stem_track in stem_tracks:
                # Find frames in stem track that overlap in time
                time_matches = np.abs(stem_track['times'] - mix_time) < 0.05  # 50ms window

                if not np.any(time_matches):
                    continue

                # Check frequency similarity at matching times
                matching_freqs = stem_track['frequencies'][time_matches]
                matching_amps = stem_track['amplitudes'][time_matches]

                for stem_freq, stem_amp in zip(matching_freqs, matching_amps):
                    freq_diff = abs(mix_freq - stem_freq)

                    if freq_diff < self.freq_threshold:
                        # Score based on frequency similarity and amplitude
                        freq_similarity = 1.0 - (freq_diff / self.freq_threshold)
                        amp_similarity = min(mix_amp, stem_amp) / (max(mix_amp, stem_amp) + 1e-8)
                        score = freq_similarity * amp_similarity

                        if score > best_score:
                            best_score = score
                            best_stem = stem_idx

        # Only return match if score is high enough
        if best_score > 0.1:
            return best_stem
        return -1

    def match_stems_framewise(self, mix_h5, stem_h5_files, stem_names, channel=0):
        """
        Match mix tracks to stems at the frame level

        Args:
            mix_h5: Path to mix HDF5 file
            stem_h5_files: List of (stem files or list of files) for each stem
                          e.g., ['vocals.h5', 'guitar.h5', ['drums_1.h5', 'drums_2.h5']]
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

        # Match each frame of each mix track
        print("\nMatching frames to stems...")
        frame_labels = {}

        for mix_track in tqdm(mix_tracks, desc="Processing mix tracks"):
            track_id = mix_track['id']
            n_frames = mix_track['n_frames']

            # Store label for each frame
            labels = []

            for frame_idx in range(n_frames):
                mix_freq = mix_track['frequencies'][frame_idx]
                mix_time = mix_track['times'][frame_idx]
                mix_amp = mix_track['amplitudes'][frame_idx]

                # Match this frame to stems
                stem_label = self.match_frame_to_stems(
                    mix_freq, mix_time, mix_amp,
                    stem_tracks_list, stem_names
                )

                labels.append(int(stem_label))

            frame_labels[track_id] = {
                'labels': labels,
                'band': int(mix_track['band']),
                'n_frames': n_frames
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

        print(f"Total frames: {total_frames}")
        for stem_name, count in stem_frame_counts.items():
            pct = 100.0 * count / total_frames if total_frames > 0 else 0
            print(f"  {stem_name}: {count} ({pct:.1f}%)")

        return frame_labels


def main():
    """Example usage"""
    # Configuration
    mix_h5 = "fullmix_tracks.h5"
    stem_h5_files = [
        "vocals_tracks.h5",
        "guitar_tracks.h5",
        "bass_tracks.h5",
        # Drums: combine all 4 drum tracks into single "drums" stem
        ["drums_1_tracks.h5", "drums_2_tracks.h5", "drums_3_tracks.h5", "drums_4_tracks.h5"]
    ]
    stem_names = ["vocals", "guitar", "bass", "drums"]

    print("Frame-wise Stem Matcher")
    print("=" * 60)

    # Create matcher
    matcher = FramewiseStemMatcher(freq_threshold=100.0)

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
