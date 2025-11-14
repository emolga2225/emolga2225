import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import json

class StemMatcher:
    """Match sinusoidal tracks between mix and stems"""

    def __init__(self, freq_threshold=50.0, time_threshold=0.1):
        """
        Args:
            freq_threshold: Maximum frequency difference in Hz for matching
            time_threshold: Maximum time difference in seconds for matching
        """
        self.freq_threshold = freq_threshold
        self.time_threshold = time_threshold

    def load_tracks(self, h5_file, channel=0):
        """Load all tracks from an HDF5 file"""
        tracks = []

        with h5py.File(h5_file, 'r') as f:
            grp = f[f'c{channel}']
            sample_rate = f.attrs['sr']
            fft_size = f.attrs['fft']
            hop_size = f.attrs['hop']

            bandwidth = fft_size * (6000.0 / 1024.0)
            band_sr = int(bandwidth * 2)

            track_lens = grp['len'][:]
            track_ids = grp['id'][:]
            track_starts = grp['s'][:]
            track_ends = grp['e'][:]
            track_bands = grp['b'][:]
            track_hops = grp['h'][:]

            all_freqs = grp['f'][:]
            all_amps = grp['a'][:]
            all_phases = grp['p'][:]
            all_indices = grp['i'][:]

            offset = 0
            for i, n in enumerate(track_lens):
                freqs = all_freqs[offset:offset+n]
                amps = all_amps[offset:offset+n]
                phases = all_phases[offset:offset+n]
                indices = all_indices[offset:offset+n]

                hop_size_track = int(track_hops[i])
                times = indices * hop_size_track / band_sr

                tracks.append({
                    'id': int(track_ids[i]),
                    'start_frame': int(track_starts[i]),
                    'end_frame': int(track_ends[i]),
                    'band': int(track_bands[i]),
                    'frequencies': freqs,
                    'amplitudes': amps,
                    'phases': phases,
                    'times': times,
                    'mean_freq': np.mean(freqs),
                    'mean_amp': np.mean(amps),
                    'start_time': times[0] if len(times) > 0 else 0,
                    'end_time': times[-1] if len(times) > 0 else 0,
                    'duration': times[-1] - times[0] if len(times) > 1 else 0
                })

                offset += n

        return tracks

    def compute_track_similarity(self, mix_track, stem_track):
        """Compute similarity between a mix track and stem track"""

        # Frequency similarity (mean frequency difference)
        freq_diff = abs(mix_track['mean_freq'] - stem_track['mean_freq'])
        if freq_diff > self.freq_threshold:
            return 0.0

        # Temporal overlap
        overlap_start = max(mix_track['start_time'], stem_track['start_time'])
        overlap_end = min(mix_track['end_time'], stem_track['end_time'])
        overlap = max(0, overlap_end - overlap_start)

        if overlap < self.time_threshold:
            return 0.0

        # Compute overlap ratio
        mix_duration = mix_track['duration']
        stem_duration = stem_track['duration']
        if mix_duration == 0 or stem_duration == 0:
            return 0.0

        overlap_ratio = overlap / max(mix_duration, stem_duration)

        # Frequency similarity score (inverse of difference, normalized)
        freq_similarity = 1.0 - (freq_diff / self.freq_threshold)

        # Combined score
        similarity = overlap_ratio * freq_similarity

        return similarity

    def match_stems(self, mix_h5, stem_h5_files, stem_names, channel=0):
        """
        Match mix tracks to stem tracks using Hungarian algorithm

        Args:
            mix_h5: Path to mix HDF5 file
            stem_h5_files: List of paths to stem HDF5 files
            stem_names: List of stem names
            channel: Channel index to process

        Returns:
            Dictionary mapping mix track IDs to stem labels
        """
        print(f"Loading mix tracks from {mix_h5}...")
        mix_tracks = self.load_tracks(mix_h5, channel)
        print(f"  Found {len(mix_tracks)} mix tracks")

        # Load all stem tracks
        all_stem_tracks = []
        stem_track_labels = []

        for stem_idx, (stem_file, stem_name) in enumerate(zip(stem_h5_files, stem_names)):
            print(f"Loading {stem_name} tracks from {stem_file}...")
            stem_tracks = self.load_tracks(stem_file, channel)
            print(f"  Found {len(stem_tracks)} {stem_name} tracks")

            all_stem_tracks.extend(stem_tracks)
            stem_track_labels.extend([stem_idx] * len(stem_tracks))

        if len(all_stem_tracks) == 0:
            print("Warning: No stem tracks found!")
            return {}

        # Build cost matrix for Hungarian algorithm
        print("\nComputing similarity matrix...")
        n_mix = len(mix_tracks)
        n_stem = len(all_stem_tracks)

        cost_matrix = np.zeros((n_mix, n_stem))

        for i, mix_track in enumerate(tqdm(mix_tracks, desc="Computing similarities")):
            for j, stem_track in enumerate(all_stem_tracks):
                similarity = self.compute_track_similarity(mix_track, stem_track)
                cost_matrix[i, j] = -similarity  # Negative because we minimize cost

        # Solve assignment problem
        print("\nSolving assignment with Hungarian algorithm...")
        mix_indices, stem_indices = linear_sum_assignment(cost_matrix)

        # Create mapping
        track_labels = {}
        matched_count = 0

        for mix_idx, stem_idx in zip(mix_indices, stem_indices):
            similarity = -cost_matrix[mix_idx, stem_idx]

            if similarity > 0.01:  # Threshold for valid match
                mix_track_id = mix_tracks[mix_idx]['id']
                stem_label = stem_track_labels[stem_idx]
                track_labels[mix_track_id] = {
                    'stem_idx': stem_label,
                    'stem_name': stem_names[stem_label],
                    'similarity': float(similarity)
                }
                matched_count += 1

        print(f"\nMatched {matched_count}/{len(mix_tracks)} mix tracks to stems")

        # Print statistics per stem
        stem_counts = {name: 0 for name in stem_names}
        for track_info in track_labels.values():
            stem_counts[track_info['stem_name']] += 1

        print("\nTracks per stem:")
        for stem_name, count in stem_counts.items():
            print(f"  {stem_name}: {count}")

        return track_labels


def main():
    """Example usage"""
    import sys

    # Configuration
    mix_h5 = "mix_tracks.h5"  # Replace with your mix HDF5 file
    stem_h5_files = [
        "vocals_tracks.h5",
        "drums_tracks.h5",
        "bass_tracks.h5",
        "other_tracks.h5"
    ]
    stem_names = ["vocals", "drums", "bass", "other"]

    print("Stem Matcher")
    print("=" * 50)

    # Create matcher
    matcher = StemMatcher(freq_threshold=100.0, time_threshold=0.05)

    # Match stems
    track_labels = matcher.match_stems(
        mix_h5=mix_h5,
        stem_h5_files=stem_h5_files,
        stem_names=stem_names,
        channel=0
    )

    # Save labels
    output_file = "track_labels.json"
    with open(output_file, 'w') as f:
        json.dump(track_labels, f, indent=2)

    print(f"\nSaved labels to {output_file}")


if __name__ == "__main__":
    main()
