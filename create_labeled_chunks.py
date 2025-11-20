#!/usr/bin/env python3
"""
Chunk .h5 sinusoidal tracks into labeled training data.

Creates training chunks where each fullmix sinusoid is pre-labeled
with its corresponding stem. This makes training fast since we don't
need to do nearest-neighbor matching on-the-fly.
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def load_all_h5_data(data_dir, stem_names):
    """Load all .h5 files for a song and combine into labeled dataset"""

    data_dir = Path(data_dir)

    # Load fullmix
    fullmix_h5 = data_dir / 'fullmix_tracks.h5'
    if not fullmix_h5.exists():
        print(f"  ERROR: Missing fullmix_tracks.h5")
        return None

    with h5py.File(fullmix_h5, 'r') as f:
        sr = f.attrs.get('sr', 44100)
        n_channels = f.attrs.get('ch', 1)

        # Collect all fullmix sinusoids
        all_freqs = []
        all_amps = []
        all_phases = []
        all_frames = []

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name in f:
                grp = f[grp_name]
                all_freqs.append(grp['f'][:])
                all_amps.append(grp['a'][:])
                all_phases.append(grp['p'][:])
                all_frames.append(grp['i'][:])

    if len(all_freqs) == 0:
        return None

    fullmix_data = {
        'frequencies': np.concatenate(all_freqs),
        'amplitudes': np.concatenate(all_amps),
        'phases': np.concatenate(all_phases),
        'frame_indices': np.concatenate(all_frames),
    }

    # Load each stem
    stems_data = {}
    for stem_name in stem_names:
        if stem_name == 'drums':
            # Combine all drum files
            drum_freqs, drum_amps, drum_phases, drum_frames = [], [], [], []

            for i in range(1, 5):
                drum_h5 = data_dir / f'drums_{i}_tracks.h5'
                if drum_h5.exists():
                    with h5py.File(drum_h5, 'r') as f:
                        n_channels = f.attrs.get('ch', 1)
                        for ch_idx in range(n_channels):
                            grp_name = f'c{ch_idx}'
                            if grp_name in f:
                                grp = f[grp_name]
                                drum_freqs.append(grp['f'][:])
                                drum_amps.append(grp['a'][:])
                                drum_phases.append(grp['p'][:])
                                drum_frames.append(grp['i'][:])

            if len(drum_freqs) > 0:
                stems_data['drums'] = {
                    'frequencies': np.concatenate(drum_freqs),
                    'amplitudes': np.concatenate(drum_amps),
                    'phases': np.concatenate(drum_phases),
                    'frame_indices': np.concatenate(drum_frames),
                }
        else:
            # Try primary name first, then alternatives
            alternatives = [stem_name]
            if stem_name == 'bass':
                alternatives.append('rhythm')  # Some songs use rhythm instead of bass

            stem_h5 = None
            for alt_name in alternatives:
                alt_path = data_dir / f'{alt_name}_tracks.h5'
                if alt_path.exists():
                    stem_h5 = alt_path
                    break

            if stem_h5 is not None:
                with h5py.File(stem_h5, 'r') as f:
                    n_channels = f.attrs.get('ch', 1)

                    stem_freqs, stem_amps, stem_phases, stem_frames = [], [], [], []
                    for ch_idx in range(n_channels):
                        grp_name = f'c{ch_idx}'
                        if grp_name in f:
                            grp = f[grp_name]
                            stem_freqs.append(grp['f'][:])
                            stem_amps.append(grp['a'][:])
                            stem_phases.append(grp['p'][:])
                            stem_frames.append(grp['i'][:])

                    if len(stem_freqs) > 0:
                        stems_data[stem_name] = {
                            'frequencies': np.concatenate(stem_freqs),
                            'amplitudes': np.concatenate(stem_amps),
                            'phases': np.concatenate(stem_phases),
                            'frame_indices': np.concatenate(stem_frames),
                        }

    # Check if all stems found
    if len(stems_data) != len(stem_names):
        print(f"  ERROR: Missing stems. Found {list(stems_data.keys())}, expected {stem_names}")
        return None

    return {
        'fullmix': fullmix_data,
        'stems': stems_data,
        'sample_rate': sr
    }


def create_labeled_chunks(song_data, stem_names, frames_per_chunk=344):
    """
    Create chunks with pre-assigned stem labels.

    For each chunk, we collect all stem sinusoids and label them.
    This creates a classification dataset.
    """

    fullmix = song_data['fullmix']
    stems = song_data['stems']

    # Collect ALL sinusoids (from all stems) with their labels
    all_freqs = []
    all_amps = []
    all_phases = []
    all_frames = []
    all_labels = []

    for stem_idx, stem_name in enumerate(stem_names):
        stem_data = stems[stem_name]
        n_sinusoids = len(stem_data['frequencies'])

        all_freqs.append(stem_data['frequencies'])
        all_amps.append(stem_data['amplitudes'])
        all_phases.append(stem_data['phases'])
        all_frames.append(stem_data['frame_indices'])
        all_labels.append(np.full(n_sinusoids, stem_idx, dtype=np.int32))

    # Concatenate all
    all_freqs = np.concatenate(all_freqs)
    all_amps = np.concatenate(all_amps)
    all_phases = np.concatenate(all_phases)
    all_frames = np.concatenate(all_frames)
    all_labels = np.concatenate(all_labels)

    print(f"  Total sinusoids across all stems: {len(all_freqs):,}")

    # Chunk by frame
    chunk_assignments = (all_frames // frames_per_chunk).astype(np.int32)
    unique_chunks = np.unique(chunk_assignments)

    print(f"  Splitting into {len(unique_chunks)} chunks...")

    chunks = []
    for chunk_idx in tqdm(unique_chunks, desc="  Creating chunks"):
        mask = chunk_assignments == chunk_idx

        chunk_data = {
            'frequencies': all_freqs[mask],
            'amplitudes': all_amps[mask],
            'phases': all_phases[mask],
            'frame_indices': all_frames[mask] - (chunk_idx * frames_per_chunk),  # Relative
            'labels': all_labels[mask],  # STEM LABELS
            'start_frame': chunk_idx * frames_per_chunk,
            'n_sinusoids': np.sum(mask)
        }

        chunks.append(chunk_data)

    return chunks


def preprocess_song_directory(data_dir, stem_names, frames_per_chunk=344):
    """Process a song directory and create labeled chunks"""

    data_dir = Path(data_dir)
    chunks_dir = data_dir / 'labeled_chunks'
    chunks_dir.mkdir(exist_ok=True)

    print(f"\nProcessing {data_dir.name}...")

    # Load all data
    song_data = load_all_h5_data(data_dir, stem_names)
    if song_data is None:
        print(f"  Skipping due to errors")
        return

    # Create labeled chunks
    chunks = create_labeled_chunks(song_data, stem_names, frames_per_chunk)

    # Save chunks
    print(f"  Saving {len(chunks)} chunks...")
    for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="  Saving"):
        output_path = chunks_dir / f'chunk_{idx:04d}.npz'
        np.savez_compressed(output_path, **chunk)

    print(f"  Done! Saved {len(chunks)} labeled chunks")


def main():
    parser = argparse.ArgumentParser(description='Create labeled training chunks from .h5 files')
    parser.add_argument('--data-dirs', nargs='+', required=True, help='Song directories to process')
    parser.add_argument('--stem-names', nargs='+', default=['vocals', 'guitar', 'bass', 'drums'],
                        help='Stem names')
    parser.add_argument('--frames-per-chunk', type=int, default=344,
                        help='Number of frames per chunk')

    args = parser.parse_args()

    print("=" * 60)
    print("Creating Labeled Training Chunks")
    print("=" * 60)
    print(f"Processing {len(args.data_dirs)} song(s)...")
    print(f"Stems: {args.stem_names}")
    print(f"Frames per chunk: {args.frames_per_chunk}")

    for data_dir in args.data_dirs:
        preprocess_song_directory(
            data_dir,
            args.stem_names,
            args.frames_per_chunk
        )

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print("\nLabeled chunks saved to <song>/labeled_chunks/ directories")
    print("Each chunk contains sinusoids from ALL stems with labels")


if __name__ == '__main__':
    main()
