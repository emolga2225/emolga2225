#!/usr/bin/env python3
"""
Chunk .h5 sinusoidal tracks into time segments.

Simply splits the raw sinusoidal data (frequencies, amplitudes, phases, frames)
into manageable time chunks. NO FFT, NO quantization - just raw precise data.

Each chunk contains all sinusoids within a time range.
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def chunk_h5_file(h5_path, frames_per_chunk=344):
    """Split .h5 sinusoidal data into time-based chunks"""

    chunks = []

    with h5py.File(h5_path, 'r') as f:
        # Get metadata
        sr = f.attrs.get('sr', 44100)
        n_channels = f.attrs.get('ch', 1)

        # Find max frame to determine number of chunks
        max_frame = 0
        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name in f:
                grp = f[grp_name]
                track_ends = grp['e'][:]
                if len(track_ends) > 0:
                    max_frame = max(max_frame, int(np.max(track_ends)))

        n_chunks = (max_frame // frames_per_chunk) + 1
        print(f"    Total frames: {max_frame}, creating {n_chunks} chunks")

        # Process all channels
        all_freqs = []
        all_amps = []
        all_phases = []
        all_frames = []

        for ch_idx in range(n_channels):
            grp_name = f'c{ch_idx}'
            if grp_name not in f:
                continue

            grp = f[grp_name]

            # Load all data for this channel
            frequencies = grp['f'][:]
            amplitudes = grp['a'][:]
            phases = grp['p'][:]
            frame_indices = grp['i'][:]

            all_freqs.append(frequencies)
            all_amps.append(amplitudes)
            all_phases.append(phases)
            all_frames.append(frame_indices)

        # Combine channels
        if len(all_freqs) > 0:
            combined_freqs = np.concatenate(all_freqs)
            combined_amps = np.concatenate(all_amps)
            combined_phases = np.concatenate(all_phases)
            combined_frames = np.concatenate(all_frames)
        else:
            return chunks

        # Vectorized chunking - assign each sinusoid to a chunk
        chunk_assignments = (combined_frames // frames_per_chunk).astype(np.int32)

        # Get unique chunk indices
        unique_chunks = np.unique(chunk_assignments)

        print(f"      Splitting into {len(unique_chunks)} chunks...")

        # Split by chunk assignment
        for chunk_idx in unique_chunks:
            mask = chunk_assignments == chunk_idx

            start_frame = chunk_idx * frames_per_chunk
            end_frame = start_frame + frames_per_chunk

            chunk_data = {
                'frequencies': combined_freqs[mask],
                'amplitudes': combined_amps[mask],
                'phases': combined_phases[mask],
                'frame_indices': combined_frames[mask] - start_frame,  # Make relative to chunk
                'start_frame': start_frame,
                'end_frame': end_frame,
                'sample_rate': sr,
                'n_sinusoids': np.sum(mask)
            }

            chunks.append(chunk_data)

    return chunks


def preprocess_song_directory(data_dir, stem_names, frames_per_chunk=344):
    """Chunk all .h5 files in a song directory"""

    data_dir = Path(data_dir)
    chunks_dir = data_dir / 'chunks'
    chunks_dir.mkdir(exist_ok=True)

    print(f"\nProcessing {data_dir.name}...")

    # Process fullmix_tracks.h5
    fullmix_h5 = data_dir / 'fullmix_tracks.h5'
    if fullmix_h5.exists():
        print(f"  Chunking fullmix_tracks.h5...")
        chunks = chunk_h5_file(fullmix_h5, frames_per_chunk)

        for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="    Saving chunks", leave=False):
            output_path = chunks_dir / f'fullmix_chunk_{idx:04d}.npz'
            np.savez_compressed(output_path, **chunk)

        print(f"    Saved {len(chunks)} chunks")
    else:
        print(f"  Skipping: fullmix_tracks.h5 not found")

    # Process each stem .h5
    for stem_name in stem_names:
        if stem_name == 'drums':
            # Process each drum file separately
            for i in range(1, 5):
                h5_path = data_dir / f'drums_{i}_tracks.h5'
                if h5_path.exists():
                    print(f"  Chunking drums_{i}_tracks.h5...")
                    chunks = chunk_h5_file(h5_path, frames_per_chunk)

                    for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="    Saving chunks", leave=False):
                        output_path = chunks_dir / f'drums_{i}_chunk_{idx:04d}.npz'
                        np.savez_compressed(output_path, **chunk)

                    print(f"    Saved {len(chunks)} chunks")
        else:
            h5_path = data_dir / f'{stem_name}_tracks.h5'
            if h5_path.exists():
                print(f"  Chunking {stem_name}_tracks.h5...")
                chunks = chunk_h5_file(h5_path, frames_per_chunk)

                for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="    Saving chunks", leave=False):
                    output_path = chunks_dir / f'{stem_name}_chunk_{idx:04d}.npz'
                    np.savez_compressed(output_path, **chunk)

                print(f"    Saved {len(chunks)} chunks")
            else:
                print(f"  Skipping: {stem_name}_tracks.h5 not found")


def main():
    parser = argparse.ArgumentParser(description='Chunk .h5 files into time segments')
    parser.add_argument('--data-dirs', nargs='+', required=True, help='Song directories to process')
    parser.add_argument('--stem-names', nargs='+', default=['vocals', 'guitar', 'bass', 'drums'],
                        help='Stem names to process')
    parser.add_argument('--frames-per-chunk', type=int, default=344,
                        help='Number of frames per chunk')

    args = parser.parse_args()

    print("=" * 60)
    print("H5 to Chunks Preprocessing")
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
    print("Chunking complete!")
    print("=" * 60)
    print("\nChunks saved to <song>/chunks/ directories")
    print("Each chunk contains raw sinusoidal data with full precision")


if __name__ == '__main__':
    main()
