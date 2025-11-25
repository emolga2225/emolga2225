#!/usr/bin/env python3
"""
Test/inference script for the stem separator model.

Usage with HDF5 (synchrosqueezed sinusoids):
    python test_stem_separator.py \
        --checkpoint stem_separator_exact_freq_epoch10.pt \
        --h5-file path/to/song_sinusoids.h5 \
        --output-dir separated_stems/

Usage with audio file (extracts sinusoids with simple STFT):
    python test_stem_separator.py \
        --checkpoint stem_separator_exact_freq_epoch10.pt \
        --audio-file test.wav \
        --output-dir separated_stems/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
import h5py
from pathlib import Path
import argparse
from tqdm import tqdm


# Import the model from training script
class TransformerStemSeparator(nn.Module):
    """Same model as in training script - WITH STEM CONDITIONING"""

    def __init__(self, n_stems=5, max_sinusoids=900, d_model=128, nhead=4,
                 num_layers=4, dim_feedforward=512):
        super().__init__()
        self.n_stems = n_stems
        self.max_sinusoids = max_sinusoids
        self.d_model = d_model

        # Embed each sinusoid [freq, amp, phase] -> d_model dimensions
        self.sinusoid_embed = nn.Linear(3, d_model)

        # Positional encoding for sinusoid index (which sinusoid in the frame)
        self.sinusoid_pos_embed = nn.Parameter(torch.randn(1, max_sinusoids, d_model))

        # STEM CONDITIONING: Learnable embeddings for each stem type
        # This tells the model "extract vocals" vs "extract drums" etc.
        self.stem_embeddings = nn.Embedding(n_stems, d_model)

        # Frame positional embeddings for temporal context (which frame in time)
        # This is separate from sinusoid position - tells model temporal order
        self.max_frames = 10  # Support up to 10 frames per chunk
        self.frame_pos_embed = nn.Parameter(torch.randn(1, self.max_frames, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Single output head shared across stems (stem conditioning handles differentiation)
        self.output_head = nn.Linear(d_model, 3)  # Predict [freq, amp, phase]

        # Initialize weights with better variance
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with reasonable values"""
        # Initialize embedding layers
        nn.init.xavier_uniform_(self.sinusoid_embed.weight, gain=0.1)
        nn.init.zeros_(self.sinusoid_embed.bias)

        # Initialize output head
        nn.init.xavier_uniform_(self.output_head.weight, gain=0.1)
        nn.init.zeros_(self.output_head.bias)

        # Initialize positional embeddings
        nn.init.normal_(self.sinusoid_pos_embed, mean=0, std=0.02)
        nn.init.normal_(self.frame_pos_embed, mean=0, std=0.02)

        # Initialize stem embeddings with larger variance (they're important!)
        nn.init.normal_(self.stem_embeddings.weight, mean=0, std=0.1)

    def forward(self, x):
        """
        x: (batch, n_frames, max_sinusoids, 3)
        returns: (batch, n_stems, n_frames, max_sinusoids, 3)

        NEW: Process ALL stems in parallel with batched stem conditioning!
        Much faster than sequential processing - uses GPU parallelism.
        """
        batch_size, n_frames, max_sines, _ = x.shape
        seq_len = n_frames * max_sines

        # NORMALIZE INPUTS to prevent numerical instability
        # x[..., 0] = frequency (Hz), x[..., 1] = amplitude, x[..., 2] = phase (radians)
        x_norm = x.clone()

        # Normalize frequency to [0, 1] range (48kHz sample rate, Nyquist = 24kHz)
        x_norm[..., 0] = x[..., 0] / 24000.0

        # Log-scale amplitude (handles large dynamic range, prevents huge values)
        # Use log1p to handle zero amplitudes gracefully
        x_norm[..., 1] = torch.log1p(torch.abs(x[..., 1])) / 10.0  # Scale down by 10

        # Phase already in [-π, π] range - normalize to [-1, 1]
        x_norm[..., 2] = x[..., 2] / 3.14159265

        # Flatten frames into sequence: (batch, n_frames, max_sinusoids, 3) -> (batch, seq_len, 3)
        x_flat = x_norm.reshape(batch_size, seq_len, 3)

        # Embed sinusoids: (batch, seq_len, 3) -> (batch, seq_len, d_model)
        x_embed = self.sinusoid_embed(x_flat)

        # Add sinusoid positional encoding (which sinusoid within each frame)
        pos_embed = self.sinusoid_pos_embed.repeat(1, n_frames, 1)
        x_embed = x_embed + pos_embed

        # Add frame positional encoding (which frame in time)
        x_embed_frames = x_embed.reshape(batch_size, n_frames, max_sines, self.d_model)
        x_embed_frames = x_embed_frames + self.frame_pos_embed[:, :n_frames, :, :]
        x_embed = x_embed_frames.reshape(batch_size, seq_len, self.d_model)

        # BATCHED STEM CONDITIONING: Process all stems in parallel!
        # Expand input for all stems: (batch, seq_len, d_model) -> (batch, n_stems, seq_len, d_model)
        # Use repeat() instead of expand() for better numerical stability with mixed precision
        x_embed_expanded = x_embed.unsqueeze(1).repeat(1, self.n_stems, 1, 1)

        # Get all stem embeddings: (n_stems, d_model)
        stem_embeds = self.stem_embeddings.weight  # All stem embeddings at once

        # Expand stem embeddings to match input: (n_stems, d_model) -> (batch, n_stems, seq_len, d_model)
        # Use repeat() for memory allocation (better with autocast)
        stem_embeds_expanded = stem_embeds.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, seq_len, 1)

        # Add stem conditioning: (batch, n_stems, seq_len, d_model)
        x_conditioned = x_embed_expanded + stem_embeds_expanded

        # Reshape to batch all stems together: (batch, n_stems, seq_len, d_model) -> (batch * n_stems, seq_len, d_model)
        x_conditioned = x_conditioned.reshape(batch_size * self.n_stems, seq_len, self.d_model)

        # Create padding mask for batched input
        padding_mask = (x_flat[:, :, 0] == 0)  # (batch, seq_len)
        padding_mask_expanded = padding_mask.unsqueeze(1).expand(-1, self.n_stems, -1).reshape(batch_size * self.n_stems, seq_len)

        # Single transformer call for ALL stems at once! (batch * n_stems as batch dimension)
        encoded = self.transformer(x_conditioned, src_key_padding_mask=padding_mask_expanded)

        # Reshape back: (batch * n_stems, seq_len, d_model) -> (batch, n_stems, seq_len, d_model)
        encoded = encoded.reshape(batch_size, self.n_stems, seq_len, self.d_model)

        # Reshape to frames: (batch, n_stems, seq_len, d_model) -> (batch, n_stems, n_frames, max_sines, d_model)
        encoded = encoded.reshape(batch_size, self.n_stems, n_frames, max_sines, self.d_model)

        # Predict sinusoid parameters for all stems at once
        stem_output = self.output_head(encoded)  # (batch, n_stems, n_frames, max_sines, 3)

        # DENORMALIZE OUTPUTS back to original scale
        # Model outputs normalized values, convert back to Hz/amplitude/radians

        # Frequency: [0, 1] -> [0, 24000] Hz (48kHz sample rate)
        freq = torch.sigmoid(stem_output[:, :, :, :, 0]) * 24000.0

        # Amplitude: log-scaled -> linear scale
        # Model outputs log1p(amp)/10, so reverse: amp = expm1(output * 10)
        amp = torch.expm1(torch.relu(stem_output[:, :, :, :, 1]) * 10.0)

        # Phase: [-1, 1] -> [-π, π] radians
        phase = torch.tanh(stem_output[:, :, :, :, 2]) * 3.14159265

        # Stack into final output: (batch, n_stems, n_frames, max_sinusoids, 3)
        output = torch.stack([freq, amp, phase], dim=-1)

        return output


def explore_h5_structure(f, prefix=''):
    """Recursively explore HDF5 structure"""
    items = []
    for key in f.keys():
        path = f"{prefix}/{key}" if prefix else key
        item = f[key]
        if isinstance(item, h5py.Group):
            items.append(f"GROUP: {path}/")
            items.extend(explore_h5_structure(item, path))
        elif isinstance(item, h5py.Dataset):
            items.append(f"DATASET: {path} {item.shape} {item.dtype}")
    return items


def load_sinusoids_from_h5(h5_file, max_sinusoids=900, channel=None):
    """
    Load synchrosqueezed sinusoids from HDF5 file.

    Supports multiple HDF5 structures:
    1. /fullmix/freqs, /fullmix/amps, /fullmix/phases
    2. /freqs, /amps, /phases
    3. /c0/freqs, /c0/amps, /c0/phases (channel-based)
    4. /c0, /c1, ... (direct arrays where each is (n_frames, n_sinusoids, 3))

    Returns: tuple of (sinusoids, metadata)
        sinusoids: (n_frames, max_sinusoids, 3) where 3 = [freq, amp, phase]
        metadata: dict with 'sample_rate', 'fft_size', 'hop_size', etc. if available
    """
    print(f"Loading sinusoids from HDF5: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        root_keys = list(f.keys())
        print(f"Available keys in HDF5: {root_keys}")

        # Try to read metadata if available
        metadata = {}
        if 'sr' in f.attrs:
            metadata['sample_rate'] = int(f.attrs['sr'])
            print(f"Found metadata: sample_rate = {metadata['sample_rate']} Hz")
        if 'fft' in f.attrs:
            metadata['fft_size'] = int(f.attrs['fft'])
            print(f"Found metadata: fft_size = {metadata['fft_size']}")
        if 'hop' in f.attrs:
            metadata['hop_size'] = int(f.attrs['hop'])
            print(f"Found metadata: hop_size = {metadata['hop_size']}")

        # Calculate derived parameters
        if 'fft_size' in metadata:
            bandwidth = metadata['fft_size'] * (6000.0 / 1024.0)
            band_sr = int(bandwidth * 2)
            metadata['bandwidth'] = bandwidth
            metadata['band_sr'] = band_sr
            print(f"Calculated: bandwidth = {bandwidth:.1f} Hz, band_sr = {band_sr} Hz")

        # Try different possible structures
        if 'fullmix' in f:
            # Structure: /fullmix/freqs, /fullmix/amps, /fullmix/phases
            group = f['fullmix']
            freqs = np.array(group['freqs'])
            amps = np.array(group['amps'])
            phases = np.array(group['phases'])

        elif 'freqs' in f:
            # Flat structure: /freqs, /amps, /phases
            freqs = np.array(f['freqs'])
            amps = np.array(f['amps'])
            phases = np.array(f['phases'])

        elif any(k.startswith('c') for k in root_keys):
            # Channel-based structure: /c0, /c1, etc.
            # Check if these are groups or datasets
            first_channel = root_keys[0]

            if isinstance(f[first_channel], h5py.Group):
                # Structure: /c0/freqs, /c0/amps, /c0/phases OR sparse format
                if channel is None:
                    channel = first_channel
                    print(f"Multiple channels found {root_keys}, using '{channel}'")

                group = f[channel]
                group_keys = list(group.keys())

                if 'freqs' in group:
                    # Dense format: /c0/freqs, /c0/amps, /c0/phases
                    freqs = np.array(group['freqs'])
                    amps = np.array(group['amps'])
                    phases = np.array(group['phases'])

                elif 'f' in group and 'a' in group and 'p' in group and 'i' in group:
                    # Sparse format: /c0/f (freq), /c0/a (amp), /c0/p (phase), /c0/i (frame indices)
                    # This is the same format as the preprocessor uses
                    print(f"Detected sparse HDF5 format with {len(group['f'])} total sinusoids")

                    # Load flat arrays
                    track_lens = np.array(group['len'])
                    frequencies = np.array(group['f'])
                    amplitudes = np.array(group['a'])
                    phases = np.array(group['p'])
                    frame_indices = np.array(group['i'])

                    # Group sinusoids by frame (same logic as preprocessor)
                    from collections import defaultdict
                    frame_sinusoids = defaultdict(list)

                    offset = 0
                    for track_len in track_lens:
                        track_freqs = frequencies[offset:offset + track_len]
                        track_amps = amplitudes[offset:offset + track_len]
                        track_phases = phases[offset:offset + track_len]
                        track_frames = frame_indices[offset:offset + track_len]

                        # Add all sinusoids from this track to their respective frames
                        for freq, amp, phase, frame_idx in zip(track_freqs, track_amps, track_phases, track_frames):
                            frame_sinusoids[int(frame_idx)].append([freq, amp, phase])

                        offset += track_len

                    # Find total number of frames
                    max_frame_idx = max(frame_sinusoids.keys())
                    n_frames = max_frame_idx + 1

                    print(f"Converting {n_frames} sparse frames to dense format...")

                    # Convert to dense format (n_frames, max_sinusoids, 3)
                    sinusoids = np.zeros((n_frames, max_sinusoids, 3), dtype=np.float32)

                    for frame_idx, sines in frame_sinusoids.items():
                        # Sort by frequency for consistency
                        sines = sorted(sines, key=lambda x: x[0])
                        n_sines = min(len(sines), max_sinusoids)
                        sinusoids[frame_idx, :n_sines, :] = sines[:n_sines]

                    print(f"Converted to dense array: {sinusoids.shape}")
                    return sinusoids, metadata

                else:
                    print(f"\nStructure of /{channel}/:")
                    for key in group.keys():
                        print(f"  {key}: {group[key].shape} {group[key].dtype}")
                    raise ValueError(f"Channel '{channel}' format not recognized. Expected 'freqs'/'amps'/'phases' or 'f'/'a'/'p'")

            elif isinstance(f[first_channel], h5py.Dataset):
                # Direct arrays: /c0, /c1 where each might be (n_frames, n_sinusoids, 3)
                if channel is None:
                    channel = first_channel
                    print(f"Multiple channels found {root_keys}, using '{channel}'")

                data = np.array(f[channel])
                print(f"Dataset shape: {data.shape}, dtype: {data.dtype}")

                # Check if it's already in the right format (n_frames, n_sinusoids, 3)
                if len(data.shape) == 3 and data.shape[2] == 3:
                    freqs = data[:, :, 0]
                    amps = data[:, :, 1]
                    phases = data[:, :, 2]
                # Or maybe it's (3, n_frames, n_sinusoids)
                elif len(data.shape) == 3 and data.shape[0] == 3:
                    freqs = data[0]
                    amps = data[1]
                    phases = data[2]
                else:
                    raise ValueError(f"Unexpected dataset shape: {data.shape}. Expected (n_frames, n_sinusoids, 3) or (3, n_frames, n_sinusoids)")

        else:
            # Unknown structure - print full exploration
            print("\nFull HDF5 structure:")
            for line in explore_h5_structure(f):
                print(f"  {line}")
            raise ValueError("Could not find sinusoid data in HDF5. See structure above.")

    n_frames = freqs.shape[0]
    n_sinusoids = freqs.shape[1]

    print(f"Found {n_frames} frames with {n_sinusoids} sinusoids each")

    # Create padded array to match max_sinusoids
    sinusoids = np.zeros((n_frames, max_sinusoids, 3), dtype=np.float32)

    # Copy data (pad or truncate to max_sinusoids)
    n_copy = min(n_sinusoids, max_sinusoids)
    sinusoids[:, :n_copy, 0] = freqs[:, :n_copy]
    sinusoids[:, :n_copy, 1] = amps[:, :n_copy]
    sinusoids[:, :n_copy, 2] = phases[:, :n_copy]

    return sinusoids, metadata


def extract_sinusoids_simple(audio, sr=44100, n_fft=1024, hop_length=512, max_sinusoids=900):
    """
    Simple sinusoid extraction using STFT peaks.
    Returns: (n_frames, max_sinusoids, 3) where 3 = [freq, amp, phase]
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    n_frames = stft.shape[1]
    sinusoids = np.zeros((n_frames, max_sinusoids, 3), dtype=np.float32)

    # For each frame, extract top sinusoids
    for frame_idx in range(n_frames):
        frame_mag = mag[:, frame_idx]
        frame_phase = phase[:, frame_idx]

        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(frame_mag) - 1):
            if frame_mag[i] > frame_mag[i-1] and frame_mag[i] > frame_mag[i+1]:
                peaks.append(i)

        # Sort by magnitude and take top max_sinusoids
        peaks = sorted(peaks, key=lambda i: frame_mag[i], reverse=True)[:max_sinusoids]

        # Store sinusoids
        for sine_idx, bin_idx in enumerate(peaks):
            freq = bin_idx * sr / n_fft
            amp = frame_mag[bin_idx]
            p = frame_phase[bin_idx]

            sinusoids[frame_idx, sine_idx] = [freq, amp, p]

    return sinusoids


def sinusoids_to_audio(sinusoids, sr=44100, hop_length=512, desc="Reconstructing"):
    """
    Reconstruct audio from sinusoids using overlap-add.
    sinusoids: (n_frames, max_sinusoids, 3) where 3 = [freq, amp, phase]
    """
    n_frames = sinusoids.shape[0]
    audio_len = n_frames * hop_length + 1024  # Extra for final frame
    audio = np.zeros(audio_len, dtype=np.float32)

    for frame_idx in tqdm(range(n_frames), desc=desc, leave=False):
        t_start = frame_idx * hop_length / sr
        t = np.arange(hop_length) / sr

        frame_audio = np.zeros(hop_length, dtype=np.float32)

        # Sum all sinusoids in this frame
        for sine_idx in range(sinusoids.shape[1]):
            freq, amp, phase = sinusoids[frame_idx, sine_idx]

            if freq > 0:  # Non-zero frequency = valid sinusoid
                sine_wave = amp * np.cos(2 * np.pi * freq * t + phase)
                frame_audio += sine_wave

        # Overlap-add
        start_sample = frame_idx * hop_length
        audio[start_sample:start_sample + hop_length] += frame_audio

    # Normalize to prevent clipping
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pt file)')

    # Input: either audio file or h5 file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--audio-file', help='Input audio file (WAV) to separate')
    input_group.add_argument('--h5-file', help='Input HDF5 file with synchrosqueezed sinusoids')

    parser.add_argument('--output-dir', default='separated_stems', help='Output directory for separated stems')
    parser.add_argument('--max-sinusoids', type=int, default=2000)
    parser.add_argument('--chunk-frames', type=int, default=1,
                       help='Frames per inference chunk (1=safest, 3=match training, 10=faster but uses more memory)')
    parser.add_argument('--channel', default=None,
                       help='HDF5 channel to load (e.g., "c0", "c1"). Default: auto-detect first channel')
    parser.add_argument('--hop-length', type=int, default=512,
                       help='Hop length for audio reconstruction (default: 512 for STFT, use ~18485 for synchrosqueezed)')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config from checkpoint
    config = checkpoint['config']
    stem_names = config['stem_names']

    print(f"Stems: {stem_names}")
    print(f"Max sinusoids: {config['max_sinusoids']}")

    # Create model
    model = TransformerStemSeparator(
        n_stems=len(stem_names),
        max_sinusoids=config['max_sinusoids'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Load or extract sinusoids
    if args.h5_file:
        # Load pre-extracted sinusoids from HDF5
        sinusoids, metadata = load_sinusoids_from_h5(args.h5_file, max_sinusoids=args.max_sinusoids, channel=args.channel)

        # Use sample rate from metadata if available
        sr = metadata.get('sample_rate', 44100)
        print(f"Using sample rate: {sr} Hz")

        # Auto-detect hop_length from metadata if not specified
        if args.hop_length == 512 and 'band_sr' in metadata and 'hop_size' in metadata:
            # Calculate correct hop_length from metadata
            # hop_at_band_sr * (original_sr / band_sr) = hop_at_original_sr
            ssq_hop = 16  # Synchrosqueeze hop from extractor
            band_sr = metadata['band_sr']
            auto_hop = int((ssq_hop / band_sr) * sr)
            print(f"Auto-detected hop_length: {auto_hop} (from metadata)")
            args.hop_length = auto_hop
    else:
        # Extract sinusoids from audio file
        print(f"\nLoading audio: {args.audio_file}")
        audio, sr = librosa.load(args.audio_file, sr=44100, mono=True)
        print(f"Audio: {len(audio)/sr:.2f}s @ {sr}Hz")

        print("\nExtracting sinusoids...")
        sinusoids = extract_sinusoids_simple(audio, sr=sr, max_sinusoids=args.max_sinusoids)
        print(f"Extracted {sinusoids.shape[0]} frames")

    # Prepare for model
    sinusoids_tensor = torch.from_numpy(sinusoids).unsqueeze(0).to(device)  # (1, n_frames, max_sines, 3)

    # Estimate memory usage
    seq_len = args.chunk_frames * args.max_sinusoids
    attn_elements = seq_len * seq_len
    attn_gb = (attn_elements * 4) / (1024**3)
    if attn_gb > 2.0:
        print(f"\n⚠️  WARNING: Large attention matrix ({attn_gb:.1f} GB)")
        print(f"   Consider reducing --chunk-frames to avoid OOM")
        print(f"   Recommended: --chunk-frames 1 or 3")

    # Run model
    print(f"\nSeparating stems (processing {args.chunk_frames} frame(s) at a time)...")
    with torch.no_grad():
        # Process in chunks to avoid OOM
        n_frames = sinusoids_tensor.shape[1]
        all_predictions = []

        for i in tqdm(range(0, n_frames, args.chunk_frames)):
            chunk = sinusoids_tensor[:, i:i+args.chunk_frames]
            pred = model(chunk)  # (1, n_stems, chunk_frames, max_sines, 3)
            all_predictions.append(pred.cpu())

        # Concatenate predictions
        predictions = torch.cat(all_predictions, dim=2)  # (1, n_stems, n_frames, max_sines, 3)

    predictions = predictions[0].numpy()  # (n_stems, n_frames, max_sines, 3)

    # Diagnostic: Check prediction statistics
    print("\nPrediction statistics:")
    for stem_idx, stem_name in enumerate(stem_names):
        stem_preds = predictions[stem_idx]
        freqs = stem_preds[:, :, 0]
        amps = stem_preds[:, :, 1]
        phases = stem_preds[:, :, 2]

        # Count non-zero sinusoids
        non_zero = (amps > 0.001).sum()

        print(f"  {stem_name}:")
        print(f"    Freq range: [{freqs.min():.2f}, {freqs.max():.2f}] Hz")
        print(f"    Amp range: [{amps.min():.6f}, {amps.max():.6f}]")
        print(f"    Non-zero amps (>0.001): {non_zero}/{amps.size}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct and save each stem
    print("\nReconstructing audio...")
    for stem_idx, stem_name in enumerate(stem_names):
        print(f"  {stem_name}...")
        stem_sinusoids = predictions[stem_idx]  # (n_frames, max_sines, 3)

        # Reconstruct audio with progress bar
        stem_audio = sinusoids_to_audio(stem_sinusoids, sr=sr, hop_length=args.hop_length, desc=f"    {stem_name}")

        # Save
        output_file = output_dir / f"{stem_name}.wav"
        sf.write(output_file, stem_audio, sr)
        print(f"    Saved: {output_file}")

    print("\n✨ Done! Separated stems saved to:", output_dir)
    print("\nNOTE: Since the model was trained with NaN loss, the results")
    print("      will likely sound terrible or be silent. This is just for fun!")
    print("      Retrain with proper loss values for real results.")


if __name__ == "__main__":
    main()
