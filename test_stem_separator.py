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
    """Same model as in training script"""

    def __init__(self, n_stems=5, max_sinusoids=2000, d_model=128, nhead=4,
                 num_layers=4, dim_feedforward=512):
        super().__init__()
        self.n_stems = n_stems
        self.max_sinusoids = max_sinusoids
        self.d_model = d_model

        # Embed each sinusoid [freq, amp, phase] -> d_model dimensions
        self.sinusoid_embed = nn.Linear(3, d_model)

        # Positional encoding for sinusoid index
        self.sinusoid_pos_embed = nn.Parameter(torch.randn(1, max_sinusoids, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads - one per stem
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, 3)  # Predict [freq, amp, phase] for each stem
            for _ in range(n_stems)
        ])

        # Initialize weights with smaller variance for numerical stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values to prevent early NaN"""
        # Initialize embedding layers
        nn.init.xavier_uniform_(self.sinusoid_embed.weight, gain=0.01)
        nn.init.zeros_(self.sinusoid_embed.bias)

        # Initialize output heads with small weights
        for head in self.output_heads:
            nn.init.xavier_uniform_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

        # Initialize positional embeddings
        nn.init.normal_(self.sinusoid_pos_embed, mean=0, std=0.01)

    def forward(self, x):
        """
        x: (batch, n_frames, max_sinusoids, 3)
        returns: (batch, n_stems, n_frames, max_sinusoids, 3)
        """
        batch_size, n_frames, max_sines, _ = x.shape

        if n_frames == 1:
            # Single frame: process efficiently
            frame_sines = x[:, 0, :, :]  # (batch, max_sinusoids, 3)

            # Embed sinusoids
            frame_embed = self.sinusoid_embed(frame_sines)
            frame_embed = frame_embed + self.sinusoid_pos_embed

            # Padding mask
            padding_mask = (frame_sines[:, :, 0] == 0)

            # Transformer
            encoded = self.transformer(frame_embed, src_key_padding_mask=padding_mask)

            # Predict for each stem
            stem_preds = []
            for stem_idx in range(self.n_stems):
                stem_output = self.output_heads[stem_idx](encoded)

                # Clamp outputs to prevent extreme values
                freq = torch.clamp(F.relu(stem_output[:, :, 0]) * 22050 / 100, 0, 22050)
                amp = torch.clamp(F.relu(stem_output[:, :, 1]), 0, 100)
                phase = torch.atan2(torch.sin(stem_output[:, :, 2]),
                                   torch.cos(stem_output[:, :, 2]))

                stem_prediction = torch.stack([freq, amp, phase], dim=-1)
                stem_preds.append(stem_prediction)

            # Stack and add frame dimension
            output = torch.stack(stem_preds, dim=1).unsqueeze(2)  # (batch, n_stems, 1, max_sines, 3)

        else:
            # Multi-frame: flatten all frames into one sequence for temporal context!
            x_flat = x.reshape(batch_size, n_frames * max_sines, 3)

            # Embed sinusoids
            x_embed = self.sinusoid_embed(x_flat)

            # Add positional encodings (tile for each frame)
            pos_embed = self.sinusoid_pos_embed.repeat(1, n_frames, 1)
            x_embed = x_embed + pos_embed

            # Create padding mask
            padding_mask = (x_flat[:, :, 0] == 0)

            # Transformer on full sequence (provides temporal context!)
            encoded = self.transformer(x_embed, src_key_padding_mask=padding_mask)

            # Reshape back
            encoded = encoded.reshape(batch_size, n_frames, max_sines, self.d_model)

            # Predict for each stem
            stem_preds = []
            for stem_idx in range(self.n_stems):
                stem_output = self.output_heads[stem_idx](encoded)

                # Clamp outputs to prevent extreme values
                freq = torch.clamp(F.relu(stem_output[:, :, :, 0]) * 22050 / 100, 0, 22050)
                amp = torch.clamp(F.relu(stem_output[:, :, :, 1]), 0, 100)
                phase = torch.atan2(torch.sin(stem_output[:, :, :, 2]),
                                   torch.cos(stem_output[:, :, :, 2]))

                stem_prediction = torch.stack([freq, amp, phase], dim=-1)
                stem_preds.append(stem_prediction)

            # Stack stems
            output = torch.stack(stem_preds, dim=1)

        return output


def load_sinusoids_from_h5(h5_file, max_sinusoids=2000):
    """
    Load synchrosqueezed sinusoids from HDF5 file.

    Expected HDF5 structure:
        /fullmix/freqs: (n_frames, n_sinusoids) - frequencies in Hz
        /fullmix/amps: (n_frames, n_sinusoids) - amplitudes
        /fullmix/phases: (n_frames, n_sinusoids) - phases in radians

    Returns: (n_frames, max_sinusoids, 3) where 3 = [freq, amp, phase]
    """
    print(f"Loading sinusoids from HDF5: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
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
        else:
            # List available keys for debugging
            print(f"Available keys in HDF5: {list(f.keys())}")
            raise ValueError("Could not find sinusoid data in HDF5. Expected 'fullmix' group or 'freqs' dataset.")

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

    return sinusoids


def extract_sinusoids_simple(audio, sr=44100, n_fft=1024, hop_length=512, max_sinusoids=2000):
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


def sinusoids_to_audio(sinusoids, sr=44100, hop_length=512):
    """
    Reconstruct audio from sinusoids using overlap-add.
    sinusoids: (n_frames, max_sinusoids, 3) where 3 = [freq, amp, phase]
    """
    n_frames = sinusoids.shape[0]
    audio_len = n_frames * hop_length + 1024  # Extra for final frame
    audio = np.zeros(audio_len, dtype=np.float32)

    for frame_idx in range(n_frames):
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
        sinusoids = load_sinusoids_from_h5(args.h5_file, max_sinusoids=args.max_sinusoids)
        sr = 44100  # Assume 44.1kHz
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

    # Run model
    print("\nSeparating stems...")
    with torch.no_grad():
        # Process in chunks to avoid OOM
        chunk_size = 100  # frames per chunk
        n_frames = sinusoids_tensor.shape[1]
        all_predictions = []

        for i in tqdm(range(0, n_frames, chunk_size)):
            chunk = sinusoids_tensor[:, i:i+chunk_size]
            pred = model(chunk)  # (1, n_stems, chunk_frames, max_sines, 3)
            all_predictions.append(pred.cpu())

        # Concatenate predictions
        predictions = torch.cat(all_predictions, dim=2)  # (1, n_stems, n_frames, max_sines, 3)

    predictions = predictions[0].numpy()  # (n_stems, n_frames, max_sines, 3)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct and save each stem
    print("\nReconstructing audio...")
    for stem_idx, stem_name in enumerate(stem_names):
        print(f"  {stem_name}...")
        stem_sinusoids = predictions[stem_idx]  # (n_frames, max_sines, 3)

        # Reconstruct audio
        stem_audio = sinusoids_to_audio(stem_sinusoids, sr=sr)

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
