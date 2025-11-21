#!/usr/bin/env python3
"""
Generative Stem Separator - Sinusoidal Model

This is the CORRECT architecture for precise stem separation:

Input: fullmix sinusoids [freq, amp, phase, frame]
Output: 4 sets of stem sinusoids (vocals, guitar, bass, drums)
Guidance: ground truth stem sinusoids
Architecture: Transformer-based generative model

Preserves full synchrosqueeze precision - no FFT, no spectrograms.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse


class SinusoidalStemDataset(Dataset):
    """Dataset that loads fullmix and stem sinusoids from HDF5 files"""

    def __init__(self, data_dirs, stem_names, max_sinusoids=2000,
                 max_output_per_stem=500, chunk_duration=4.0, hop_length=512):
        """
        Args:
            data_dirs: List of directories with .h5 files
            stem_names: List of stem names
            max_sinusoids: Max input sinusoids from fullmix
            max_output_per_stem: Max output sinusoids per stem
            chunk_duration: Duration of chunks in seconds
            hop_length: Hop length for frame calculation
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dirs = [Path(d) for d in data_dirs]
        self.stem_names = stem_names
        self.max_sinusoids = max_sinusoids
        self.max_output_per_stem = max_output_per_stem
        self.chunk_duration = chunk_duration
        self.hop_length = hop_length
        self.chunk_frames = int(chunk_duration * 44100 / hop_length)

        # Load metadata from all songs
        print(f"Loading sinusoidal data from {len(self.data_dirs)} song(s)...")
        self.chunks = []

        for data_dir in self.data_dirs:
            print(f"\n  Loading from {data_dir.name}...")

            # Check for fullmix
            fullmix_h5 = data_dir / 'fullmix_tracks.h5'
            if not fullmix_h5.exists():
                fullmix_h5 = data_dir / 'fullmix.h5'
                if not fullmix_h5.exists():
                    print(f"    Warning: No fullmix .h5 found, skipping")
                    continue

            # Load fullmix to determine chunk count
            with h5py.File(fullmix_h5, 'r') as f:
                # Get max frame from first channel
                if 'c0' in f and 'i' in f['c0']:
                    max_frame = f['c0']['i'][:].max() if len(f['c0']['i']) > 0 else 0
                else:
                    print(f"    Warning: No sinusoids in fullmix, skipping")
                    continue

            n_chunks = max_frame // self.chunk_frames
            if n_chunks == 0:
                n_chunks = 1

            # Store chunk info
            for chunk_idx in range(n_chunks):
                stem_h5_paths = {}
                for stem_name in stem_names:
                    stem_h5 = data_dir / f'{stem_name}.h5'
                    if stem_h5.exists():
                        stem_h5_paths[stem_name] = stem_h5

                self.chunks.append({
                    'fullmix_h5': fullmix_h5,
                    'stem_h5_paths': stem_h5_paths,
                    'chunk_idx': chunk_idx,
                    'dir': data_dir
                })

            print(f"    Loaded {n_chunks} chunks")

        print(f"\nTotal chunks: {len(self.chunks)}")

    def _load_sinusoids_from_chunk(self, h5_path, chunk_idx):
        """Load sinusoids from a specific time chunk"""
        start_frame = chunk_idx * self.chunk_frames
        end_frame = start_frame + self.chunk_frames

        sinusoids = []

        with h5py.File(h5_path, 'r') as f:
            n_channels = f.attrs.get('ch', 1)

            for ch_idx in range(n_channels):
                grp_name = f'c{ch_idx}'
                if grp_name not in f:
                    continue

                grp = f[grp_name]

                # Load track data
                track_lens = grp['len'][:]
                track_starts = grp['s'][:]
                track_ends = grp['e'][:]

                frequencies = grp['f'][:]
                amplitudes = grp['a'][:]
                frame_indices = grp['i'][:]

                # Process tracks that overlap with this chunk
                offset = 0
                for track_idx, track_len in enumerate(track_lens):
                    track_start = track_starts[track_idx]
                    track_end = track_ends[track_idx]

                    # Check overlap with chunk
                    if track_end >= start_frame and track_start < end_frame:
                        track_freqs = frequencies[offset:offset + track_len]
                        track_amps = amplitudes[offset:offset + track_len]
                        track_frames = frame_indices[offset:offset + track_len]

                        # Extract sinusoids in this chunk
                        for freq, amp, frame in zip(track_freqs, track_amps, track_frames):
                            if start_frame <= frame < end_frame:
                                # Normalize frame to chunk-relative
                                local_frame = frame - start_frame
                                sinusoids.append([freq, amp, local_frame])

                    offset += track_len

        return np.array(sinusoids) if len(sinusoids) > 0 else np.zeros((0, 3))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk_info = self.chunks[idx]

        # Load fullmix sinusoids
        fullmix_sines = self._load_sinusoids_from_chunk(
            chunk_info['fullmix_h5'],
            chunk_info['chunk_idx']
        )

        # Limit to max_sinusoids
        if len(fullmix_sines) > self.max_sinusoids:
            # Sample uniformly
            indices = np.linspace(0, len(fullmix_sines) - 1, self.max_sinusoids, dtype=int)
            fullmix_sines = fullmix_sines[indices]

        # Pad if needed
        if len(fullmix_sines) < self.max_sinusoids:
            padding = np.zeros((self.max_sinusoids - len(fullmix_sines), 3))
            fullmix_sines = np.vstack([fullmix_sines, padding])

        # Load stem sinusoids
        stem_sines = []
        for stem_name in self.stem_names:
            if stem_name in chunk_info['stem_h5_paths']:
                sines = self._load_sinusoids_from_chunk(
                    chunk_info['stem_h5_paths'][stem_name],
                    chunk_info['chunk_idx']
                )
            else:
                sines = np.zeros((0, 3))

            # Limit and pad
            if len(sines) > self.max_output_per_stem:
                indices = np.linspace(0, len(sines) - 1, self.max_output_per_stem, dtype=int)
                sines = sines[indices]

            if len(sines) < self.max_output_per_stem:
                padding = np.zeros((self.max_output_per_stem - len(sines), 3))
                sines = np.vstack([sines, padding])

            stem_sines.append(sines)

        stem_sines = np.stack(stem_sines, axis=0)  # [n_stems, max_output_per_stem, 3]

        return {
            'fullmix': torch.from_numpy(fullmix_sines).float(),  # [max_sinusoids, 3]
            'stems': torch.from_numpy(stem_sines).float(),  # [n_stems, max_output_per_stem, 3]
        }


class SinusoidalTransformer(nn.Module):
    """Transformer that generates stem sinusoids from fullmix sinusoids"""

    def __init__(self, n_stems=4, max_input_sinusoids=2000, max_output_per_stem=500,
                 d_model=256, nhead=8, num_layers=6):
        super().__init__()

        self.n_stems = n_stems
        self.max_input_sinusoids = max_input_sinusoids
        self.max_output_per_stem = max_output_per_stem
        self.d_model = d_model

        # Input embedding: [freq, amp, frame] -> d_model
        self.input_embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Positional encoding
        self.pos_encoder = nn.Embedding(max_input_sinusoids, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Stem-specific decoders
        self.stem_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, max_output_per_stem * 3)
            )
            for _ in range(n_stems)
        ])

    def forward(self, x):
        """
        Args:
            x: [batch, max_sinusoids, 3] - fullmix sinusoids [freq, amp, frame]

        Returns:
            stems: [batch, n_stems, max_output_per_stem, 3] - generated stem sinusoids
        """
        batch_size = x.shape[0]

        # Embed input sinusoids
        x_embed = self.input_embed(x)  # [batch, max_sinusoids, d_model]

        # Add positional encoding
        positions = torch.arange(self.max_input_sinusoids, device=x.device)
        pos_embed = self.pos_encoder(positions)  # [max_sinusoids, d_model]
        x_embed = x_embed + pos_embed.unsqueeze(0)

        # Encode with transformer
        encoded = self.transformer_encoder(x_embed)  # [batch, max_sinusoids, d_model]

        # Global pooling to get fixed representation
        pooled = encoded.mean(dim=1)  # [batch, d_model]

        # Generate sinusoids for each stem
        stem_outputs = []
        for decoder in self.stem_decoders:
            stem_sines = decoder(pooled)  # [batch, max_output_per_stem * 3]
            stem_sines = stem_sines.view(batch_size, self.max_output_per_stem, 3)
            stem_outputs.append(stem_sines)

        stems = torch.stack(stem_outputs, dim=1)  # [batch, n_stems, max_output_per_stem, 3]

        # Apply constraints:
        # - freq should be positive
        # - amp should be positive
        # - frame should be in valid range
        stems[:, :, :, 0] = torch.relu(stems[:, :, :, 0])  # freq >= 0
        stems[:, :, :, 1] = torch.relu(stems[:, :, :, 1])  # amp >= 0
        stems[:, :, :, 2] = torch.relu(stems[:, :, :, 2])  # frame >= 0

        return stems


def sinusoidal_loss(pred_stems, target_stems):
    """
    Loss between predicted and target stem sinusoids.
    Uses L1 loss on [freq, amp, frame] values.
    """
    # Simple L1 loss on all parameters
    loss = nn.functional.l1_loss(pred_stems, target_stems)
    return loss


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        fullmix = batch['fullmix'].to(device)  # [batch, max_sinusoids, 3]
        stems = batch['stems'].to(device)  # [batch, n_stems, max_output_per_stem, 3]

        optimizer.zero_grad()

        # Forward pass
        pred_stems = model(fullmix)

        # Compute loss
        loss = sinusoidal_loss(pred_stems, stems)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train generative sinusoidal stem separator')
    parser.add_argument('--data-dirs', nargs='+', required=True,
                       help='Directories containing .h5 sinusoidal data')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--max-input-sinusoids', type=int, default=2000,
                       help='Max input sinusoids from fullmix')
    parser.add_argument('--max-output-per-stem', type=int, default=500,
                       help='Max output sinusoids per stem')
    args = parser.parse_args()

    config = {
        'stem_names': ['vocals', 'guitar', 'bass', 'drums'],
        'data_dirs': args.data_dirs,
        'max_input_sinusoids': args.max_input_sinusoids,
        'max_output_per_stem': args.max_output_per_stem,
        'batch_size': args.batch_size,
        'learning_rate': 1e-4,
        'num_epochs': args.epochs,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6
    }

    print("=" * 60)
    print("Generative Stem Separator - Sinusoidal Model")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Stems: {config['stem_names']}")
    print(f"Max input sinusoids: {config['max_input_sinusoids']}")
    print(f"Max output sinusoids per stem: {config['max_output_per_stem']}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = SinusoidalStemDataset(
        data_dirs=config['data_dirs'],
        stem_names=config['stem_names'],
        max_sinusoids=config['max_input_sinusoids'],
        max_output_per_stem=config['max_output_per_stem']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # HDF5 requires single-threaded
        pin_memory=True if config['device'] == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    model = SinusoidalTransformer(
        n_stems=len(config['stem_names']),
        max_input_sinusoids=config['max_input_sinusoids'],
        max_output_per_stem=config['max_output_per_stem'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(config['device'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss = train_epoch(model, dataloader, optimizer, config['device'])

        print(f"  Train Loss: {train_loss:.6f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'train_loss': train_loss
            }
            checkpoint_path = f'sinusoidal_stem_generator_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'sinusoidal_stem_generator_final.pt')
    print("\nTraining complete! Saved final model: sinusoidal_stem_generator_final.pt")


if __name__ == "__main__":
    main()
