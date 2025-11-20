#!/usr/bin/env python3
"""
Train Model 1: Stem Separator using raw sinusoidal chunks.

Takes fullmix sinusoids and separates them into stem sinusoids.
Preserves full precision - no FFT, no quantization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


class SinusoidChunkDataset(Dataset):
    """Dataset that loads chunked sinusoidal data"""

    def __init__(self, data_dirs, stem_names=['vocals', 'guitar', 'bass', 'drums'], max_sinusoids=10000):
        self.stem_names = stem_names
        self.max_sinusoids = max_sinusoids
        self.chunks = []

        print("Loading chunked data...")
        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            chunks_dir = data_dir / 'chunks'

            if not chunks_dir.exists():
                print(f"  Skipping {data_dir.name}: no chunks/ directory")
                continue

            print(f"  Loading {data_dir.name}...")

            # Find all fullmix chunks
            fullmix_chunks = sorted(chunks_dir.glob('fullmix_chunk_*.npz'))

            for fullmix_path in fullmix_chunks:
                chunk_idx = fullmix_path.stem.split('_')[-1]

                # Load corresponding stem chunks
                stem_chunks = {}
                all_found = True

                for stem_name in stem_names:
                    if stem_name == 'drums':
                        # Combine all drum chunks
                        drum_sinusoids = []
                        for i in range(1, 5):
                            drum_path = chunks_dir / f'drums_{i}_chunk_{chunk_idx}.npz'
                            if drum_path.exists():
                                drum_data = np.load(drum_path)
                                drum_sinusoids.append({
                                    'frequencies': drum_data['frequencies'],
                                    'amplitudes': drum_data['amplitudes'],
                                    'phases': drum_data['phases'],
                                    'frame_indices': drum_data['frame_indices']
                                })

                        if len(drum_sinusoids) == 0:
                            all_found = False
                            break

                        # Combine all drum sinusoids
                        stem_chunks['drums'] = {
                            'frequencies': np.concatenate([d['frequencies'] for d in drum_sinusoids]),
                            'amplitudes': np.concatenate([d['amplitudes'] for d in drum_sinusoids]),
                            'phases': np.concatenate([d['phases'] for d in drum_sinusoids]),
                            'frame_indices': np.concatenate([d['frame_indices'] for d in drum_sinusoids])
                        }
                    else:
                        stem_path = chunks_dir / f'{stem_name}_chunk_{chunk_idx}.npz'
                        if not stem_path.exists():
                            all_found = False
                            break

                        stem_data = np.load(stem_path)
                        stem_chunks[stem_name] = {
                            'frequencies': stem_data['frequencies'],
                            'amplitudes': stem_data['amplitudes'],
                            'phases': stem_data['phases'],
                            'frame_indices': stem_data['frame_indices']
                        }

                if not all_found:
                    continue

                # Load fullmix
                fullmix_data = np.load(fullmix_path)

                self.chunks.append({
                    'fullmix': {
                        'frequencies': fullmix_data['frequencies'],
                        'amplitudes': fullmix_data['amplitudes'],
                        'phases': fullmix_data['phases'],
                        'frame_indices': fullmix_data['frame_indices']
                    },
                    'stems': stem_chunks,
                    'song': data_dir.name
                })

        print(f"Total chunks loaded: {len(self.chunks)}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        """Get a training chunk"""
        chunk = self.chunks[idx]

        # Get fullmix sinusoids
        fullmix = chunk['fullmix']
        n_fullmix = len(fullmix['frequencies'])

        # Pad or truncate to max_sinusoids
        if n_fullmix > self.max_sinusoids:
            # Randomly sample
            indices = np.random.choice(n_fullmix, self.max_sinusoids, replace=False)
        else:
            # Pad with zeros
            indices = np.arange(n_fullmix)

        # Create input features [max_sinusoids, 4] (freq, amp, phase, frame)
        input_features = np.zeros((self.max_sinusoids, 4), dtype=np.float32)
        input_features[:len(indices), 0] = fullmix['frequencies'][indices]
        input_features[:len(indices), 1] = fullmix['amplitudes'][indices]
        input_features[:len(indices), 2] = fullmix['phases'][indices]
        input_features[:len(indices), 3] = fullmix['frame_indices'][indices]

        # Create mask for valid sinusoids
        mask = np.zeros(self.max_sinusoids, dtype=np.float32)
        mask[:len(indices)] = 1.0

        # Create targets: for each fullmix sinusoid, which stem does it belong to?
        # We'll match based on nearest frequency/frame
        stem_labels = np.zeros(self.max_sinusoids, dtype=np.int64)

        for i, fullmix_idx in enumerate(indices):
            fullmix_freq = fullmix['frequencies'][fullmix_idx]
            fullmix_frame = fullmix['frame_indices'][fullmix_idx]

            # Find closest match in each stem
            best_stem = 0
            best_distance = float('inf')

            for stem_idx, stem_name in enumerate(self.stem_names):
                stem = chunk['stems'][stem_name]
                if len(stem['frequencies']) == 0:
                    continue

                # Find nearest sinusoid in this stem
                freq_diff = np.abs(stem['frequencies'] - fullmix_freq)
                frame_diff = np.abs(stem['frame_indices'] - fullmix_frame)
                distance = freq_diff + frame_diff * 0.1  # Weight frame less

                min_dist = np.min(distance)
                if min_dist < best_distance:
                    best_distance = min_dist
                    best_stem = stem_idx

            stem_labels[i] = best_stem

        return {
            'input': torch.from_numpy(input_features),  # [max_sinusoids, 4]
            'mask': torch.from_numpy(mask),  # [max_sinusoids]
            'labels': torch.from_numpy(stem_labels),  # [max_sinusoids]
        }


class TransformerStemSeparator(nn.Module):
    """Transformer-based stem separator for sinusoidal data"""

    def __init__(self, n_stems=4, d_model=256, nhead=8, num_layers=6, max_sinusoids=10000):
        super().__init__()

        self.n_stems = n_stems
        self.max_sinusoids = max_sinusoids

        # Input projection: [freq, amp, phase, frame] -> d_model
        self.input_proj = nn.Linear(4, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_sinusoids, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: classify each sinusoid to a stem
        self.stem_classifier = nn.Linear(d_model, n_stems)

    def forward(self, x, mask):
        """
        Args:
            x: [batch, max_sinusoids, 4] input features
            mask: [batch, max_sinusoids] validity mask
        Returns:
            stem_logits: [batch, max_sinusoids, n_stems]
        """
        # Project input
        x = self.input_proj(x)  # [batch, max_sinusoids, d_model]

        # Add positional encoding
        x = x + self.pos_encoding

        # Create attention mask (True = ignore)
        attn_mask = (mask == 0)  # [batch, max_sinusoids]

        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Classify
        stem_logits = self.stem_classifier(x)  # [batch, max_sinusoids, n_stems]

        return stem_logits


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        inputs = batch['input'].to(device)  # [batch, max_sinusoids, 4]
        mask = batch['mask'].to(device)  # [batch, max_sinusoids]
        labels = batch['labels'].to(device)  # [batch, max_sinusoids]

        # Forward
        logits = model(inputs, mask)  # [batch, max_sinusoids, n_stems]

        # Loss: only on valid sinusoids
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, model.n_stems),
            labels.reshape(-1),
            reduction='none'
        )
        loss = (loss * mask.reshape(-1)).sum() / mask.sum()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct = ((preds == labels).float() * mask).sum()
            total_correct += correct.item()
            total_samples += mask.sum().item()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct / mask.sum():.4f}'})

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Train stem separator on sinusoidal chunks')
    parser.add_argument('--data-dirs', nargs='+', required=True, help='Song directories')
    parser.add_argument('--stem-names', nargs='+', default=['vocals', 'guitar', 'bass', 'drums'])
    parser.add_argument('--max-sinusoids', type=int, default=10000, help='Max sinusoids per chunk')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("=" * 60)
    print("Model 1: Stem Separator Training (Sinusoidal)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Stems: {args.stem_names}")
    print(f"Max sinusoids per chunk: {args.max_sinusoids}")
    print()

    # Create dataset
    dataset = SinusoidChunkDataset(
        args.data_dirs,
        stem_names=args.stem_names,
        max_sinusoids=args.max_sinusoids
    )

    if len(dataset) == 0:
        print("ERROR: No valid chunks found!")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )

    # Create model
    model = TransformerStemSeparator(
        n_stems=len(args.stem_names),
        d_model=256,
        nhead=8,
        num_layers=6,
        max_sinusoids=args.max_sinusoids
    ).to(args.device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\nStarting training...\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        loss, acc = train_epoch(model, dataloader, optimizer, args.device)

        print(f"  Loss: {loss:.4f}, Accuracy: {acc:.4f}\n")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'stem_separator_sinusoids_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'accuracy': acc,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}\n")

    # Save final model
    final_path = 'stem_separator_sinusoids_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")


if __name__ == '__main__':
    main()
