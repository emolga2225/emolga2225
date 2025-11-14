#!/usr/bin/env python3
"""
Fast frame-level stem separator training using preprocessed data.

This version loads from preprocessed .npy files instead of HDF5 for much faster training.
Run preprocess_training_data.py first to create the preprocessed dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


class PreprocessedStemDataset(Dataset):
    """Fast dataset using memory-mapped numpy arrays"""

    def __init__(self, data_dir='preprocessed_data'):
        """
        Initialize dataset from preprocessed data.

        Args:
            data_dir: Directory containing preprocessed .npy files
        """
        self.data_dir = Path(data_dir)

        # Load metadata
        with open(self.data_dir / 'metadata.json') as f:
            self.metadata = json.load(f)

        self.n_windows = self.metadata['n_windows']
        self.window_size = self.metadata['window_size']
        self.n_stems = self.metadata['n_stems']

        # Memory-map the arrays for fast access
        self.features = np.load(
            self.data_dir / 'features.npy',
            mmap_mode='r'
        )
        self.labels = np.load(
            self.data_dir / 'labels.npy',
            mmap_mode='r'
        )
        self.masks = np.load(
            self.data_dir / 'masks.npy',
            mmap_mode='r'
        )
        self.bands = np.load(
            self.data_dir / 'bands.npy',
            mmap_mode='r'
        )

        print(f"Loaded preprocessed dataset: {self.n_windows:,} windows")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        """Get a single training sample"""
        return {
            'features': torch.from_numpy(self.features[idx].copy()),
            'mask': torch.from_numpy(self.masks[idx].copy()),
            'band': torch.tensor([self.bands[idx]], dtype=torch.long),
            'labels': torch.from_numpy(self.labels[idx].copy())
        }


class FramewiseStemClassifier(nn.Module):
    """Frame-level stem classifier using transformer"""

    def __init__(self, n_stems, d_model=128, nhead=8, num_layers=4, window_size=32):
        super().__init__()

        self.n_stems = n_stems + 1  # +1 for unknown class
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(4, d_model)  # [freq, log_amp, cos_phase, sin_phase]

        # Band embedding (32 bands)
        self.band_embedding = nn.Embedding(32, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Frame-level classifier
        self.classifier = nn.Linear(d_model, self.n_stems)

    def forward(self, features, mask, band):
        """
        Args:
            features: [batch, window_size, 4] input features
            mask: [batch, window_size] validity mask
            band: [batch, 1] band index

        Returns:
            logits: [batch, window_size, n_stems] frame-level predictions
        """
        batch_size, window_size, _ = features.shape

        # Project input features
        x = self.input_proj(features)  # [batch, window_size, d_model]

        # Add band embedding
        band_emb = self.band_embedding(band.squeeze(-1))  # [batch, d_model]
        x = x + band_emb.unsqueeze(1)  # Broadcast to all frames

        # Add positional encoding
        x = x + self.pos_encoding[:, :window_size, :]

        # Create attention mask (True = ignore)
        attn_mask = (mask == 0)  # [batch, window_size]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Frame-level classification
        logits = self.classifier(x)  # [batch, window_size, n_stems]

        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)
        band = batch['band'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(features, mask, band)

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = labels.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Compute loss only on valid frames
        loss = criterion(logits_flat, labels_flat)
        loss = (loss * mask_flat).sum() / mask_flat.sum()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Stats (only on valid frames)
        total_loss += loss.item()
        _, predicted = logits_flat.max(1)

        valid_mask = mask_flat > 0
        if valid_mask.sum() > 0:
            correct += predicted[valid_mask].eq(labels_flat[valid_mask]).sum().item()
            total += valid_mask.sum().item()

        # Update progress bar
        if total > 0:
            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0

    return avg_loss, accuracy


def main():
    """Main training loop"""

    # Configuration
    config = {
        'data_dir': 'preprocessed_data',
        'batch_size': 256,  # Increased batch size for faster training
        'n_stems': 5,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'window_size': 32,
        'learning_rate': 1e-3,
        'n_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': 'checkpoints',
        'save_every': 10
    }

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)

    # Create dataset
    print("Loading dataset...")
    dataset = PreprocessedStemDataset(config['data_dir'])

    # Create dataloader (can use multiple workers with preprocessed data!)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,  # Use multiple workers for faster loading
        pin_memory=True if config['device'] == 'cuda' else False,
        persistent_workers=True
    )

    # Create model
    print("\nCreating model...")
    model = FramewiseStemClassifier(
        n_stems=config['n_stems'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        window_size=config['window_size']
    ).to(config['device'])

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss for masking

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, config['n_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['n_epochs']}")

        train_loss, train_acc = train_epoch(
            model, dataloader, optimizer, criterion, config['device']
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Save checkpoint
        if epoch % config['save_every'] == 0:
            checkpoint_path = Path(config['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
