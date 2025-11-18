import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

class FramewiseSinusoidalDataset(Dataset):
    """Dataset for frame-level sinusoidal track-to-stem mapping with multi-worker support"""

    def __init__(self, mix_h5_file, frame_labels_file, n_stems, window_size=32, augment=True):
        """
        Args:
            mix_h5_file: HDF5 file containing mix sinusoids
            frame_labels_file: JSON file with frame-level labels
            n_stems: Number of stem classes
            window_size: Number of frames in each training window
            augment: Whether to apply data augmentation to prevent overfitting
        """
        self.mix_h5_file = mix_h5_file
        self.n_stems = n_stems
        self.window_size = window_size
        self.augment = augment
        self._h5_file = None  # Per-worker file handle

        # Load frame labels
        with open(frame_labels_file, 'r') as f:
            self.frame_labels = json.load(f)

        # Build index of all training windows
        self.windows = []
        self._build_index()

    def _get_h5_file(self):
        """Get HDF5 file handle for current worker"""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.mix_h5_file, 'r')
        return self._h5_file

    def _build_index(self):
        """Build index of all sliding windows"""
        print("Building dataset index...")

        with h5py.File(self.mix_h5_file, 'r') as f:
            for ch_idx in range(f.attrs['ch']):
                grp_name = f'c{ch_idx}'
                if grp_name not in f:
                    continue

                grp = f[grp_name]
                track_lens = grp['len'][:]
                track_ids = grp['id'][:]

                for track_idx, (track_id, track_len) in enumerate(zip(track_ids, track_lens)):
                    track_id_str = str(track_id)

                    if track_id_str not in self.frame_labels:
                        continue

                    labels = self.frame_labels[track_id_str]['labels']

                    # Skip tracks shorter than window size
                    if track_len < self.window_size:
                        continue

                    # Create sliding windows
                    for start_frame in range(0, track_len - self.window_size + 1, self.window_size // 2):
                        end_frame = start_frame + self.window_size

                        # Get labels for this window
                        window_labels = labels[start_frame:end_frame]

                        # Ensure window is exactly window_size
                        if len(window_labels) != self.window_size:
                            continue

                        # Skip windows with too many unmatched frames
                        unmatched_count = sum(1 for l in window_labels if l == -1)
                        if unmatched_count > self.window_size * 0.5:  # Skip if >50% unmatched
                            continue

                        self.windows.append({
                            'channel': ch_idx,
                            'track_idx': track_idx,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'labels': window_labels
                        })

        print(f"Built {len(self.windows)} training windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """Get a single training window"""
        window_info = self.windows[idx]

        # Use per-worker file handle for multi-worker DataLoader
        f = self._get_h5_file()
        grp = f[f'c{window_info["channel"]}']

        track_lens = grp['len'][:]
        track_bands = grp['b'][:]

        all_freqs = grp['f'][:]
        all_amps = grp['a'][:]
        all_phases = grp['p'][:]

        # Find offset for this track
        offset = sum(track_lens[:window_info['track_idx']])

        # Extract window data
        start = offset + window_info['start_frame']
        end = offset + window_info['end_frame']

        freqs = all_freqs[start:end]
        amps = all_amps[start:end]
        phases = all_phases[start:end]
        band = track_bands[window_info['track_idx']]

        # Apply data augmentation to prevent overfitting
        if self.augment:
            # Add small random perturbations to prevent memorization
            # Frequency jitter: ±2%
            freq_jitter = np.random.uniform(0.98, 1.02, size=len(freqs))
            freqs = freqs * freq_jitter

            # Amplitude jitter: ±10% (in log space for better distribution)
            amp_jitter = np.random.uniform(-0.1, 0.1, size=len(amps))
            amps = amps * np.exp(amp_jitter)

            # Phase jitter: ±π/8
            phase_jitter = np.random.uniform(-np.pi/8, np.pi/8, size=len(phases))
            phases = phases + phase_jitter

            # Random amplitude scaling of entire window: 0.8-1.2x
            global_amp_scale = np.random.uniform(0.8, 1.2)
            amps = amps * global_amp_scale

        # Create feature matrix [window_size, n_features]
        features = np.zeros((self.window_size, 4), dtype=np.float32)

        actual_len = len(freqs)
        features[:actual_len, 0] = freqs / 96000.0  # Normalized frequency
        features[:actual_len, 1] = np.log1p(amps)   # Log amplitude
        features[:actual_len, 2] = np.cos(phases)   # Phase cos
        features[:actual_len, 3] = np.sin(phases)   # Phase sin

        # Add Gaussian noise to features during training
        if self.augment:
            noise = np.random.normal(0, 0.01, features.shape).astype(np.float32)
            features = features + noise

        # Create mask for valid frames
        mask = np.zeros(self.window_size, dtype=np.float32)
        mask[:actual_len] = 1.0

        # Labels (replace -1 with n_stems as "unknown" class)
        labels = np.array(window_info['labels'], dtype=np.int64)
        labels[labels == -1] = self.n_stems  # Unknown class

        return {
            'features': torch.from_numpy(features.copy()),
            'mask': torch.from_numpy(mask.copy()),
            'band': torch.tensor([band], dtype=torch.long),
            'labels': torch.from_numpy(labels.copy())  # [window_size] frame-level labels
        }


class FramewiseStemClassifier(nn.Module):
    """Frame-level stem classifier using transformer"""

    def __init__(self, n_stems, d_model=128, nhead=8, num_layers=4, window_size=32):
        super().__init__()

        self.n_stems = n_stems + 1  # +1 for unknown class
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(4, d_model)

        # Band embedding
        self.band_embed = nn.Embedding(32, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(window_size, d_model))

        # Transformer encoder with increased dropout to prevent overfitting
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.3,  # Increased from 0.1 to prevent memorization
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Frame-level classification head with higher dropout
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased from 0.1
            nn.Linear(d_model, self.n_stems)
        )

    def forward(self, features, mask, band):
        """
        Args:
            features: [batch, window_size, 4]
            mask: [batch, window_size]
            band: [batch, 1]

        Returns:
            logits: [batch, window_size, n_stems]
        """
        batch_size, window_size, _ = features.shape

        # Project input features
        x = self.input_proj(features)  # [batch, window_size, d_model]

        # Add band embedding
        band_emb = self.band_embed(band.squeeze(-1))  # [batch, d_model]
        x = x + band_emb.unsqueeze(1)

        # Add positional encoding
        x = x + self.pos_encoding[:window_size].unsqueeze(0)

        # Create attention mask
        attn_mask = (mask == 0)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Frame-level classification
        logits = self.classifier(x)  # [batch, window_size, n_stems]

        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing to prevent overconfidence"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] logits
            target: [N] class indices
        """
        pred = pred.log_softmax(dim=-1)
        n_class = pred.size(-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)
        band = batch['band'].to(device)
        labels = batch['labels'].to(device)  # [batch, window_size]

        optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(features, mask, band)  # [batch, window_size, n_stems]

                # Flatten for loss computation
                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = labels.reshape(-1)
                mask_flat = mask.reshape(-1)

                # Filter to only valid frames
                valid_mask = mask_flat > 0
                if valid_mask.sum() == 0:
                    continue

                logits_valid = logits_flat[valid_mask]
                labels_valid = labels_flat[valid_mask]

                # Compute loss with label smoothing
                loss = criterion(logits_valid, labels_valid)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without AMP
            logits = model(features, mask, band)

            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            mask_flat = mask.reshape(-1)

            valid_mask = mask_flat > 0
            if valid_mask.sum() == 0:
                continue

            logits_valid = logits_flat[valid_mask]
            labels_valid = labels_flat[valid_mask]

            loss = criterion(logits_valid, labels_valid)

            loss.backward()
            optimizer.step()

        # Stats (only on valid frames)
        total_loss += loss.item()
        _, predicted = logits_valid.max(1)
        correct += predicted.eq(labels_valid).sum().item()
        total += labels_valid.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(dataloader), 100. * correct / total if total > 0 else 0


def main():
    # Configuration with anti-overfitting settings
    config = {
        'stem_names': ['vocals', 'guitar', 'bass', 'drums'],
        'window_size': 32,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'batch_size': 512,  # Large enough to saturate GPU, won't OOM
        'learning_rate': 1e-4,
        'weight_decay': 0.01,  # L2 regularization to prevent overfitting
        'label_smoothing': 0.1,  # Prevent overconfidence
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'augment': True,  # Enable data augmentation to prevent overfitting
        'use_amp': True  # Mixed precision training for 2-3x speedup
    }

    print("Frame-wise Stem Separator Training")
    print(f"Device: {config['device']}")
    print(f"Stems: {config['stem_names']}")

    # File paths
    mix_h5_file = 'fullmix_tracks.h5'
    frame_labels_file = 'frame_labels.json'

    # Create dataset and dataloader
    print("\nCreating dataset...")
    print(f"Data augmentation: {'enabled' if config['augment'] else 'disabled'}")
    dataset = FramewiseSinusoidalDataset(
        mix_h5_file=mix_h5_file,
        frame_labels_file=frame_labels_file,
        n_stems=len(config['stem_names']),
        window_size=config['window_size'],
        augment=config['augment']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=6,  # Balance between throughput and HDF5 contention
        pin_memory=True if config['device'] == 'cuda' else False,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=3  # Prefetch batches to keep GPU fed
    )

    # Create model
    print("\nCreating model...")
    model = FramewiseStemClassifier(
        n_stems=len(config['stem_names']),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        window_size=config['window_size']
    ).to(config['device'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss with label smoothing to prevent overconfidence
    print(f"\nOptimization settings:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Mixed precision (AMP): {'enabled' if config.get('use_amp', False) else 'disabled'}")
    print(f"\nRegularization settings:")
    print(f"  Label smoothing: {config['label_smoothing']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Dropout: 0.3")

    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])

    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Mixed precision training scaler
    scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', False) and config['device'] == 'cuda' else None

    # Training loop
    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, dataloader, optimizer, criterion, config['device'], scaler)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'train_loss': train_loss,
                'train_acc': train_acc
            }
            torch.save(checkpoint, f'stem_separator_framewise_epoch_{epoch+1}.pt')
            print(f"Saved checkpoint: stem_separator_framewise_epoch_{epoch+1}.pt")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'stem_separator_framewise_final.pt')
    print("\nTraining complete! Saved final model: stem_separator_framewise_final.pt")


if __name__ == "__main__":
    main()
