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
    """Dataset for frame-level sinusoidal track-to-stem mapping"""

    def __init__(self, mix_h5_file, frame_labels_file, n_stems, window_size=32):
        """
        Args:
            mix_h5_file: HDF5 file containing mix sinusoids
            frame_labels_file: JSON file with frame-level labels
            n_stems: Number of stem classes
            window_size: Number of frames in each training window
        """
        self.mix_h5_file = mix_h5_file
        self.n_stems = n_stems
        self.window_size = window_size

        # Load frame labels
        with open(frame_labels_file, 'r') as f:
            self.frame_labels = json.load(f)

        # Build index of all training windows
        self.windows = []
        self._build_index()

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

        with h5py.File(self.mix_h5_file, 'r') as f:
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

        # Create feature matrix [window_size, n_features]
        features = np.zeros((self.window_size, 4), dtype=np.float32)

        actual_len = len(freqs)
        features[:actual_len, 0] = freqs / 96000.0  # Normalized frequency
        features[:actual_len, 1] = np.log1p(amps)   # Log amplitude
        features[:actual_len, 2] = np.cos(phases)   # Phase cos
        features[:actual_len, 3] = np.sin(phases)   # Phase sin

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

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Frame-level classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        labels = batch['labels'].to(device)  # [batch, window_size]

        optimizer.zero_grad()

        # Forward pass
        logits = model(features, mask, band)  # [batch, window_size, n_stems]

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, logits.size(-1))  # [batch*window_size, n_stems]
        labels_flat = labels.reshape(-1)  # [batch*window_size]
        mask_flat = mask.reshape(-1)  # [batch*window_size]

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

        if total > 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    return total_loss / len(dataloader), 100. * correct / total if total > 0 else 0


def main():
    # Configuration
    config = {
        'stem_names': ['vocals', 'guitar', 'bass', 'drums'],
        'window_size': 32,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("Frame-wise Stem Separator Training")
    print(f"Device: {config['device']}")
    print(f"Stems: {config['stem_names']}")

    # File paths
    mix_h5_file = 'fullmix_tracks.h5'
    frame_labels_file = 'frame_labels.json'

    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = FramewiseSinusoidalDataset(
        mix_h5_file=mix_h5_file,
        frame_labels_file=frame_labels_file,
        n_stems=len(config['stem_names']),
        window_size=config['window_size']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for HDF5 compatibility
        pin_memory=True if config['device'] == 'cuda' else False
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

    # Loss and optimizer (use reduction='none' for per-frame loss)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Training loop
    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, dataloader, optimizer, criterion, config['device'])

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
