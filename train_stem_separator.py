import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

class SinusoidalTrackDataset(Dataset):
    """Dataset for sinusoidal track-to-stem mapping"""

    def __init__(self, mix_h5_files, stem_h5_files, stem_names, max_seq_len=512):
        """
        Args:
            mix_h5_files: List of HDF5 files containing mix sinusoids
            stem_h5_files: List of lists, where each inner list contains HDF5 files for each stem
            stem_names: List of stem names (e.g., ['vocals', 'drums', 'bass', 'other'])
            max_seq_len: Maximum sequence length for tracks
        """
        self.mix_h5_files = mix_h5_files
        self.stem_h5_files = stem_h5_files
        self.stem_names = stem_names
        self.n_stems = len(stem_names)
        self.max_seq_len = max_seq_len

        # Build index of all tracks
        self.tracks = []
        self._build_index()

    def _build_index(self):
        """Build index of all tracks from mix files"""
        print("Building dataset index...")

        for mix_file in tqdm(self.mix_h5_files, desc="Indexing mix files"):
            with h5py.File(mix_file, 'r') as f:
                for ch_idx in range(f.attrs['ch']):
                    grp_name = f'c{ch_idx}'
                    if grp_name not in f:
                        continue

                    grp = f[grp_name]
                    track_lens = grp['len'][:]

                    for track_idx in range(len(track_lens)):
                        self.tracks.append({
                            'mix_file': mix_file,
                            'channel': ch_idx,
                            'track_idx': track_idx
                        })

        print(f"Found {len(self.tracks)} tracks")

    def __len__(self):
        return len(self.tracks)

    def _load_track_features(self, h5_file, channel, track_idx):
        """Load features for a single track"""
        with h5py.File(h5_file, 'r') as f:
            grp = f[f'c{channel}']

            track_lens = grp['len'][:]
            all_freqs = grp['f'][:]
            all_amps = grp['a'][:]
            all_phases = grp['p'][:]
            track_bands = grp['b'][:]

            # Find offset for this track
            offset = sum(track_lens[:track_idx])
            n = track_lens[track_idx]

            # Extract track data
            freqs = all_freqs[offset:offset+n]
            amps = all_amps[offset:offset+n]
            phases = all_phases[offset:offset+n]
            band = track_bands[track_idx]

            return freqs, amps, phases, band, n

    def _create_features(self, freqs, amps, phases, band, seq_len):
        """Create feature vector from track parameters"""
        # Pad or truncate to max_seq_len
        if seq_len > self.max_seq_len:
            freqs = freqs[:self.max_seq_len]
            amps = amps[:self.max_seq_len]
            phases = phases[:self.max_seq_len]
            seq_len = self.max_seq_len

        # Create feature matrix [seq_len, n_features]
        features = np.zeros((self.max_seq_len, 4), dtype=np.float32)

        # Fill features
        features[:seq_len, 0] = freqs / 96000.0  # Normalized frequency
        features[:seq_len, 1] = np.log1p(amps)   # Log amplitude
        features[:seq_len, 2] = np.cos(phases)   # Phase cos
        features[:seq_len, 3] = np.sin(phases)   # Phase sin

        # Create mask for valid timesteps
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        # Add band as a scalar feature
        band_feature = np.array([band], dtype=np.float32)

        return features, mask, band_feature

    def __getitem__(self, idx):
        """Get a single training example"""
        track_info = self.tracks[idx]

        # Load mix track
        mix_freqs, mix_amps, mix_phases, mix_band, mix_len = self._load_track_features(
            track_info['mix_file'], track_info['channel'], track_info['track_idx']
        )

        # Create features
        features, mask, band_feature = self._create_features(
            mix_freqs, mix_amps, mix_phases, mix_band, mix_len
        )

        # For now, create dummy label (we'll implement stem matching later)
        # Label will be the stem index this track belongs to
        label = np.random.randint(0, self.n_stems)  # Placeholder

        return {
            'features': torch.from_numpy(features),
            'mask': torch.from_numpy(mask),
            'band': torch.from_numpy(band_feature),
            'label': torch.tensor(label, dtype=torch.long)
        }


class StemClassifierTransformer(nn.Module):
    """Transformer model for classifying sinusoidal tracks to stems"""

    def __init__(self, n_stems, d_model=128, nhead=8, num_layers=4, max_seq_len=512):
        super().__init__()

        self.n_stems = n_stems
        self.d_model = d_model

        # Input projection (4 features: freq, log_amp, cos_phase, sin_phase)
        self.input_proj = nn.Linear(4, d_model)

        # Band embedding
        self.band_embed = nn.Embedding(32, d_model)  # Support up to 32 bands

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_stems)
        )

    def forward(self, features, mask, band):
        """
        Args:
            features: [batch, seq_len, 4]
            mask: [batch, seq_len]
            band: [batch, 1]
        """
        batch_size, seq_len, _ = features.shape

        # Project input features
        x = self.input_proj(features)  # [batch, seq_len, d_model]

        # Add band embedding (broadcasted across sequence)
        band_emb = self.band_embed(band.squeeze(-1))  # [batch, d_model]
        x = x + band_emb.unsqueeze(1)

        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)

        # Create attention mask (True = masked position)
        attn_mask = (mask == 0)  # [batch, seq_len]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Global average pooling (masked)
        mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        x_masked = x * mask_expanded
        x_sum = x_masked.sum(dim=1)
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1)
        x_avg = x_sum / mask_sum

        # Classification
        logits = self.classifier(x_avg)

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
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(features, mask, band)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(dataloader), 100. * correct / total


def main():
    # Configuration
    config = {
        'stem_names': ['vocals', 'drums', 'bass', 'other'],
        'max_seq_len': 512,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("Stem Separator Training")
    print(f"Device: {config['device']}")
    print(f"Stems: {config['stem_names']}")

    # TODO: Replace with actual file paths
    mix_h5_files = ['mix_tracks.h5']  # List of mix HDF5 files
    stem_h5_files = [
        ['vocals_tracks.h5'],  # Vocals stem files
        ['drums_tracks.h5'],   # Drums stem files
        ['bass_tracks.h5'],    # Bass stem files
        ['other_tracks.h5']    # Other stem files
    ]

    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = SinusoidalTrackDataset(
        mix_h5_files=mix_h5_files,
        stem_h5_files=stem_h5_files,
        stem_names=config['stem_names'],
        max_seq_len=config['max_seq_len']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    model = StemClassifierTransformer(
        n_stems=len(config['stem_names']),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len']
    ).to(config['device'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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
            torch.save(checkpoint, f'stem_separator_epoch_{epoch+1}.pt')
            print(f"Saved checkpoint: stem_separator_epoch_{epoch+1}.pt")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'stem_separator_final.pt')
    print("\nTraining complete! Saved final model: stem_separator_final.pt")


if __name__ == "__main__":
    main()
