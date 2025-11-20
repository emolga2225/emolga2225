#!/usr/bin/env python3
"""
Train Model 1: Generative Stem Separator

Takes fullmix sinusoids and GENERATES stem sinusoids.
Uses ground truth stem sinusoids as targets during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


class GenerativeStemDataset(Dataset):
    """Dataset with fullmix input and stem outputs"""

    def __init__(self, data_dirs, stem_names=['vocals', 'guitar', 'bass', 'drums'],
                 max_input_sinusoids=2000, max_output_sinusoids=500):
        self.stem_names = stem_names
        self.max_input_sinusoids = max_input_sinusoids
        self.max_output_sinusoids = max_output_sinusoids  # Per stem
        self.chunks = []

        print("Loading data for generative model...")
        for data_dir in data_dirs:
            data_dir = Path(data_dir)

            # Load fullmix chunks
            fullmix_dir = data_dir / 'chunks'
            stems_dir = data_dir / 'labeled_chunks'

            if not fullmix_dir.exists() or not stems_dir.exists():
                print(f"  Skipping {data_dir.name}: missing directories")
                continue

            # Load labeled chunks (contains stem sinusoids)
            labeled_chunks = sorted(stems_dir.glob('chunk_*.npz'))

            # Load fullmix chunks
            fullmix_chunks = sorted(fullmix_dir.glob('fullmix_chunk_*.npz'))

            print(f"  Loading {data_dir.name}: {len(labeled_chunks)} chunks...")

            for labeled_path, fullmix_path in tqdm(zip(labeled_chunks, fullmix_chunks),
                                                   total=len(labeled_chunks),
                                                   desc=f"    Loading {data_dir.name}",
                                                   leave=False):
                # Load fullmix
                fullmix_data = np.load(fullmix_path)

                # Load stems (labeled chunk has all stems combined with labels)
                labeled_data = np.load(labeled_path)

                # Separate by stem label
                stems_separated = {}
                for stem_idx, stem_name in enumerate(stem_names):
                    mask = labeled_data['labels'] == stem_idx
                    stems_separated[stem_name] = {
                        'frequencies': labeled_data['frequencies'][mask],
                        'amplitudes': labeled_data['amplitudes'][mask],
                        'phases': labeled_data['phases'][mask],
                        'frame_indices': labeled_data['frame_indices'][mask]
                    }

                self.chunks.append({
                    'fullmix': {
                        'frequencies': fullmix_data['frequencies'],
                        'amplitudes': fullmix_data['amplitudes'],
                        'phases': fullmix_data['phases'],
                        'frame_indices': fullmix_data['frame_indices']
                    },
                    'stems': stems_separated
                })

        print(f"Total chunks loaded: {len(self.chunks)}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        """Get fullmix input and stem targets"""
        chunk = self.chunks[idx]

        # Prepare fullmix input
        fullmix = chunk['fullmix']
        n_fullmix = len(fullmix['frequencies'])

        # Sample/pad fullmix
        if n_fullmix > self.max_input_sinusoids:
            indices = np.random.choice(n_fullmix, self.max_input_sinusoids, replace=False)
            n_valid_input = self.max_input_sinusoids
        else:
            indices = np.arange(n_fullmix)
            n_valid_input = n_fullmix

        # Input: [max_input_sinusoids, 4]
        input_features = np.zeros((self.max_input_sinusoids, 4), dtype=np.float32)
        input_features[:n_valid_input, 0] = fullmix['frequencies'][indices]
        input_features[:n_valid_input, 1] = fullmix['amplitudes'][indices]
        input_features[:n_valid_input, 2] = fullmix['phases'][indices]
        input_features[:n_valid_input, 3] = fullmix['frame_indices'][indices]

        input_mask = np.zeros(self.max_input_sinusoids, dtype=np.float32)
        input_mask[:n_valid_input] = 1.0

        # Prepare stem targets: [n_stems, max_output_sinusoids, 4]
        stem_targets = np.zeros((len(self.stem_names), self.max_output_sinusoids, 4), dtype=np.float32)
        stem_masks = np.zeros((len(self.stem_names), self.max_output_sinusoids), dtype=np.float32)

        for stem_idx, stem_name in enumerate(self.stem_names):
            stem_data = chunk['stems'][stem_name]
            n_stem = len(stem_data['frequencies'])

            if n_stem > self.max_output_sinusoids:
                stem_indices = np.random.choice(n_stem, self.max_output_sinusoids, replace=False)
                n_valid_stem = self.max_output_sinusoids
            else:
                stem_indices = np.arange(n_stem)
                n_valid_stem = n_stem

            stem_targets[stem_idx, :n_valid_stem, 0] = stem_data['frequencies'][stem_indices]
            stem_targets[stem_idx, :n_valid_stem, 1] = stem_data['amplitudes'][stem_indices]
            stem_targets[stem_idx, :n_valid_stem, 2] = stem_data['phases'][stem_indices]
            stem_targets[stem_idx, :n_valid_stem, 3] = stem_data['frame_indices'][stem_indices]

            stem_masks[stem_idx, :n_valid_stem] = 1.0

        return {
            'input': torch.from_numpy(input_features),  # [max_input, 4]
            'input_mask': torch.from_numpy(input_mask),  # [max_input]
            'targets': torch.from_numpy(stem_targets),  # [n_stems, max_output, 4]
            'target_masks': torch.from_numpy(stem_masks),  # [n_stems, max_output]
        }


class GenerativeStemSeparator(nn.Module):
    """Transformer encoder-decoder that generates stem sinusoids from fullmix"""

    def __init__(self, n_stems=4, d_model=128, nhead=4, num_layers=4,
                 max_input_sinusoids=2000, max_output_sinusoids=500):
        super().__init__()

        self.n_stems = n_stems
        self.max_output_sinusoids = max_output_sinusoids

        # Input encoder
        self.input_proj = nn.Linear(4, d_model)
        self.pos_encoding_input = nn.Parameter(torch.randn(1, max_input_sinusoids, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder for each stem (shared across stems)
        self.output_queries = nn.Parameter(torch.randn(1, max_output_sinusoids, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection: generate sinusoid parameters
        self.output_proj = nn.Linear(d_model, 4)  # freq, amp, phase, frame

    def forward(self, x, input_mask):
        """
        Args:
            x: [batch, max_input, 4] fullmix sinusoids
            input_mask: [batch, max_input] validity mask
        Returns:
            stem_outputs: [batch, n_stems, max_output, 4] generated stem sinusoids
        """
        batch_size = x.shape[0]

        # Encode fullmix
        x = self.input_proj(x) + self.pos_encoding_input

        attn_mask = (input_mask == 0)  # True = ignore
        memory = self.encoder(x, src_key_padding_mask=attn_mask)  # [batch, max_input, d_model]

        # Decode for each stem
        stem_outputs = []

        # Expand queries for batch
        queries = self.output_queries.expand(batch_size, -1, -1)  # [batch, max_output, d_model]

        for stem_idx in range(self.n_stems):
            # Decode
            decoded = self.decoder(queries, memory, memory_key_padding_mask=attn_mask)

            # Project to sinusoid parameters
            stem_sinusoids = self.output_proj(decoded)  # [batch, max_output, 4]
            stem_outputs.append(stem_sinusoids)

        # Stack: [batch, n_stems, max_output, 4]
        stem_outputs = torch.stack(stem_outputs, dim=1)

        return stem_outputs


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        inputs = batch['input'].to(device)  # [batch, max_input, 4]
        input_mask = batch['input_mask'].to(device)  # [batch, max_input]
        targets = batch['targets'].to(device)  # [batch, n_stems, max_output, 4]
        target_masks = batch['target_masks'].to(device)  # [batch, n_stems, max_output]

        # Forward
        outputs = model(inputs, input_mask)  # [batch, n_stems, max_output, 4]

        # Loss: MSE on valid sinusoids only
        diff = (outputs - targets) ** 2  # [batch, n_stems, max_output, 4]
        diff = diff.mean(dim=-1)  # Average over 4 features -> [batch, n_stems, max_output]

        # Mask invalid sinusoids
        masked_loss = (diff * target_masks).sum() / target_masks.sum()

        # Backward
        optimizer.zero_grad()
        masked_loss.backward()
        optimizer.step()

        total_loss += masked_loss.item()
        pbar.set_postfix({'loss': f'{masked_loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train generative stem separator')
    parser.add_argument('--data-dirs', nargs='+', required=True, help='Song directories')
    parser.add_argument('--stem-names', nargs='+', default=['vocals', 'guitar', 'bass', 'drums'])
    parser.add_argument('--max-input-sinusoids', type=int, default=2000)
    parser.add_argument('--max-output-sinusoids', type=int, default=500, help='Per stem')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("=" * 60)
    print("Model 1: Generative Stem Separator Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Stems: {args.stem_names}")
    print(f"Max input sinusoids: {args.max_input_sinusoids}")
    print(f"Max output sinusoids per stem: {args.max_output_sinusoids}")
    print()

    # Create dataset
    dataset = GenerativeStemDataset(
        args.data_dirs,
        stem_names=args.stem_names,
        max_input_sinusoids=args.max_input_sinusoids,
        max_output_sinusoids=args.max_output_sinusoids
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
    model = GenerativeStemSeparator(
        n_stems=len(args.stem_names),
        d_model=128,
        nhead=4,
        num_layers=4,
        max_input_sinusoids=args.max_input_sinusoids,
        max_output_sinusoids=args.max_output_sinusoids
    ).to(args.device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\nStarting training...\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        loss = train_epoch(model, dataloader, optimizer, args.device)

        print(f"  Loss: {loss:.4f}\n")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'generative_stem_separator_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}\n")

    # Save final model
    final_path = 'generative_stem_separator_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")


if __name__ == '__main__':
    main()
