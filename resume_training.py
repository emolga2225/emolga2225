#!/usr/bin/env python3
"""
Resume training from a checkpoint.

Loads the last checkpoint and continues training for additional epochs.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Import from the original training script
from train_stem_separator_fast import PreprocessedStemDataset, FramewiseStemClassifier, train_epoch


def resume_training(checkpoint_path, additional_epochs=20, new_lr=None):
    """Resume training from a checkpoint"""

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)

    config = checkpoint['config']
    start_epoch = checkpoint['epoch'] + 1  # Continue from next epoch

    # Override with new epochs
    config['n_epochs'] = start_epoch + additional_epochs - 1

    # Optionally set new learning rate
    if new_lr is not None:
        config['learning_rate'] = new_lr
        print(f"Using new learning rate: {new_lr}")

    print("\nResuming training configuration:")
    print(f"  Starting from epoch: {start_epoch}")
    print(f"  Training until epoch: {config['n_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Previous train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Previous train acc: {checkpoint['train_acc']:.2f}%")
    print()

    # Create dataset
    print("Loading dataset...")
    dataset = PreprocessedStemDataset(config['data_dir'])

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False,
        persistent_workers=True
    )

    # Create model
    print("Creating model...")
    model = FramewiseStemClassifier(
        n_stems=config['n_stems'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        window_size=config['window_size']
    ).to(config['device'])

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Create new scheduler for remaining epochs
    remaining_epochs = config['n_epochs'] - start_epoch + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_epochs,
        eta_min=1e-5
    )

    # If checkpoint had scheduler state, optionally load it
    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("Note: Could not load scheduler state, using fresh scheduler")

    criterion = nn.CrossEntropyLoss(reduction='none')

    # Training loop
    print("\nResuming training...\n")
    for epoch in range(start_epoch, config['n_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['n_epochs']} (lr: {scheduler.get_last_lr()[0]:.6f})")

        train_loss, train_acc = train_epoch(
            model, dataloader, optimizer, criterion, config['device']
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Step the scheduler
        scheduler.step()

        # Save checkpoint
        if epoch % config['save_every'] == 0 or epoch == config['n_epochs']:
            checkpoint_path = Path(config['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint_epoch_10.pt',
                        help='Checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Additional epochs to train')
    parser.add_argument('--lr', type=float, default=None,
                        help='New learning rate (optional, defaults to checkpoint LR)')

    args = parser.parse_args()

    resume_training(args.checkpoint, additional_epochs=args.epochs, new_lr=args.lr)
