#!/usr/bin/env python3
"""Check what the model is actually learning"""

import torch
import glob

# Find latest checkpoint
checkpoints = sorted(glob.glob('stem_separator_framewise_epoch_*.pt'))

if not checkpoints:
    print("No checkpoints found!")
    exit(1)

latest = checkpoints[-1]
print(f"Loading checkpoint: {latest}")

checkpoint = torch.load(latest, map_location='cpu')

epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']
train_acc = checkpoint['train_acc']
config = checkpoint['config']

print(f"\nEpoch: {epoch + 1}")
print(f"Loss: {train_loss:.4f}")
print(f"Accuracy: {train_acc:.2f}%")

print(f"\nConfiguration:")
print(f"  Stems: {config['stem_names']}")
print(f"  Label smoothing: {config.get('label_smoothing', 'N/A')}")
print(f"  Weight decay: {config.get('weight_decay', 'N/A')}")
print(f"  Augmentation: {config.get('augment', 'N/A')}")
print(f"  Batch size: {config['batch_size']}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

# Check if accuracy is plateauing
if train_acc < 80:
    print(f"\n⚠️  Accuracy {train_acc:.1f}% is lower than expected")
    print("Possible causes:")
    print("  1. Label smoothing (0.1) is capping max accuracy ~80-85%")
    print("  2. High dropout (0.3) is slowing learning")
    print("  3. Too much augmentation noise")
    print("  4. Poor quality labels (lots of 'unknown' class)")

print(f"\nWith label smoothing = {config.get('label_smoothing', 0.1)}:")
print(f"  Maximum achievable accuracy: ~{100 * (1 - config.get('label_smoothing', 0.1)):.1f}%")
print(f"  Current accuracy: {train_acc:.1f}%")
print(f"  Room for improvement: {100 * (1 - config.get('label_smoothing', 0.1)) - train_acc:.1f}%")
