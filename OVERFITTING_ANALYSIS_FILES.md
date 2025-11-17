# Complete File Summary with Absolute Paths

## Core Project Files

### Audio Processing (Sinusoidal Extraction)
- `/home/user/emolga2225/hybrid_resolution_sinusoidal_extractor.py` (859 lines)
  - Extracts sinusoidal tracks from audio
  - Input: Audio files (.ogg)
  - Output: HDF5 files with frequency, amplitude, phase data

- `/home/user/emolga2225/batch_extract.py` (97 lines)
  - Batch processes multiple audio files
  - Calls hybrid_resolution_sinusoidal_extractor.py

### Data Preparation & Label Generation
- `/home/user/emolga2225/match_stems_framewise.py` (242 lines)
  - **PRIMARY OVERFITTING SOURCE**: Creates deterministic frame-level labels
  - Matches sinusoids between mix and individual stems
  - Hard threshold: frequency within 100 Hz, score > 0.1
  - Output: `frame_labels.json`

- `/home/user/emolga2225/preprocess_training_data.py` (209 lines)
  - Converts HDF5 to memory-mapped numpy arrays
  - Creates feature vectors (freq, log_amp, cos_phase, sin_phase)
  - Window filtering: Skips windows with >50% unmatched frames
  - Output: features.npy, labels.npy, masks.npy, bands.npy

- `/home/user/emolga2225/preprocess_training_data_v2.py` (233 lines)
  - Alternative preprocessing (v2 variant)

### Model Training (3 Variants)
- `/home/user/emolga2225/train_stem_separator_framewise.py` (347 lines)
  - Frame-level classification from HDF5 files
  - **SLOWEST** due to HDF5 I/O per batch
  - Learning rate: 1e-4, Epochs: 100, Batch: 64
  - NO dropout scheduling, NO validation split

- `/home/user/emolga2225/train_stem_separator_fast.py` (282 lines)
  - Frame-level classification from preprocessed numpy
  - **FAST** due to memory-mapped arrays
  - Learning rate: 5e-3, Epochs: 10, Batch: 256
  - Includes cosine annealing scheduler
  - NO weight decay, NO data augmentation, NO validation split

- `/home/user/emolga2225/train_stem_separator.py` (338 lines)
  - Track-level classification (older approach)
  - Different architecture (global pooling instead of per-frame)

- `/home/user/emolga2225/resume_training.py` (180 lines)
  - Resume from checkpoint with additional epochs
  - Supports class-weighted loss
  - **PROBLEM**: Caps unknown class weight to 1.0 (line 73)

### Inference & Utilities
- `/home/user/emolga2225/infer_stem_separation.py` (523 lines)
  - Loads trained model, classifies new audio
  - Sliding window inference (32-frame window, 50% overlap)
  - Synthesizes separated stem audio

- `/home/user/emolga2225/check_bands.py` (21 lines)
  - Inspects band values in preprocessed data

- `/home/user/emolga2225/debug_preprocessing.py` (44 lines)
  - Debug script for preprocessing issues

- `/home/user/emolga2225/debug_preprocessing_detailed.py` (63 lines)
  - Detailed preprocessing debug

- `/home/user/emolga2225/inspect_hdf5_structure.py` (25 lines)
  - Inspects HDF5 file structure

### Documentation
- `/home/user/emolga2225/STEM_TRAINING_README.md` (141 lines)
  - Workflow documentation
  - Model architecture description
  - Configuration guide

- `/home/user/emolga2225/README.md` (6 lines)
  - Generic GitHub profile README

### Matching Alternatives (Performance Variants)
- `/home/user/emolga2225/match_stems.py` (223 lines)
  - Original matching approach (basic)

- `/home/user/emolga2225/match_stems_band_filtered.py` (288 lines)
  - Band-filtered matching (optimized)

- `/home/user/emolga2225/match_stems_framewise_fast.py` (275 lines)
  - Fast frame-wise matching

- `/home/user/emolga2225/match_stems_ultra_fast.py` (261 lines)
  - Ultra-fast vectorized matching

**NOTE**: All matching variants use same hard threshold approach (100 Hz frequency window)

---

## Model Architecture Breakdown

```
FramewiseStemClassifier (from train_stem_separator_fast.py lines 71-133)
├── Input Projection
│   └── Linear(4 → 128)
├── Band Embedding
│   └── Embedding(32 bands → 128)
├── Positional Encoding
│   └── Parameter(window_size × 128)
├── Transformer Encoder
│   ├── 4 TransformerEncoderLayers
│   │   ├── d_model: 128
│   │   ├── nhead: 8
│   │   ├── dim_feedforward: 512 (128 × 4)
│   │   ├── dropout: 0.1
│   │   └── batch_first: True
│   └── num_layers: 4
└── Classification Head
    ├── Linear(128 → 128)
    ├── ReLU()
    ├── Dropout(0.1)
    └── Linear(128 → 5 stems) or (128 → 6 with unknown)
```

**Total Parameters**: ~65,000+
**Dropout**: Only 0.1 (very conservative)
**Regularization**: Only via mask filtering

---

## Key Code Issues - Overfitting

### Issue 1: Hard Threshold Matching (match_stems_framewise.py:61-106)
```python
def match_frame_to_stems(self, mix_freq, mix_time, mix_amp, stem_tracks_list, stem_names):
    # Line 94: Hard frequency threshold (100 Hz)
    if freq_diff < self.freq_threshold:
        freq_similarity = 1.0 - (freq_diff / self.freq_threshold)
        amp_similarity = min(mix_amp, stem_amp) / (max(mix_amp, stem_amp) + 1e-8)
        score = freq_similarity * amp_similarity
    
    # Line 104: Binary match decision (> 0.1 threshold)
    if best_score > 0.1:
        return best_stem
    return -1
```
**Problem**: No soft labeling, no confidence scores - model doesn't learn uncertainty

### Issue 2: Window Filtering (preprocess_training_data.py:73-78)
```python
# Skip windows with too many unmatched frames
unmatched_count = sum(1 for l in window_labels if l != -1)
if unmatched_count < window_size * match_threshold:
    continue  # SKIP THIS WINDOW - filters out hard examples
```
**Problem**: Only includes "easy" well-matched frames - biases toward overconfident matching

### Issue 3: No Regularization (train_stem_separator_fast.py:246)
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['learning_rate']
    # weight_decay=0 (MISSING - no L2 regularization!)
)

criterion = nn.CrossEntropyLoss(reduction='none')
# No Label Smoothing, no focal loss
```
**Problem**: No weight decay, no regularization beyond dropout

### Issue 4: No Data Augmentation (train_stem_separator_fast.py:61-68)
```python
def __getitem__(self, idx):
    return {
        'features': torch.from_numpy(self.features[idx].copy()),
        'mask': torch.from_numpy(self.masks[idx].copy()),
        'band': torch.tensor([self.bands[idx]], dtype=torch.long),
        'labels': torch.from_numpy(self.labels[idx].copy())
    }
    # NO AUGMENTATION - same features every epoch!
```
**Problem**: No jitter, noise, mixup - model sees identical data

### Issue 5: No Validation Split (train_stem_separator_fast.py:222-229)
```python
dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True if config['device'] == 'cuda' else False,
    persistent_workers=True
    # NO validation_split, NO separate validation dataloader
)
```
**Problem**: Can't detect overfitting - only training metrics reported

### Issue 6: Unknown Class Weight Capping (resume_training.py:66-73)
```python
class_weights_np = total / (6 * class_counts + 1e-6)
class_weights_np = class_weights_np / class_weights_np.sum() * 6

# Line 73: Hard cap unknown class to 1.0
class_weights_np[5] = 1.0  # Set unknown to baseline weight
```
**Problem**: Artificially down-weights ambiguous/unmatched frames

---

## Training Hyperparameters Comparison

| Parameter | train_framewise.py | train_fast.py | resume_training.py |
|-----------|-------------------|---------------|-------------------|
| Learning Rate | 1e-4 | 5e-3 | Variable |
| Batch Size | 64 | 256 | 256 |
| Epochs | 100 | 10 | Variable |
| Optimizer | AdamW | Adam | Adam |
| Scheduler | None | CosineAnnealing | CosineAnnealing |
| Weight Decay | None | None | None |
| Dropout | 0.1 | 0.1 | 0.1 |
| Label Smoothing | No | No | No |
| Data Augmentation | No | No | No |
| Validation Split | No | No | No |
| Validation Samples | 0% | 0% | 0% |

---

## Git History - Optimization Attempts
```
d7dad8b Cap unknown class weight to prevent training instability
afe236b Add class-weighted loss to resume training
9d087af Add script to resume training from checkpoint
5ae5641 Add sliding window classification for immortal tracks support
0e73090 Optimize inference speed with batching
cd6aebe Implement full synthesis pipeline
55d5260 Add inference script for stem separation
7cba60e Set learning rate to 5e-3 for balanced convergence
f16785b Reduce learning rate from 2e-2 to 1e-2
c76aeeb Configure for fast 10-epoch training with aggressive learning rate
c1e6fe2 Increase learning rate from 1e-3 to 5e-3
...
```

**Pattern**: Multiple learning rate changes (1e-3 → 5e-3 → 1e-2 → 5e-3 → 1e-4) 
- Suggests instability
- No systematic approach to hyperparameter tuning
- Indicates overfitting was a known issue

---

## Dataset Statistics
- Typical dataset size: 50,000+ windows
- Window size: 32 frames (32 sinusoids per sample)
- Window stride: 16 frames (50% overlap means same frames repeated)
- Features per frame: 4 (frequency, log_amplitude, cos_phase, sin_phase)
- Classes: 5 stems (vocals, guitar, bass, drums, other) + unknown class
- Unmatched frames: -1 (unknown class during inference)
- Batch size: 64-256 samples (2,048-8,192 frames per batch)

---

## Data Flow with Overfitting Points

```
Audio Files (.ogg, 44.1-192kHz)
    ↓
[1] hybrid_resolution_sinusoidal_extractor.py
    - FFT-based frequency tracking
    - Extracts: frequencies, amplitudes, phases
    Output: HDF5 (fullmix_tracks.h5, vocals_tracks.h5, etc.)
    
    ↓
[2] match_stems_framewise.py ← OVERFITTING SOURCE #1
    - Hard matching: freq within 100 Hz
    - Binary decision: match or no-match
    - NO soft labels, NO confidence scores
    - Deterministic (same input → same labels)
    Output: frame_labels.json
    
    ↓
[3] preprocess_training_data.py ← OVERFITTING SOURCE #2
    - Sliding windows: 32-frame, 16-stride (50% overlap)
    - Window filtering: Skip >50% unmatched
    - Feature normalization: freq/96000, log1p(amp), cos/sin(phase)
    - NO augmentation applied
    Output: features.npy, labels.npy, masks.npy, bands.npy
    
    ↓
[4] train_stem_separator_fast.py ← OVERFITTING SOURCE #3
    - Dataset: PreprocessedStemDataset (no augmentation in __getitem__)
    - Model: FramewiseStemClassifier (65k params, transformer)
    - Training: Adam optimizer (no weight decay), CrossEntropyLoss
    - NO validation split
    - Metrics: Only training loss/accuracy
    Output: Trained model (checkpoint_epoch_*.pt)
    
    ↓
[5] infer_stem_separation.py
    - Loads checkpoint
    - Classifies new audio
    - Outputs: Separated stems (.wav files)
```

