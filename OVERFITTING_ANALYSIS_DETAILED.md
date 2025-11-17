# Stem Separation ML Training - Codebase Analysis & Overfitting Issues

## Project Overview

**Project Type**: Music stem separation (source separation) using sinusoidal track classification
**Purpose**: Classify sinusoidal tracks extracted from audio mixes into individual stems (vocals, drums, bass, guitar, other)
**Approach**: Frame-level transformer-based classification with sinusoidal features

## Architecture & Workflow

### 1. Sinusoidal Extraction (`hybrid_resolution_sinusoidal_extractor.py` - 859 lines)
- Extracts sinusoidal tracks from audio using FFT-based frequency tracking
- Tracks per-frame parameters:
  - **Frequency** (Hz)
  - **Amplitude** (linear)
  - **Phase** (radians)
  - **Frame index** (temporal position)
- Multi-band extraction with hybrid resolution
- Stores in HDF5 format (fullmix_tracks.h5, vocals_tracks.h5, etc.)

### 2. Frame-Level Stem Matching (`match_stems_framewise.py` - 242 lines)
**CRITICAL FOR OVERFITTING**: Creates labels by frame-by-frame matching
- **Matching Algorithm**:
  - Time window: 50ms overlap detection
  - Frequency similarity: Within `freq_threshold` (default: 100 Hz)
  - Amplitude similarity: Correlation-based weighting
  - Score: `freq_similarity * amp_similarity`
  - Match threshold: `score > 0.1`
- **Output**: `frame_labels.json` with per-frame stem labels (-1 = unmatched)
- **Deterministic nature**: Same input always produces same labels (no noise/uncertainty)

### 3. Data Preprocessing (`preprocess_training_data.py` - 209 lines)
- Converts HDF5 to memory-mapped numpy arrays for fast loading
- Creates feature vectors:
  ```
  features[:, 0] = normalized_frequency (freq / 96000.0)
  features[:, 1] = log_amplitude (log1p(amp))
  features[:, 2] = cos(phase)
  features[:, 3] = sin(phase)
  ```
- Sliding windows: `window_size=32`, `window_stride=16` (50% overlap)
- Window filtering: Skips windows with >50% unmatched frames
- Output: features.npy, labels.npy, masks.npy, bands.npy

### 4. Model Architecture

**Model**: `FramewiseStemClassifier` (transformer-based)
```
Input: [batch, window_size=32, 4_features]
       ↓
Input Projection: Linear(4 → 128)  [batch, 32, 128]
       ↓
+ Band Embedding: Embedding(32 bands → 128)  [batch, 128] broadcast
       ↓
+ Positional Encoding: Learnable [32, 128]
       ↓
Transformer Encoder (4 layers, 8 heads, 128 dims):
  - Hidden dim: 128 * 4 = 512
  - Dropout: 0.1
       ↓
Frame-level Classifier: Linear(128 → 5_stems)  [batch, 32, 5]
Output: [batch, 32, 5] per-frame logits (softmax for classification)
```

**Parameters**: ~65,000+ parameters

### 5. Training Variants

| Script | Data Format | Loading Speed | Features |
|--------|------------|---------------|----------|
| `train_stem_separator_framewise.py` | HDF5 (direct) | **Slow** | Basic training |
| `train_stem_separator_fast.py` | Preprocessed numpy | **Fast** | Cosine annealing LR scheduler, supports class weights |
| `resume_training.py` | Preprocessed numpy | **Fast** | Resume from checkpoint, class-weighted loss, capped unknown weight |

---

## Root Causes of Overfitting

### Problem 1: Precise/Excess Training Data (PRIMARY ISSUE)

**The deterministic matching creates unnaturally clean labels:**

1. **No uncertainty in labels**:
   - Frame-by-frame matching uses hard thresholds (freq within 100 Hz, score > 0.1)
   - Same sinusoid always maps to same stem (deterministic, not probabilistic)
   - Real-world audio has ambiguity; model is given unrealistic certainty

2. **Data augmentation conflict**:
   - NO data augmentation applied to features
   - Frequencies, amplitudes, phases used as-is
   - Model sees exact same sinusoid representation every epoch
   - Window overlap (50%) means same frames repeated across windows

3. **Window redundancy**:
   - Window stride = 16 frames, window size = 32 frames
   - Frame 1-32 appears in window 1, 2-33 in window 2, 3-34 in window 3, etc.
   - Same frame seen ~1-2 times per epoch
   - Creates artificial sample multiplication

4. **Frequency matching threshold too strict**:
   - 100 Hz threshold with 96 kHz sample rate (0.1% of audio bandwidth)
   - May force perfect matches where human ears hear ambiguity
   - No smoothness/tolerance for slight frequency variations

5. **Window-level filtering**:
   ```python
   # Skip windows with >50% unmatched frames
   unmatched_count = sum(1 for l in window_labels if l == -1)
   if unmatched_count > self.window_size * 0.5:
       continue
   ```
   - Biases training toward well-matched (easy) data
   - Doesn't include hard negative examples
   - Model never learns to handle ambiguous/noisy frames

### Problem 2: Insufficient Regularization

**Current regularization**:
- ✅ Dropout: 0.1 (quite low)
- ✅ Mask-based loss filtering (only on valid frames)
- ❌ **NO L1/L2 weight regularization** (weight_decay=0 in AdamW)
- ❌ **NO data augmentation**
- ❌ **NO batch normalization**
- ❌ **NO layer normalization**
- ❌ **NO mixup/cutmix**
- ❌ **NO early stopping**
- ❌ **NO validation set**

### Problem 3: Model Capacity vs Data Quality

- **Small model**: ~65k parameters
- **Simple features**: Only 4 features (freq, log_amp, cos_phase, sin_phase)
- **Small window**: 32 frames = 32 sinusoids
- **Transformer advantage unused**: Self-attention is powerful but over-capacitated for 32 clean frames

The model likely memorizes the training set patterns:
- Frame 1-5 from band 3 → vocal
- Frame 6-10 from band 3 → guitar
- Exact phase patterns → exact stem

### Problem 4: Unknown Class Weight Capping

```python
# In resume_training.py line 73
class_weights_np[5] = 1.0  # Set unknown to baseline weight
```

**Issues**:
- Unknown class (-1 frames) is artificially down-weighted
- Model gets signal that unknown frames are less important
- May overfit to matching stems instead of learning what's genuinely "unknown"
- No gradual weighting strategy

### Problem 5: Training Configuration

**No validation split:**
```python
dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    # NO validation_split=0.2
)
```

**Observed metrics**:
- Only training loss/accuracy reported
- No generalization metrics
- No way to detect overfitting in real-time

**Learning rate oscillations**:
- Git history shows multiple LR changes: 1e-3 → 5e-3 → 1e-2 → 5e-3 → 1e-4
- Suggests instability in training
- No clear convergence criteria

---

## Key Files for Overfitting Analysis

### Primary Culprits:

1. **`match_stems_framewise.py` (lines 61-106)**
   - `match_frame_to_stems()` - Hard threshold matching (freq within 100Hz)
   - Line 104: `if best_score > 0.1:` - Binary match/no-match decision
   - **Fix needed**: Probabilistic matching, confidence scores, softer thresholds

2. **`preprocess_training_data.py` (lines 66-78)**
   - Line 73: Skips windows with >50% unmatched frames
   - **Fix needed**: Include hard examples, weighted sampling by match confidence

3. **`train_stem_separator_fast.py` (lines 136-187)**
   - Line 248: `criterion = nn.CrossEntropyLoss(reduction='none')`
   - No L1/L2 regularization in optimizer (line 246)
   - No data augmentation in `__getitem__` (line 64-68)
   - **Fix needed**: Add weight decay, data augmentation, validation split

4. **`resume_training.py` (lines 50-82)**
   - Class weight capping (line 73)
   - Unknown class weighted equally to others
   - **Fix needed**: Better unknown class handling

### Configuration Issues:

1. **`train_stem_separator_fast.py` main() - lines 190-282**
   - Window size: 32 (quite small)
   - Learning rate: 5e-3 (quite high for 10 epochs)
   - No learning rate scheduling in base version
   - Batch size: 256 (large, may miss hard examples)
   - **Training for only 10 epochs** (should be more with smaller LR)

2. **`train_stem_separator_framewise.py` main() - lines 260-315**
   - Learning rate: 1e-4 (very low)
   - 100 epochs (but no scheduler)
   - Batch size: 64
   - No validation split

---

## Data Flow Summary

```
Audio Files (.ogg)
    ↓
hybrid_resolution_sinusoidal_extractor.py
    ↓
HDF5 Files (fullmix_tracks.h5, vocals_tracks.h5, etc.)
    ↓ [PRECISE MATCHING HAPPENS HERE]
match_stems_framewise.py
    ↓
frame_labels.json (deterministic, no uncertainty)
    ↓ [WINDOW FILTERING BIASES DATA]
preprocess_training_data.py
    ↓
Preprocessed numpy arrays (features.npy, labels.npy, masks.npy, bands.npy)
    [NO AUGMENTATION APPLIED]
    ↓
train_stem_separator_fast.py
    ↓
Trained Model (overfits to training data patterns)
```

---

## Dataset Statistics Points

From `preprocess_training_data.py` output example:
```
Total valid windows: 50,000+ (depends on data)
Window size: 32 frames per window
Window stride: 16 frames (50% overlap)
Features per frame: 4 (frequency, log_amplitude, cos_phase, sin_phase)
Labels: 0-4 stems + unknown class
File sizes: Features ~GB, Labels ~GB, Masks ~GB, Bands ~MB
```

**Issue**: High window count due to overlap, but same underlying frames

---

## Recommendations for Fixing Overfitting

### High Priority (Address Primary Cause):

1. **Soft Label Matching** (`match_stems_framewise.py`)
   - Return match confidence (0.0-1.0) instead of binary match/no-match
   - Create soft labels: `label_probs = [0.1, 0.8, 0.1, ...]` for ambiguous frames
   - Use KL divergence loss instead of hard cross-entropy

2. **Include Hard Negatives** (`preprocess_training_data.py`)
   - Remove window filtering that skips >50% unmatched
   - Weight windows by match confidence
   - Include frames that barely fail matching threshold

3. **Data Augmentation** (`train_stem_separator_fast.py` __getitem__)
   ```python
   # Add random jitter
   features[:, 0] += np.random.normal(0, 0.01)  # frequency jitter
   features[:, 1] += np.random.normal(0, 0.05)  # amplitude jitter
   features[:, 2:4] += np.random.normal(0, 0.05)  # phase jitter
   ```

### Medium Priority (Reduce Overfitting):

4. **Add Regularization** (`train_stem_separator_fast.py`)
   - Change line 246: `lr=5e-3, weight_decay=0.01`
   - Increase dropout: 0.1 → 0.3
   - Add L1 regularization if needed

5. **Add Validation Split** (`train_stem_separator_fast.py`)
   - Split dataset: 80% train, 20% validation
   - Monitor validation loss for early stopping
   - Detect overfitting in real-time

6. **Reduce Frequency Threshold** (`match_stems_framewise.py`)
   - 100 Hz → 50 Hz for stricter matching
   - OR increase threshold to 200 Hz for softer matching
   - Test both approaches for better trade-off

### Low Priority (Optimization):

7. **Reduce Model Capacity**
   - Transformer layers: 4 → 2-3
   - Model dimension: 128 → 64-96
   - Nhead: 8 → 4

8. **Better Learning Rate Scheduling**
   - Cosine annealing (already in fast.py line 247)
   - Warmup phase (first 10% of epochs with lower LR)
   - ReduceLROnPlateau if validation plateaus

9. **Unknown Class Handling** (`resume_training.py`)
   - Remove hard cap at 1.0 (line 73)
   - Let class weights learn naturally
   - Or use focal loss for better unknown class handling

---

## File Modification Checklist

- [ ] `match_stems_framewise.py`: Add confidence scores to matching
- [ ] `preprocess_training_data.py`: Include hard negatives, weight by confidence
- [ ] `train_stem_separator_fast.py`: Add augmentation, validation, regularization
- [ ] `resume_training.py`: Remove unknown class weight capping
- [ ] Create `train_stem_separator_regularized.py`: Combined fixes version

