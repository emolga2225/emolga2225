# Stem Separator ML Training

Train a machine learning model to classify sinusoidal tracks into stems (vocals, drums, bass, other).

## Workflow

### 1. Extract Sinusoids

First, extract sinusoids from your mix and all stems using the batch script:

```bash
# Extract from all Guitar Hero .ogg files
python batch_extract.py
```

This creates HDF5 files:
- `fullmix_tracks.h5` (or `song_tracks.h5`) - the complete mix
- `vocals_tracks.h5`, `guitar_tracks.h5`, `bass_tracks.h5`, `drums_1_tracks.h5`, etc.

### 2. Match Stems to Mix (Frame-Level)

**IMPORTANT**: Use frame-level matching for immortal tracks!

With immortal tracks, a single mix track can contain multiple instruments at different times:
- Frames 0-100: vocals
- Frames 101-200: guitar
- Frames 201-300: bass

Use the frame-wise matcher:

```bash
python match_stems_framewise.py
```

This creates `frame_labels.json` containing:
- **Per-frame labels** for each mix track
- Each frame independently matched to a stem
- Handles immortal tracks correctly

The matching algorithm:
- **Frequency similarity**: Frames at similar frequencies
- **Temporal overlap**: 50ms time window
- **Amplitude similarity**: Weighted by amplitude correlation
- **Frame-by-frame**: Each frame matched independently

### 3. Train the Model (Frame-Level)

Train the frame-level transformer classifier:

```bash
python train_stem_separator_framewise.py
```

This predicts stem labels **per-frame** instead of per-track.

## Model Architecture

**FramewiseStemClassifier**:
- Input: Sliding windows of 32 frames from sinusoidal tracks
- Features per frame: normalized frequency, log amplitude, phase cos/sin
- Embedding: Band position + positional encoding
- Encoder: Multi-head attention transformer (4 layers, 8 heads, 128 dims)
- Output: **Per-frame** stem classification (softmax over stems + unknown class)

**Key differences from track-level:**
- Predicts label for **each frame independently**
- Uses sliding windows (32 frames, 16-frame stride)
- Handles immortal tracks where different frames = different stems
- Unknown class for unmatched frames

## Dataset Format

**Input HDF5 Structure**:
```
/c0/                    # Channel 0
  /f                    # Frequencies (float32)
  /a                    # Amplitudes (float32)
  /p                    # Phases (float32)
  /i                    # Frame indices (int32)
  /len                  # Track lengths (int32)
  /b                    # Band indices (uint8)
```

**Frame Label JSON Structure** (`frame_labels.json`):
```json
{
  "track_id": {
    "labels": [0, 0, 0, 1, 1, 1, 2, 2, -1, -1],
    "band": 3,
    "n_frames": 10
  }
}
```

Where:
- `labels`: Array of stem indices per frame (-1 = unmatched)
- `band`: Frequency band index
- `n_frames`: Total frames in track

Example: Track switches from vocals (0) → guitar (1) → bass (2) → unmatched

## Configuration

Edit `train_stem_separator_framewise.py` to configure:
- `stem_names`: List of stem labels (e.g., ['vocals', 'guitar', 'bass', 'drums'])
- `window_size`: Frames per training window (default: 32)
- `d_model`: Model dimension (default: 128)
- `nhead`: Attention heads (default: 8)
- `num_layers`: Transformer layers (default: 4)
- `batch_size`: Batch size (default: 64)
- `learning_rate`: Learning rate (default: 1e-4)
- `num_epochs`: Training epochs (default: 100)

Edit `match_stems_framewise.py` to configure:
- `freq_threshold`: Max frequency difference for matching (default: 100 Hz)
- Adjust stem file paths in `main()`

## Output

Training produces:
- Checkpoints every 10 epochs: `stem_separator_framewise_epoch_X.pt`
- Final model: `stem_separator_framewise_final.pt`

## Inference

TODO: Add inference script to:
1. Extract sinusoids from new mix
2. Classify each frame of each track
3. Group frames by predicted stem
4. Synthesize separated stems

## Notes

- **GPU highly recommended** - training much faster on CUDA
- **Frame-level approach** handles immortal tracks correctly
- Windows overlap by 50% (16-frame stride) for smooth predictions
- **Unknown class** for frames that don't match any stem (e.g., noise)
- Loss computed only on valid frames (uses masking)
- Currently processes mono (channel 0) - extend for stereo if needed
- For Guitar Hero: merge drums_1-4 into single "drums" stem for training
