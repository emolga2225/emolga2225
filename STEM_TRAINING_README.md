# Stem Separator ML Training

Train a machine learning model to classify sinusoidal tracks into stems (vocals, drums, bass, other).

## Workflow

### 1. Extract Sinusoids

First, extract sinusoids from your mix and all stems:

```bash
# Extract from full mix
python hybrid_resolution_sinusoidal_extractor.py
# This creates: mix_tracks.h5

# Extract from each stem separately
# Modify the script to process each stem file
# This creates: vocals_tracks.h5, drums_tracks.h5, bass_tracks.h5, other_tracks.h5
```

### 2. Match Stems to Mix

Match sinusoidal tracks from stems to the mix to create training labels:

```bash
python match_stems.py
```

This creates `track_labels.json` containing:
- Mix track ID → Stem label mappings
- Similarity scores
- Statistics

The matching algorithm uses:
- **Frequency similarity**: Tracks at similar frequencies
- **Temporal overlap**: Tracks that exist at the same time
- **Hungarian algorithm**: Optimal assignment between mix and stem tracks

### 3. Train the Model

Train the transformer-based stem classifier:

```bash
python train_stem_separator.py
```

## Model Architecture

**StemClassifierTransformer**:
- Input: Sinusoidal track features (frequency, amplitude, phase trajectories)
- Embedding: Band position + positional encoding
- Encoder: Multi-head attention transformer (4 layers, 8 heads)
- Output: Softmax over stem classes

**Features per timestep**:
- Normalized frequency (0-1)
- Log amplitude
- Phase cos/sin components

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

**Label JSON Structure**:
```json
{
  "track_id": {
    "stem_idx": 0,
    "stem_name": "vocals",
    "similarity": 0.85
  }
}
```

## Configuration

Edit `train_stem_separator.py` to configure:
- `stem_names`: List of stem labels
- `max_seq_len`: Maximum track length (default: 512 frames)
- `d_model`: Model dimension (default: 128)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-4)
- `num_epochs`: Training epochs (default: 100)

## Output

Training produces:
- Checkpoints every 10 epochs: `stem_separator_epoch_X.pt`
- Final model: `stem_separator_final.pt`

## Inference

TODO: Add inference script to separate stems from a new mix using the trained model.

## Notes

- GPU highly recommended for training
- Tracks are padded/truncated to `max_seq_len`
- Uses masked attention to handle variable-length sequences
- Currently processes mono (channel 0) - extend for stereo if needed
