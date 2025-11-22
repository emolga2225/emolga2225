# 2D Grid Approach for Stem Separation

## The Problem with Variable-Length Sinusoids

**Original approach:** Represent sinusoids as variable-length lists of `[freq, amp, frame]` tuples.

**Issues:**
- Full song has **52 million sinusoids**
- Transformers require fixed-length inputs
- Had to limit to 2000 input / 500 output sinusoids
- **Threw away 99.9% of the data** → terrible audio quality
- No neural network can process 52M tokens efficiently

## The 2D Grid Solution

### Key Insight

**Each "track" in the HDF5 file represents ONE frequency component evolving over time.**

This is perfect for a 2D representation!

### Conversion Process

**From:** Variable-length list of sinusoids
```
[freq1, amp1, phase1, frame1]
[freq2, amp2, phase2, frame2]
...
[freq_52M, amp_52M, phase_52M, frame_52M]
```

**To:** Fixed-size 2D grid with 2 channels
```
Grid shape: (2, frequency_bins, time_frames)
Grid[0, freq_bin, time_idx] = magnitude at that (frequency, time) location
Grid[1, freq_bin, time_idx] = phase at that (frequency, time) location
```

**Phase handling:** When multiple sinusoids fall in the same frequency bin at the same time, they are combined as complex numbers (magnitude * e^(i*phase)), then converted back to magnitude and phase. This preserves phase coherence.

### Benefits

1. **Keeps 100% of sinusoidal data** - Nothing is thrown away!
2. **Fixed-size representation** - Works perfectly with CNNs and transformers
3. **Natural 2D structure** - Ideal for convolutional operations
4. **Memory efficient** - 52M sinusoids → 512×40000 grid = 80 MB (8× smaller!)
5. **Similar to spectrogram** - But built directly from sinusoids (no STFT loss)

### Example

Suppose we have 3 sinusoidal tracks:
- Track 0: 440 Hz over frames 0-100 with varying amplitude
- Track 1: 880 Hz over frames 50-150 with varying amplitude
- Track 2: 1320 Hz over frames 0-80 with varying amplitude

**Traditional representation:**
- ~250 variable-length sinusoids
- Can't process with standard neural networks

**2D Grid representation:**
- Shape: (512 freq bins, 150 time frames)
- Each cell contains the amplitude at that (frequency, time)
- Fixed-size, ready for CNN/transformer

## Architecture: U-Net

We use **U-Net**, the standard architecture for source separation:

```
Input: (1, freq_bins, time_frames) - fullmix grid

Encoder:
  Conv → Pool → Conv → Pool → Conv → Pool → Conv → Pool
  (Extracts hierarchical features)

Bottleneck:
  Conv (deepest features)

Decoder (with skip connections):
  Upsample → Concat → Conv → Upsample → Concat → Conv → ...
  (Reconstructs at original resolution)

Output: (n_stems, freq_bins, time_frames) - one grid per stem
```

**Why U-Net?**
- Proven for audio source separation (Open-Unmix, Demucs, etc.)
- Skip connections preserve fine-grained details
- Encoder-decoder structure perfect for separation tasks

## Data Flow

### Training
```
1. Load HDF5 sinusoidal tracks
2. Convert to 2D grid: (freq_bins, time_frames)
3. Feed to U-Net model
4. Model outputs 5 grids (one per stem)
5. Compare with ground truth stem grids
6. Optimize with L1 loss
```

### Inference
```
1. Load fullmix HDF5
2. Convert to 2D grid
3. Feed to trained U-Net
4. Get 5 predicted stem grids
5. Convert each grid back to sinusoids
6. Synthesize audio via additive synthesis
```

## Grid Parameters

- **Frequency bins:** 512 (covers 0-22050 Hz, ~43 Hz per bin)
- **Time frames:** Based on chunk duration (4 sec = 344 frames @ 512 hop)
- **Frequency binning:** Linear spacing from 0 to 22050 Hz
- **Amplitude accumulation:** If multiple sinusoids fall in same bin, add amplitudes

## Comparison with Original Approach

| Aspect | Original (Variable-Length) | New (2D Grid) |
|--------|---------------------------|---------------|
| Data kept | 0.1% (2000/500K) | **100%** |
| Size | Variable (52M sinusoids) | Fixed (512×344) |
| Architecture | Transformer | U-Net (proven for audio) |
| Memory | 624 MB | 80 MB |
| Neural network compatibility | Poor | Excellent |
| Information loss | Massive | Minimal (binning only) |

## Files

- `convert_sinusoids_to_2d_grid.py` - Conversion utilities and demo
- `train_stem_separator_2d_grid.py` - Training script with U-Net
- `infer_2d_grid_separator.py` - Inference script with audio synthesis

## Usage

### Training
```bash
python train_stem_separator_2d_grid.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --stem-names vocals guitar bass drums song \
  --n-freq-bins 512 \
  --chunk-duration 4.0 \
  --batch-size 8 \
  --epochs 100
```

### Inference
```bash
python infer_2d_grid_separator.py \
  --fullmix-h5 ajfa/fullmix_tracks.h5 \
  --checkpoint stem_separator_2d_epoch100.pt \
  --output-dir separated_stems/
```

## Why This Works

1. **Sinusoidal tracks ARE frequency components over time** - perfect for 2D grid
2. **CNNs excel at 2D patterns** - can learn which frequencies belong to which instruments
3. **100% data retention** - model sees ALL sinusoids, can learn full distribution
4. **Fixed-size inputs** - no more variable-length issues
5. **Proven architecture** - U-Net is standard for source separation

This approach keeps all your sinusoidal data while making it compatible with modern neural networks!
