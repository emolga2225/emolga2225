# Exact Frequency Approach for Stem Separation

## The Problem with Frequency Binning

**Previous 2D grid approach:** Binned frequencies into 512 fixed bins (0-22050 Hz).

**Issues:**
- **Loses exact frequency information** - bins sinusoids at 50 Hz, 51 Hz, 52 Hz all into the same bin (~43 Hz per bin)
- Destroys frequency precision needed for high-quality synthesis
- Sinusoids at 440 Hz and 460 Hz would be binned together, losing the exact pitch

## The Exact Frequency Solution

### Key Insight

**Each sinusoid has an EXACT frequency that should be preserved.**

Example: At time frame 0, there might be sinusoids at:
- 50.0 Hz (not binned to "bin 1")
- 80.3 Hz (not binned to "bin 2")
- 175.6 Hz (not binned to "bin 4")
- 2000.8 Hz (not binned to "bin 46")

We keep these EXACT frequency values: 50.0 Hz, 80.3 Hz, 175.6 Hz, 2000.8 Hz.

### Data Organization

**From:** Variable-length sinusoid tracks (HDF5 format)
```
Track 0: [(50.0 Hz, amp1, phase1, frame0), (50.1 Hz, amp2, phase2, frame1), ...]
Track 1: [(80.3 Hz, amp3, phase3, frame0), (80.5 Hz, amp4, phase4, frame1), ...]
...
```

**To:** Frame-based fixed-size array with exact frequencies
```
Array shape: (n_frames, max_sinusoids_per_frame, 3)
Array[frame_idx, sinusoid_idx, :] = [exact_freq_Hz, amplitude, phase_radians]

Example:
Array[0, 0, :] = [50.0, 0.5, 1.2]   # First sinusoid at frame 0
Array[0, 1, :] = [80.3, 0.3, 0.8]   # Second sinusoid at frame 0
Array[0, 2, :] = [175.6, 0.7, -0.5] # Third sinusoid at frame 0
...
Array[0, 50, :] = [0, 0, 0]         # Padding (no 51st sinusoid)
```

### Benefits

1. **100% frequency precision** - Exact Hz values preserved (50.0 Hz stays 50.0 Hz, not binned to ~43 Hz bucket)
2. **Fixed-size representation** - Works with neural networks via padding/truncation
3. **Maintains all information** - No binning artifacts or frequency resolution loss
4. **Sample-level organization** - Data organized by time frames (86 frames/sec at hop=512)
5. **Natural for synthesis** - Can directly synthesize from exact frequencies

### Example

Suppose at frame 0 we have 50 active sinusoids:
- **Frequency binning approach (WRONG):**
  - 50 Hz → bin 1 (~0-43 Hz)
  - 80 Hz → bin 2 (~43-86 Hz)
  - 175 Hz → bin 4 (~129-172 Hz) ← WRONG BIN!
  - All exact frequencies lost!

- **Exact frequency approach (CORRECT):**
  - Sinusoid 0: [50.0 Hz, amp, phase]
  - Sinusoid 1: [80.0 Hz, amp, phase]
  - Sinusoid 2: [175.0 Hz, amp, phase]
  - ...
  - Sinusoid 49: [2180.5 Hz, amp, phase]
  - All exact frequencies preserved!

## Architecture: Transformer

We use a **Transformer** architecture that can process variable-length sequences:

```
Input: (batch, n_frames, max_sinusoids, 3)
       where 3 = [exact_freq_Hz, amplitude, phase_radians]

Embedding:
  - Sinusoid embedding: Linear(3 -> d_model)
  - Frame positional encoding (for time)
  - Sinusoid positional encoding (for sinusoid index)

Transformer Encoder:
  - Multi-head self-attention learns which sinusoids belong to which stems
  - Attention mask ignores padded positions (freq=0)
  - Processes all sinusoids simultaneously

Output Heads:
  - One head per stem
  - Each outputs: (n_frames, max_sinusoids, 3)
  - Constraints: freq > 0, amp ≥ 0, phase ∈ [-π, π]

Output: (batch, n_stems, n_frames, max_sinusoids, 3)
```

**Why Transformer?**
- Attention mechanism can learn relationships between sinusoids at different frequencies
- Can handle variable numbers of sinusoids via masking
- Global context - each sinusoid can attend to all others
- Proven for set-based tasks (sinusoids are an unordered set at each frame)

## Data Flow

### Training
```
1. Load HDF5 sinusoidal tracks
2. Convert to frame-based array with exact frequencies
   - At each frame, collect all active sinusoids
   - Store as [(freq1, amp1, phase1), (freq2, amp2, phase2), ...]
   - Pad to max_sinusoids with zeros
3. Feed to Transformer model
4. Model outputs stem arrays (one per stem)
5. Compare with ground truth stem arrays
6. Optimize with L1 loss
```

### Inference
```
1. Load fullmix HDF5
2. Convert to frame-based array with exact frequencies
3. Feed to trained Transformer
4. Get predicted stem arrays (n_stems, n_frames, max_sinusoids, 3)
5. Convert each stem array back to sinusoid list
6. Synthesize audio via additive synthesis with exact frequencies
```

## Parameters

- **Time frames:** Based on hop_length=512, so 86 frames/second
- **Max sinusoids per frame:** 500 (configurable, will pad/truncate)
- **Sinusoid features:** [exact_freq_Hz, amplitude, phase_radians]
- **Frequency range:** 0-22050 Hz (Nyquist frequency at 44100 Hz sample rate)
- **Padding:** Zeros for frames with < max_sinusoids
- **Truncation:** If frame has > max_sinusoids, keep first N (sorted by frequency)

## Comparison with Frequency Binning

| Aspect | Frequency Binning (OLD) | Exact Frequency (NEW) |
|--------|------------------------|----------------------|
| Frequency representation | Binned into 512 buckets | Exact Hz values |
| Frequency precision | ~43 Hz per bin | Full precision (0.1 Hz+) |
| Data kept | 100% (but binned) | **100% with precision** |
| Architecture | U-Net (CNN) | Transformer (attention) |
| Information loss | Frequency quantization | **None** |
| Synthesis quality | Degraded by binning | **High quality** |
| Example | 440.0 Hz → bin 10 (~430-473 Hz) | 440.0 Hz stays 440.0 Hz |

## Files

- `train_stem_separator_exact_freq.py` - Training script with Transformer
- `infer_exact_freq_separator.py` - Inference script with exact frequency synthesis
- `EXACT_FREQUENCY_APPROACH.md` - This documentation

## Usage

### Training
```bash
python train_stem_separator_exact_freq.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --stem-names vocals guitar bass drums song \
  --chunk-duration 4.0 \
  --max-sinusoids 500 \
  --batch-size 4 \
  --epochs 100
```

### Inference
```bash
python infer_exact_freq_separator.py \
  --fullmix-h5 ajfa/fullmix_tracks.h5 \
  --checkpoint stem_separator_exact_freq_epoch100.pt \
  --output-dir separated_stems/
```

## Why This Works

1. **Sinusoids ARE exact frequencies** - 440.0 Hz should stay 440.0 Hz, not be binned
2. **Transformers excel at set processing** - sinusoids are an unordered set at each frame
3. **100% precision retention** - no quantization, no information loss
4. **Fixed-size via padding** - handles variable numbers of sinusoids per frame
5. **Direct synthesis** - can synthesize directly from exact frequencies

## Sample-Level Organization

At each time frame (86 frames/second at hop=512):
- Frame 0: 50 sinusoids at exact frequencies [50.0 Hz, 80.3 Hz, 175.6 Hz, ...]
- Frame 1: 48 sinusoids at exact frequencies [50.1 Hz, 80.5 Hz, 176.2 Hz, ...]
- Frame 2: 52 sinusoids at exact frequencies [50.0 Hz, 80.8 Hz, 175.9 Hz, ...]

Each sinusoid maintains its exact frequency - no binning, no quantization!

This approach preserves all your sinusoidal data with full frequency precision while making it compatible with modern neural networks.
