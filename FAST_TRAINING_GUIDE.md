# Fast Training with Preprocessed Data

## The Problem

Training directly from HDF5 files is **very slow** (~13 seconds per iteration):
- Load HDF5 file (slow I/O)
- Parse track data structure
- Convert to frame-based arrays
- Repeat for EVERY frame, EVERY epoch

With 122,786 frames:
- **1 epoch**: 13s × 15,349 iterations ≈ **55 hours!**
- **100 epochs**: ~230 days! 😱

## The Solution: Preprocessing

Convert HDF5 files to numpy arrays **once**, then load pre-processed data during training.

**Speed improvement:** 13s/iter → <1s/iter (~10-20× faster!)

## Two-Step Workflow

### Step 1: Preprocess Data (Run Once)

Convert all HDF5 files to compressed numpy arrays:

```bash
python preprocess_hdf5_to_arrays.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --output-dir preprocessed_data/ \
  --max-sinusoids 2000 \
  --stem-names vocals guitar bass drums song
```

**What this does:**
- Reads each HDF5 file once
- For each frame: extracts sinusoids and converts to exact frequency array
- Saves as compressed `.npz` files (one per frame)
- Creates `index.json` with frame metadata

**Time:** ~10-30 minutes (one-time cost)

**Output structure:**
```
preprocessed_data/
├── index.json
├── ajfa/
│   ├── frame_000000.npz
│   ├── frame_000001.npz
│   └── ...
├── blackned/
│   ├── frame_000000.npz
│   └── ...
└── dyerseve/
    ├── frame_000000.npz
    └── ...
```

Each `.npz` file contains:
- `fullmix`: (max_sinusoids, 3) array with [freq, amp, phase]
- `stems`: (n_stems, max_sinusoids, 3) array with stem sinusoids

### Step 2: Train Fast

Train using preprocessed data:

```bash
python train_stem_separator_fast.py \
  --preprocessed-dir preprocessed_data/ \
  --batch-size 32 \
  --num-workers 4 \
  --mixed-precision \
  --epochs 100
```

**Speed improvements:**
- Load `.npz` files: ~100× faster than HDF5 processing
- Can use multiple DataLoader workers (num-workers=4)
- Can use larger batch sizes (32 vs 8)
- Mixed precision for even more speed

**Expected speed:**
- **Before:** 13s/iter → 55 hours/epoch
- **After:** <1s/iter → ~4-5 hours/epoch

## Disk Space

Preprocessing creates compressed `.npz` files.

**Estimate:** ~0.1-0.5 MB per frame (depending on compression)

With 122,786 frames:
- **Total size:** ~12-60 GB

**Tip:** Use `--max-sinusoids 2000` (not 5000) to reduce file sizes.

## Benefits

1. **Training speed:** 10-20× faster iterations
2. **Multiple workers:** Can use num-workers > 0 (HDF5 doesn't support this)
3. **Larger batches:** More efficient GPU utilization
4. **Cleaner code:** Simple numpy loading vs complex HDF5 parsing
5. **Reproducibility:** Fixed preprocessed data ensures consistent training

## Comparison

| Method | Speed/Iter | Epoch Time | Workers | Batch Size |
|--------|------------|------------|---------|------------|
| **Direct HDF5** | 13s | 55 hours | 0 (not supported) | 8 |
| **Preprocessed** | <1s | 4-5 hours | 4 | 32 |

**Speedup:** ~10-15× faster!

## When to Repreprocess

You need to rerun preprocessing if you:
- Change `--max-sinusoids` value
- Add new data directories
- Modify the HDF5 files
- Change which stems to use

Otherwise, you can train multiple times using the same preprocessed data.

## Advanced: Preprocessing on Different Machine

If you have a powerful CPU machine, you can preprocess there:

```bash
# On CPU machine (lots of RAM)
python preprocess_hdf5_to_arrays.py \
  --data-dirs /path/to/data/ \
  --output-dir preprocessed_data/ \
  --max-sinusoids 2000

# Copy preprocessed_data/ to GPU machine
rsync -av preprocessed_data/ gpu-machine:/path/to/preprocessed_data/

# Train on GPU machine
python train_stem_separator_fast.py \
  --preprocessed-dir preprocessed_data/ \
  --batch-size 32 \
  --mixed-precision
```

## Memory Usage

Preprocessing loads one frame at a time, so memory usage is low (~few hundred MB).

Fast training loads batches of preprocessed frames, which is also efficient:
- Batch size 32: ~32 MB per batch
- With num-workers=4: ~128 MB for prefetching

Much less memory than processing HDF5 on-the-fly!

## Complete Workflow Example

```bash
# Step 1: Preprocess (one time, ~30 mins)
python preprocess_hdf5_to_arrays.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --output-dir preprocessed_data/

# Step 2: Train fast (can run multiple times)
python train_stem_separator_fast.py \
  --preprocessed-dir preprocessed_data/ \
  --batch-size 32 \
  --num-workers 4 \
  --mixed-precision \
  --epochs 100

# Continue training from checkpoint
python train_stem_separator_fast.py \
  --preprocessed-dir preprocessed_data/ \
  --checkpoint stem_separator_exact_freq_epoch50.pt \
  --epochs 100
```

## Summary

**Problem:** 13s/iter → 55 hours/epoch (unusable)
**Solution:** Preprocess once, train fast
**Result:** <1s/iter → 4-5 hours/epoch (practical!)

The preprocessing step takes ~30 minutes but saves hundreds of hours of training time.
