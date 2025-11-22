# Memory Optimization Guide for Exact Frequency Stem Separator

## The Memory Problem

When training with exact frequencies, memory usage scales with:
- **Number of frames**: More frames = more memory
- **Max sinusoids per frame**: More sinusoids = more memory
- **Batch size**: Larger batches = more memory

Example: With 344 frames, 999999 max sinusoids, batch size 4:
```
Memory = 344 frames × 999999 sinusoids × 3 features × 4 bytes (float32) × 5 stems × 4 batch
       = 19.2 GB per batch!
```

This is unsustainable. Here's how to fix it.

## Step 1: Find Your Actual Max Sinusoids

First, inspect your data to find the ACTUAL maximum sinusoids per frame (not the worst-case guess):

```bash
python inspect_max_sinusoids.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --stem-names vocals guitar bass drums song
```

This will show you:
- Maximum sinusoids per frame across all files
- 95th and 99th percentile values
- Memory estimates for different configurations

**Example output:**
```
Overall maximum sinusoids per frame: 4,523

Recommended --max-sinusoids values:
  Conservative (99th percentile): 3,200
  Safe (covers max):              4,523
  Memory-efficient:               2,000 (may truncate some frames)

Memory estimates (float32, per sample in batch):
  500ms ( 43 frames),  2,000 sinusoids:    0.5 MB fullmix,    2.5 MB stems
  1000ms ( 86 frames),  2,000 sinusoids:    1.0 MB fullmix,    5.0 MB stems
  2000ms (172 frames),  2,000 sinusoids:    2.0 MB fullmix,   10.0 MB stems
  4000ms (344 frames),  2,000 sinusoids:    4.0 MB fullmix,   20.0 MB stems
```

## Step 2: Choose Memory-Efficient Settings

Based on your inspection results, choose settings that fit in memory:

### Option A: Reduce Chunk Duration (Recommended)
Process shorter audio chunks (0.5-1 second instead of 4 seconds):

```bash
python train_stem_separator_exact_freq.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --chunk-duration 1.0 \              # 1 second chunks (86 frames)
  --max-sinusoids 2000 \              # From inspection
  --batch-size 1 \                     # Small batch
  --gradient-accumulation-steps 4      # Effective batch = 4
```

**Memory:** ~6 MB per sample (1 MB fullmix + 5 MB stems) × 1 batch = **6 MB**

### Option B: Use Gradient Accumulation
Keep batch size at 1, but accumulate gradients to simulate larger batches:

```bash
python train_stem_separator_exact_freq.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --chunk-duration 1.0 \
  --max-sinusoids 2000 \
  --batch-size 1 \                     # Actual batch size
  --gradient-accumulation-steps 8      # Effective batch = 8
```

This gives you the training benefits of batch size 8 with the memory footprint of batch size 1.

### Option C: Mixed Precision Training
Use fp16 instead of fp32 to halve memory usage:

```bash
python train_stem_separator_exact_freq.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --chunk-duration 1.0 \
  --max-sinusoids 2000 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --mixed-precision                    # Use fp16
```

**Memory:** ~3 MB per sample (half of fp32)

### Option D: Combination (Maximum Memory Efficiency)
Combine all optimizations:

```bash
python train_stem_separator_exact_freq.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --chunk-duration 0.5 \               # Very short chunks (43 frames)
  --max-sinusoids 2000 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --mixed-precision
```

**Memory:** ~1.5 MB per sample

## Step 3: Monitor Memory Usage

During training, the script will show estimated memory usage:

```
Memory optimizations:
  Chunk duration: 1.0s (86 frames)
  Batch size: 1
  Gradient accumulation: 4 steps
  Effective batch size: 4
  Mixed precision: True

Estimated memory per sample:
  Fullmix: 1.0 MB
  All stems: 5.0 MB
  Per batch: 6.0 MB
```

## Memory Comparison Table

| Chunk Duration | Frames | Max Sinusoids | Batch | Memory/Sample | Memory/Batch (bs=1) |
|----------------|--------|---------------|-------|---------------|---------------------|
| 4.0s (OLD)     | 344    | 999999        | 4     | 4,800 MB      | 19,200 MB           |
| 4.0s           | 344    | 2000          | 4     | 9.6 MB        | 38.4 MB             |
| 2.0s           | 172    | 2000          | 4     | 4.8 MB        | 19.2 MB             |
| 1.0s (NEW)     | 86     | 2000          | 1     | 6.0 MB        | 6.0 MB              |
| 0.5s           | 43     | 2000          | 1     | 3.0 MB        | 3.0 MB              |

**With mixed precision (fp16), divide all values by 2**

## Best Practices

1. **Always run `inspect_max_sinusoids.py` first** to find your actual data requirements
2. **Start with short chunks** (0.5-1 second) and increase if memory allows
3. **Use gradient accumulation** instead of large batch sizes
4. **Enable mixed precision** if you have a modern GPU (Volta/Turing/Ampere)
5. **Set max_sinusoids** to 99th percentile from inspection (some truncation OK)
6. **Monitor GPU memory** with `nvidia-smi` or `watch -n 1 nvidia-smi`

## Troubleshooting

### Error: "Unable to allocate X GiB for an array"
**Solution:** Reduce one or more of:
- `--chunk-duration` (e.g., from 1.0 to 0.5)
- `--max-sinusoids` (e.g., from 2000 to 1500)
- `--batch-size` (e.g., from 1 to 1... already at minimum!)

### Training is too slow
**Solution:** Increase effective batch size without increasing memory:
```bash
--batch-size 1 \
--gradient-accumulation-steps 16  # Larger effective batch
```

### GPU out of memory during forward pass
**Solution:** Enable mixed precision:
```bash
--mixed-precision
```

### Want to use all sinusoids (no truncation)
**Solution:** Set `--max-sinusoids` to the actual maximum from `inspect_max_sinusoids.py`, then reduce `--chunk-duration` to fit in memory.

## Example Commands

### For 16 GB GPU
```bash
python train_stem_separator_exact_freq.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --chunk-duration 1.0 \
  --max-sinusoids 3000 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --mixed-precision \
  --epochs 100
```

### For 8 GB GPU
```bash
python train_stem_separator_exact_freq.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --chunk-duration 0.5 \
  --max-sinusoids 2000 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --mixed-precision \
  --epochs 100
```

### For CPU (lots of RAM)
```bash
python train_stem_separator_exact_freq.py \
  --data-dirs ajfa/ blackned/ dyerseve/ \
  --chunk-duration 2.0 \
  --max-sinusoids 5000 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --epochs 100
```

## Key Insight

You can keep **all sinusoids** (no truncation) by:
1. Finding actual max from `inspect_max_sinusoids.py` (e.g., 4,523 instead of 999,999)
2. Reducing chunk duration to fit in memory (e.g., 0.5s instead of 4s)
3. Using gradient accumulation for effective large batch size

This way you get:
- ✅ 100% of sinusoidal data (no truncation)
- ✅ Exact frequencies preserved (no binning)
- ✅ Fits in GPU memory
- ✅ Efficient training via gradient accumulation
