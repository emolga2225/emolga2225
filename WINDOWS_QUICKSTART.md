# Windows Quick Start Guide - NES Stem Separator

This guide helps you run the NES training pipeline on Windows.

## Prerequisites

1. **Python 3.8 or higher**
   - Download from: https://www.python.org/downloads/
   - **IMPORTANT**: Check "Add Python to PATH" during installation

2. **Git** (if cloning the repository)
   - Download from: https://git-scm.com/download/win

## Setup

### Step 1: Get the code

If you haven't already, clone the repository:
```cmd
git clone <your-repo-url>
cd emolga2225
git checkout claude/train-stem-separator-01292LDfNmNgMeCCMa3AwLFw
```

Or if you already have the code, just pull the latest changes:
```cmd
git pull origin claude/train-stem-separator-01292LDfNmNgMeCCMa3AwLFw
```

### Step 2: Install dependencies

**Option A: Using the batch file (easiest)**
```cmd
setup_windows.bat
```

**Option B: Manual installation**
```cmd
pip install torch numpy scipy h5py tqdm soundfile game-music-emu
```

## Usage

### Prepare Training Data from NSF

**Option A: Using the batch file (easiest)**
```cmd
prepare_nes_windows.bat your_game.nsf 120
```

**Option B: Direct Python command**
```cmd
python prepare_nes_training_data.py your_game.nsf --duration 120
```

### What This Does

1. Renders 5 WAV files from your NSF:
   - `pulse1.wav` - Pulse wave channel 1 (melody)
   - `pulse2.wav` - Pulse wave channel 2 (harmony)
   - `triangle.wav` - Triangle wave (bass)
   - `noise.wav` - Noise channel (drums)
   - `mix.wav` - All channels combined

2. Extracts sinusoidal features to HDF5 format

3. Creates training chunks with stem mapping:
   - pulse1 → "vocals" (melody)
   - pulse2 → "guitar" (harmony)
   - triangle → "bass" (bass)
   - noise → "drums" (percussion)
   - mix → "song" (full mix)

### Output Location

All output goes to: `nsf_data\<game_name>\`
```
nsf_data\
  your_game\
    wav\          <- Rendered WAV files
    h5\           <- HDF5 sinusoidal features
    chunks\       <- Training data (use this for training)
```

### Train the Model

After preparing data:
```cmd
python train_stem_separator_fast.py ^
    --data-file nsf_data\your_game\chunks\your_game_chunks.h5 ^
    --max-sinusoids 900 ^
    --batch-size 32 ^
    --epochs 100 ^
    --save-every 10
```

**Note**: Use `^` for line continuation in Windows CMD, or just put it all on one line.

### Test the Model

After training:
```cmd
python test_stem_separator.py ^
    --checkpoint stem_separator_exact_freq_epoch50.pt ^
    --h5-file nsf_data\your_game\h5\your_game_mix.h5 ^
    --max-sinusoids 900 ^
    --output-dir separated_nes\
```

Then synthesize to audio:
```cmd
python batch_synthesize.py ^
    --input-dir separated_nes\ ^
    --output-dir nes_audio\ ^
    --stems vocals guitar bass drums
```

## Expected Results with NES Data

- **Much faster training**: Loss should drop below 5 in 50-100 epochs (vs 390+ on complex music)
- **Easy verification**: Only 5-15 sinusoids per channel (vs hundreds)
- **Clear separation**: Simple waveforms make it obvious if stem conditioning works

If stems are still identical on NES data, we know there's a fundamental issue with stem conditioning. If it works, we can scale up to more complex music.

## Troubleshooting

### "Python is not recognized"
- Reinstall Python and check "Add Python to PATH"
- Or manually add Python to PATH in System Environment Variables

### "pip is not recognized"
- Python installation should include pip
- Try: `python -m pip install <package>`

### "torch installation failed"
- The setup script tries GPU version first, then falls back to CPU
- For GPU support, make sure you have CUDA installed
- CPU version works fine for testing

### Unicode errors in output
- The Python scripts use ASCII output for Windows compatibility
- All checkmarks and emojis are replaced with [OK], [DONE], etc.

## File Locations

Put your NSF files anywhere convenient:
- Same directory as scripts: `your_game.nsf`
- Subdirectory: `nsf_files\your_game.nsf`
- Anywhere: `C:\Users\YourName\Music\NES\your_game.nsf`

Just use the path in the command:
```cmd
python prepare_nes_training_data.py C:\Users\YourName\Music\NES\your_game.nsf
```

## Support

If you encounter issues, check:
1. Python version: `python --version` (should be 3.8+)
2. Dependencies installed: `pip list | findstr "torch numpy h5py"`
3. NSF file exists and is valid
