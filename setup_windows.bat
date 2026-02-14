@echo off
REM Setup script for Windows
REM This installs all dependencies needed for NES training data preparation

echo ============================================================
echo NES Stem Separator - Windows Setup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

echo Installing dependencies...
echo.

REM Install requirements
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo WARNING: PyTorch installation failed, trying CPU version...
    pip install torch>=2.0.0
)

pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install h5py>=3.8.0
pip install tqdm>=4.65.0
pip install soundfile>=0.12.0
pip install game-music-emu>=0.6.3

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo You can now prepare NES training data:
echo   python prepare_nes_training_data.py your_game.nsf --duration 120
echo.
pause
