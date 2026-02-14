@echo off
REM Windows batch wrapper for prepare_nes_training_data.py
REM Usage: prepare_nes_windows.bat game.nsf [duration_in_seconds]

setlocal

if "%~1"=="" (
    echo Usage: prepare_nes_windows.bat game.nsf [duration]
    echo.
    echo Examples:
    echo   prepare_nes_windows.bat metroid.nsf
    echo   prepare_nes_windows.bat megaman2.nsf 180
    echo.
    echo Default duration: 120 seconds
    pause
    exit /b 1
)

set NSF_FILE=%~1
set DURATION=%~2

if "%DURATION%"=="" set DURATION=120

if not exist "%NSF_FILE%" (
    echo ERROR: NSF file not found: %NSF_FILE%
    pause
    exit /b 1
)

echo ============================================================
echo NES Training Data Pipeline
echo ============================================================
echo NSF file: %NSF_FILE%
echo Duration: %DURATION% seconds
echo ============================================================
echo.

python prepare_nes_training_data.py "%NSF_FILE%" --duration %DURATION%

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pipeline failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! Training data is ready
echo ============================================================
echo.
echo Next step - Train the model:
echo   python train_stem_separator_fast.py --help
echo.
pause
