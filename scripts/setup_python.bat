@echo off
REM ================================================================
REM Watheq Project — Python Dependencies Setup
REM Auto-detects GPU (NVIDIA CUDA) and installs the correct PyTorch.
REM Works on any device: GPU or CPU-only.
REM
REM Usage:  scripts\setup_python.bat
REM         scripts\setup_python.bat --cpu     (force CPU-only)
REM         scripts\setup_python.bat --gpu     (force GPU/CUDA)
REM ================================================================
echo.
echo ========================================
echo   Watheq — Python Setup
echo ========================================
echo.

cd /d "%~dp0.."

REM ── Step 1: Install regular requirements ──
echo [1/3] Installing base Python dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install base requirements.
    pause
    exit /b 1
)
echo      Base dependencies installed.
echo.

REM ── Step 2: Detect GPU or use forced mode ──
set "TORCH_MODE=auto"
if /i "%~1"=="--cpu" set "TORCH_MODE=cpu"
if /i "%~1"=="--gpu" set "TORCH_MODE=gpu"

set "USE_CUDA=0"

if "%TORCH_MODE%"=="cpu" (
    echo [2/3] Forced CPU mode — installing PyTorch CPU...
    set "USE_CUDA=0"
    goto :install_torch
)

if "%TORCH_MODE%"=="gpu" (
    echo [2/3] Forced GPU mode — installing PyTorch CUDA...
    set "USE_CUDA=1"
    goto :install_torch
)

REM Auto-detect: check for nvidia-smi
echo [2/3] Detecting GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo      No NVIDIA GPU detected — installing PyTorch CPU version.
    echo      ^(This is fine, training will just be slower.^)
    set "USE_CUDA=0"
) else (
    echo      NVIDIA GPU detected — installing PyTorch with CUDA support.
    echo      ^(Training will use GPU acceleration.^)
    set "USE_CUDA=1"
)

:install_torch
echo.

if "%USE_CUDA%"=="1" (
    echo      Installing torch + torchvision with CUDA 12.4 support...
    echo      ^(Download ~2.5 GB — this is a one-time download.^)
    echo.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
) else (
    echo      Installing torch + torchvision CPU-only...
    echo      ^(Download ~200 MB^)
    echo.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install PyTorch.
    echo If you have network issues, try running this script again.
    pause
    exit /b 1
)
echo      PyTorch installed successfully.
echo.

REM ── Step 3: Verify installation ──
echo [3/3] Verifying installation...
echo.
python -c "import torch; cuda = torch.cuda.is_available(); gpu_name = torch.cuda.get_device_name(0) if cuda else 'N/A'; print(f'  PyTorch {torch.__version__}'); print(f'  CUDA available: {cuda}'); print(f'  GPU: {gpu_name}')"

if errorlevel 1 (
    echo.
    echo WARNING: PyTorch verification failed. Try reinstalling.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup complete! 
echo   Run:  scripts\start_all.bat
echo ========================================
echo.
pause
