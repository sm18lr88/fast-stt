@echo off
setlocal enabledelayedexpansion
title fast-stt Setup
REM Ensure we run from the project root even if launched elsewhere
pushd "%~dp0"
REM Avoid uv warnings about an unrelated active venv
set "VIRTUAL_ENV="
set "CONDA_PREFIX="

REM ---- User-tunable ----
set PYTHON_VERSION=3.12
set TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128
set UV_LINK_MODE=copy

echo === Preflight: ensuring Python and uv are available ===
where py >nul 2>nul || (
  echo [ERROR] Python launcher "py" not found. Install Python 3.%PYTHON_VERSION%+ from python.org and re-run.
  exit /b 1
)

where uv >nul 2>nul || (
  echo [info] uv not found; installing with pip...
  py -3 -m pip install --upgrade pip >nul 2>nul
  py -3 -m pip install -U uv || (
    echo [ERROR] Failed to install uv via pip.
    exit /b 1
  )
)
for /f "usebackq tokens=*" %%i in (`uv --version`) do set UV_VER=%%i
echo [ok] uv: !UV_VER!

 echo === Checking ffmpeg on PATH ===
set "FFMPEG="
for /f "delims=" %%F in ('where ffmpeg 2^>nul') do set "FFMPEG=%%F"
if not defined FFMPEG (
  echo [ERROR] ffmpeg not found on PATH. Install it and re-run.
  exit /b 1
)
"%FFMPEG%" -version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] ffmpeg was found at "%FFMPEG%" but is not callable.
  exit /b 1
)
echo [ok] ffmpeg: "%FFMPEG%"

echo === GPU / CUDA detection ===
set "HAS_NVIDIA=0"
set "GPU_NAME="
set "DRIVER_VER="

where nvidia-smi >nul 2>nul && set "HAS_NVIDIA=1"
if "!HAS_NVIDIA!"=="1" (
  rem Try modern query for GPU name (ignore driver to avoid old CLI errors)
  for /f "usebackq delims=" %%i in (`nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul`) do (
    if not defined GPU_NAME set "GPU_NAME=%%i"
  )
  rem Fallback to legacy listing
  if not defined GPU_NAME (
    for /f "usebackq delims=" %%i in (`nvidia-smi -L 2^>nul`) do (
      if not defined GPU_NAME set "GPU_NAME=%%i"
    )
  )
  rem Best-effort parse of driver version from plain output
  for /f "usebackq tokens=2 delims=:" %%i in (`nvidia-smi ^| findstr /C:"Driver Version"`) do (
    set "DRIVER_VER=%%i"
  )
  rem trim leading spaces
  for /f "tokens=* delims= " %%j in ("!DRIVER_VER!") do set "DRIVER_VER=%%j"
  if not defined GPU_NAME set "GPU_NAME=NVIDIA GPU"
  if not defined DRIVER_VER (
    echo [gpu] !GPU_NAME!
  ) else (
    echo [gpu] !GPU_NAME!  driver=!DRIVER_VER!
  )
) else (
  echo [cpu] NVIDIA GPU not detected; will install CPU PyTorch.
)

echo === Checking CUDA Toolkit (optional) ===
set "HAS_CUDA_TOOLKIT=0"
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "HAS_CUDA_TOOLKIT=1"
if "!HAS_CUDA_TOOLKIT!"=="0" if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" set "HAS_CUDA_TOOLKIT=1"
where nvcc >nul 2>nul && set "HAS_CUDA_TOOLKIT=1"
if "!HAS_CUDA_TOOLKIT!"=="1" (
  echo [ok] CUDA Toolkit detected.
) else (
  echo [info] CUDA Toolkit not found; not required for PyTorch wheels.
)

echo === Creating fresh uv virtual environment (.venv) ===
if exist .venv (
  echo [info] Removing existing .venv ...
  rmdir /s /q .venv
)
uv venv --python=%PYTHON_VERSION% || (
  echo [ERROR] Failed to create virtual environment.
  exit /b 1
)

call .venv\Scripts\activate.bat

echo === Installing PyTorch (wheel selection based on GPU presence) ===
if "!HAS_NVIDIA!"=="1" (
  uv pip install --index-url %TORCH_CUDA_INDEX% torch torchvision torchaudio || (
    echo [ERROR] Failed to install CUDA-enabled PyTorch.
    exit /b 1
  )
  echo === Installing CUDA Python bindings (optional speedups) ===
  uv pip install "cuda-python>=12.3" || echo [warn] cuda-python install skipped.
) else (
  uv pip install torch torchvision torchaudio || (
    echo [ERROR] Failed to install CPU PyTorch.
    exit /b 1
  )
)

echo === Syncing project dependencies from pyproject.toml (uv) ===
uv sync || (
  echo [ERROR] uv sync failed.
  exit /b 1
)
 
echo === Prefetching model to user cache (first run only) ===
uv run python -m fast_stt.cli_download || echo [warn] Could not pre-download model; it will download on first run.

echo === Writing run.bat ===
 >run.bat (
   echo @echo off
   echo call .venv\Scripts\activate.bat
   echo fast-stt
 )

echo.
echo [DONE] Setup complete.
echo - Double-click run.bat to launch the app.
popd
exit /b 0
