@echo off
REM Convenience wrapper for Windows users without `make`
REM Always execute from the repo root (this file lives in /scripts)
setlocal
pushd "%~dp0\.."
  REM Avoid uv warning about mismatched active venv vs project .venv
  set "VIRTUAL_ENV="
  set "CONDA_PREFIX="
  uv run pytest -q --cov=fast_stt --cov-report=term-missing --cov-report=xml
popd
endlocal
