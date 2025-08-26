from __future__ import annotations
import os
from pathlib import Path
import platform


def ensure_cuda_on_path() -> None:
    """
    Prepend repo-local / torch-bundled CUDA runtime dirs to PATH so torch/NeMo
    can load nvidia* and cuda* DLLs/SOs when present.
    Windows: prefer <venv>\Lib\site-packages\torch\lib
    Linux:   <pkg>/cuda/lib64  (not used on Windows target but kept for parity)
    """
    root = Path(__file__).resolve().parents[2]  # .../src/fast_stt -> repo root
    if platform.system() == "Windows":
        try:
            import torch  # type: ignore

            torch_lib = Path(torch.__file__).resolve().parent / "lib"
        except Exception:
            torch_lib = None
        candidates = []
        if torch_lib and torch_lib.is_dir():
            candidates.append(torch_lib)
        candidates += [root / "cuda" / "bin" / "x64", root / "cuda" / "bin"]
        sep = os.pathsep
        path = os.environ.get("PATH", "")
        existing = {p.lower() for p in path.split(sep) if p}
        inserts = [
            str(p) for p in candidates if p.is_dir() and str(p).lower() not in existing
        ]
        if inserts:
            os.environ["PATH"] = sep.join(inserts + [path])
    else:
        # Linux (not primary target here)
        candidates = [root / "cuda" / "lib64", root / "cuda" / "lib"]
        sep = os.pathsep
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        existing = {p for p in ld.split(sep) if p}
        inserts = [str(p) for p in candidates if p.is_dir() and str(p) not in existing]
        if inserts:
            os.environ["LD_LIBRARY_PATH"] = sep.join(inserts + [ld])
            try:
                import ctypes  # noqa: F401

                ctypes.CDLL(None)
            except Exception:
                pass


__all__ = ["ensure_cuda_on_path"]
