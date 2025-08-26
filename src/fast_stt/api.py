from __future__ import annotations
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
import tempfile, shutil, os
from .engine import FastSTTEngine
from . import __version__

router = APIRouter()
ENGINE = FastSTTEngine()  # API-only engine; no Gradio imports


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = Field(default=__version__)
    model_ready: bool
    device: str
    ffmpeg: bool


class Segment(BaseModel):
    start: float
    end: float
    segment: str


class TranscribeResponse(BaseModel):
    device: str
    segments: list[Segment]
    text: str
    srt: str
    md: str


@router.get("/health", response_model=HealthResponse)
def health():
    try:
        import shutil as _sh

        ff = _sh.which("ffmpeg")
    except Exception:
        ff = None
    ready = ENGINE.preload_done.is_set()
    device = ENGINE.device or "unknown"
    return HealthResponse(model_ready=ready, device=device, ffmpeg=bool(ff))


@router.get("/warmup")
def warmup():
    """
    Kick off background preload and report readiness.
    Useful for orchestrators to prime the model after deploy.
    """
    ENGINE.start_preload()
    return {"status": "ok", "started": True, "ready": ENGINE.preload_done.is_set()}


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    device: str = Form("auto"),
    vad: int = Form(0),
    chunking: int = Form(1),
    compile: int = Form(0),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    suffix = Path(file.filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        try:
            # Local-first app: no upload size cap; stream to temp file
            shutil.copyfileobj(file.file, tmp)
        finally:
            await file.close()
    try:
        # lazy-load with current toggles
        ENGINE.load_model(device=device, compile=bool(compile), chunking=bool(chunking))
        result = ENGINE.transcribe_file(
            str(tmp_path),
            device=device,
            vad=bool(vad),
            chunking=bool(chunking),
            compile=bool(compile),
        )
        return TranscribeResponse(**result)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# convenience: kick off preload when the API router is imported (server boot)
ENGINE.start_preload()
