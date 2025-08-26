# Fast-STT

Extremely fast speech-to-text with very low error-rate. Fully local, no internet required (after installation).

## Install & Run

### From source (UV-native)

1. Install **ffmpeg** and ensure `ffmpeg` is on PATH.  
2. Double-click **`setup.bat`**. It will:
   - create `.venv`, install the right PyTorch (CUDA if GPU present, else CPU),
   - `uv sync` all dependencies from `pyproject.toml`, and
   - pre-download the model to `models/parakeet-tdt-0.6b-v2.nemo`.  
     *(No progress bar — the download is several gigabytes, be patient.)*
3. When setup.bat finishes, you can double-click **`run.bat`** or run `fast-stt` to start the app.

### Environment variables (optional)

- `PARAKEET_MODEL_NAME` — override HF model id (default `nvidia/parakeet-tdt-0.6b-v2`)  
- `PARAKEET_MODEL_PATH` — load/save local `.nemo` (default `models/parakeet-tdt-0.6b-v2.nemo`)  
- `FAST_STT_HOST` / `FAST_STT_PORT` — set host/port (default `127.0.0.1:7860`)  
- `FAST_STT_SHARE=1` — enable Gradio share link  
- `FAST_STT_CORS=1` and `FAST_STT_CORS_ORIGINS=http://localhost:3000,https://your.app` to enable CORS for API clients  
- `FAST_STT_OPEN_BROWSER=0` — disable auto-opening the browser  

---

## Developer Quick-Start

Install dev tools:

```bash
uv pip install -e ".[dev]"
````

Common tasks:

```bash
# Run tests with coverage
scripts\test.bat
# or explicitly from repo root:
# uv run pytest -q --cov=fast_stt --cov-report=term-missing --cov-report=xml

# Lint / format / typecheck
uv run ruff check .
uv run ruff format .
uv run mypy src

# Optional: install VAD support (quiet Pylance warning)
uv pip install '.[vad]'
```

*(The old `Makefile` is no longer required — everything runs directly via `uv` and batch scripts.)*

---

## REST API

Fast-STT exposes a FastAPI under `/api` alongside the UI.

### Health

```bash
curl -s http://127.0.0.1:7860/api/health | jq
```

Response:

```json
{ "status":"ok", "version":"0.1.0", "model_ready":true, "device":"cuda", "ffmpeg":true }
```

### Warmup

Trigger background model preload after deploy / on cold start:

```bash
curl -s http://127.0.0.1:7860/api/warmup
```

Response:

```json
{ "status":"ok", "started": true, "ready": false }
```

### Transcribe (multipart upload)

```bash
curl -s -X POST http://127.0.0.1:7860/api/transcribe \
  -F "file=@path/to/audio.mp3" > out.json
```

Response fields:

* `device`: `"cuda"` or `"cpu"`
* `segments`: `[{ "start": 0.12, "end": 2.34, "segment": "..." }, ...]`
* `text`: full transcript
* `srt`: SRT content
* `md`: Markdown bullets

---

## Future

My personal goal is to create a responsive alternative to Nuance Dragon using a complete redesign. Just wanted to share this project now.


