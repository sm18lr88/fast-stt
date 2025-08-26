# pyright: reportMissingImports=false
from __future__ import annotations
from pathlib import Path
import os, threading, wave, struct, logging, tempfile
from typing import Any, List, Tuple
from pydub import AudioSegment
from .paths import get_model_file, get_gradio_tmp_dir
from .utils.cuda_loader import ensure_cuda_on_path
from .formatting import generate_md_content, generate_srt_content, generate_txt_content

_LOG = logging.getLogger("fast_stt.engine")
if not _LOG.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _ensure_ffmpeg() -> None:
    from shutil import which

    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH. Please install ffmpeg.")


def maybe_apply_vad(audio: AudioSegment) -> AudioSegment:
    """Optional VAD using webrtcvad if installed; otherwise returns original."""
    try:
        import importlib

        spec = importlib.util.find_spec("webrtcvad")
        if spec is None:
            return audio
        import webrtcvad  # type: ignore  # optional dependency; silence Pylance/mypy
    except Exception:
        return audio
    # mono 16k 16-bit
    a = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    vad = webrtcvad.Vad(1)
    frame_bytes = int(0.03 * 16000) * 2  # 30ms * 2 bytes/sample
    raw = a.raw_data
    voiced = bytearray()
    for i in range(0, len(raw), frame_bytes):
        frame = raw[i : i + frame_bytes]
        if len(frame) < frame_bytes:
            break
        if vad.is_speech(frame, 16000):
            voiced += frame
    if not voiced:
        return a
    try:
        return AudioSegment(
            data=bytes(voiced), sample_width=2, frame_rate=16000, channels=1
        )
    except Exception:
        return a


class FastSTTEngine:
    def __init__(
        self, model_name: str | None = None, model_path: str | None = None
    ) -> None:
        # Ensure CUDA DLLs on PATH before touching torch/NeMo
        try:
            ensure_cuda_on_path()
        except Exception as _e:
            _LOG.debug("CUDA DLL preload skipped: %s", _e)
        self.model_name = model_name or os.environ.get(
            "PARAKEET_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2"
        )
        self.model_file = Path(
            model_path
            or os.environ.get(
                "PARAKEET_MODEL_PATH", str(get_model_file(self.model_name))
            )
        ).resolve()
        self.tmp_dir = get_gradio_tmp_dir()
        self._model = None
        self._device = None
        self._compiled = False
        self._lock = threading.Lock()
        self.preload_started = threading.Event()
        self.preload_done = threading.Event()

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def model_name_short(self) -> str:
        return self.model_name.split("/")[-1]

    def load_model(
        self, device: str = "auto", compile: bool = False, chunking: bool = True
    ):
        """Load model (singleton). Returns loaded torch/NeMo model and device."""
        if self._model is not None:
            # Ensure runtime toggles apply to existing model
            self._apply_runtime_toggles(
                self._model, self._device or "cpu", compile=compile, chunking=chunking
            )
            return self._model, self._device
        with self._lock:
            if self._model is not None:
                self._apply_runtime_toggles(
                    self._model,
                    self._device or "cpu",
                    compile=compile,
                    chunking=chunking,
                )
                return self._model, self._device
            _ensure_ffmpeg()
            import torch
            from nemo.collections.asr.models import ASRModel
            import gzip
            import tarfile
            import zipfile

            if device.lower() == "cuda":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            elif device.lower() == "cpu":
                dev = "cpu"
            else:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = dev
            _LOG.info("loading NeMo model on device=%s", dev)

            def _download_and_cache() -> None:
                self._model = ASRModel.from_pretrained(
                    model_name=self.model_name, map_location=dev
                )
                try:
                    self.model_file.parent.mkdir(parents=True, exist_ok=True)
                    self._model.save_to(str(self.model_file))
                except Exception as e:
                    _LOG.warning("model cache save skipped: %s", e)

            if self.model_file.exists():
                try:
                    # quick integrity check: file should be a valid zip/tar; try restore
                    # Some NeMo releases write .nemo as tar.gz; others as zip. Try both probes.
                    # Probe small header first to throw early if it's HTML/text.
                    with open(self.model_file, "rb") as fh:
                        _ = fh.read(4)
                    self._model = ASRModel.restore_from(
                        str(self.model_file), map_location=dev
                    )
                except (
                    gzip.BadGzipFile,
                    tarfile.ReadError,
                    zipfile.BadZipFile,
                    OSError,
                    RuntimeError,
                ) as e:
                    _LOG.warning(
                        "cached model at %s appears corrupted (%s); re-downloading",
                        self.model_file,
                        e,
                    )
                    try:
                        self.model_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _download_and_cache()
                except Exception:
                    # Unknown failure restoring; fall back to re-download once.
                    try:
                        self.model_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _download_and_cache()
            else:
                _download_and_cache()
            # Ensure fully frozen baseline so any partial unfreeze inside NeMo is valid
            try:
                if hasattr(self._model, "freeze"):
                    self._model.freeze()
            except Exception as e:
                _LOG.debug("freeze skipped: %s", e)
            self._model.eval()
            self._apply_runtime_toggles(
                self._model, dev, compile=compile, chunking=chunking
            )
            return self._model, self._device

    def _apply_runtime_toggles(self, model, device: str, compile: bool, chunking: bool):
        import torch

        # chunking
        try:
            model.change_subsampling_conv_chunking_factor(1 if chunking else -1)
        except Exception:
            pass
        # compile (CUDA only; only once)
        if (
            compile
            and not self._compiled
            and hasattr(torch, "compile")
            and device == "cuda"
        ):
            try:
                if hasattr(model, "encoder"):
                    model.encoder = torch.compile(model.encoder)  # type: ignore[attr-defined]
                if hasattr(model, "decoder"):
                    model.decoder = torch.compile(model.decoder)  # type: ignore[attr-defined]
                self._compiled = True
                _LOG.info("torch.compile enabled")
            except Exception as e:
                _LOG.warning("torch.compile skipped: %s", e)

    def _long_audio_ctx(self, model, duration_sec: float, chunking_default: bool):
        class _Ctx:
            def __init__(self, outer, m, d, cd):
                self.outer = outer
                self.m = m
                self.dur = d
                self.cd = cd
                self.applied = False
                self._froze = False

            def __enter__(self):
                if self.dur > 480:
                    try:
                        # Create a consistent state so NeMo's partial unfreeze paths won't error
                        if hasattr(self.m, "freeze"):
                            self.m.freeze()
                            self._froze = True
                        # Best-effort long-form tweaks; ignore if unsupported
                        try:
                            self.m.change_attention_model(
                                "rel_pos_local_attn", [256, 256]
                            )
                        except Exception:
                            pass
                        if not self.cd:
                            try:
                                self.m.change_subsampling_conv_chunking_factor(1)
                            except Exception:
                                pass
                        self.applied = True
                    except Exception:
                        self.applied = False
                return self

            def __exit__(self, exc_type, exc, tb):
                if self.applied:
                    try:
                        try:
                            self.m.change_attention_model("rel_pos")
                        except Exception:
                            pass
                        if not self.cd:
                            try:
                                self.m.change_subsampling_conv_chunking_factor(-1)
                            except Exception:
                                pass
                    finally:
                        if self._froze and hasattr(self.m, "unfreeze"):
                            try:
                                self.m.unfreeze()
                            except Exception:
                                pass

        return _Ctx(self, model, duration_sec, chunking_default)

    def maybe_empty_cuda_cache(self):
        try:
            import torch  # noqa

            if self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    def transcribe_file(
        self,
        input_path: str,
        device: str = "auto",
        vad: bool = False,
        chunking: bool = True,
        compile: bool = False,
    ) -> dict[str, Any]:
        model, dev = self.load_model(device=device, compile=compile, chunking=chunking)
        audio = AudioSegment.from_file(input_path)
        duration_sec = float(audio.duration_seconds or 0.0)
        # preprocessing
        a = audio.set_frame_rate(16000)
        a = a.set_channels(1) if a.channels != 1 else a
        if vad:
            a = maybe_apply_vad(a)
        tmp_wav = Path(self.tmp_dir) / f"api_{Path(input_path).stem}_mono16k.wav"
        a.export(tmp_wav, format="wav")
        try:
            import torch

            # Always run inference in FP32 to avoid dtype mismatch errors in some kernels.
            # Prefer higher matmul precision on CUDA (TF32) for speed without dtype changes.
            if dev == "cuda":
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            model.to(dev)
            try:
                model.float()  # ensure consistent FP32 weights/activations
            except Exception:
                pass
            with self._long_audio_ctx(model, duration_sec, chunking):
                out = model.transcribe([str(tmp_wav)], timestamps=True)
        finally:
            try:
                tmp_wav.unlink(missing_ok=True)
            except Exception:
                pass
        if (
            not out
            or not isinstance(out, list)
            or not out[0]
            or not hasattr(out[0], "timestamp")
        ):
            raise RuntimeError("Unexpected transcription output")
        segs = out[0].timestamp.get("segment", [])
        segments = [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "segment": s["segment"],
            }
            for s in segs
        ]
        text = generate_txt_content(segments)
        srt = generate_srt_content(segments)
        md = generate_md_content(segments)
        return {"device": dev, "segments": segments, "text": text, "srt": srt, "md": md}

    def _warmup(self, model, device: str):
        sr = 16000
        n = sr // 2
        tmp = Path(self.tmp_dir) / "warmup_silence.wav"
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"".join(struct.pack("<h", 0) for _ in range(n)))
        try:
            model.to(device)
            _ = model.transcribe([str(tmp)], timestamps=False)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _preload_worker(self):
        try:
            _LOG.info("[preload] starting background model load…")
            m, d = self.load_model(device="auto", compile=False, chunking=True)
            try:
                self._warmup(m, d or "cpu")
            except Exception as e:
                _LOG.debug("[preload] warmup skipped: %s", e)
            self.preload_done.set()
            _LOG.info("[preload] completed")
        except Exception as e:
            _LOG.warning("[preload] failed: %s", e)
            self.preload_done.set()

    def start_preload(self):
        if self.preload_started.is_set():
            return
        self.preload_started.set()
        threading.Thread(
            target=self._preload_worker, name="fast-stt-preload", daemon=True
        ).start()
