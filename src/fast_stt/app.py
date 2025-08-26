"""
fast_stt.app

Fast-start Gradio app for Parakeet-TDT-0.6b-v2.
- GUI renders ASAP.
- Heavy deps (torch/NeMo) are lazy-imported and preloaded in the background.
- CUDA DLLs are resolved from torch's bundled libs.
"""

from __future__ import annotations

import os, csv, datetime, tempfile, shutil, gc, logging, zipfile
from typing import Any, List, Tuple
from importlib import resources
from pathlib import Path
from pydub import AudioSegment

# Put Gradio temp/cache under the package directory to avoid Windows %TEMP% issues
_PKG_ROOT = Path(__file__).resolve().parent
from .paths import get_gradio_tmp_dir
from .formatting import (
    generate_md_content,
    generate_srt_content,
    generate_txt_content,
    format_srt_time,
)

_GR_TMP = get_gradio_tmp_dir()
os.environ.setdefault("GRADIO_TEMP_DIR", str(_GR_TMP))

# Light third-party imports
import numpy as np  # noqa: E402
import gradio as gr  # noqa: E402
import gradio.themes as gr_themes  # noqa: E402

# Optional Spaces decorator shim
try:
    import spaces  # type: ignore

    GPU_DECORATOR = spaces.GPU
except Exception:  # pragma: no cover

    def GPU_DECORATOR(fn):  # type: ignore
        return fn


# Engine singleton (no Gradio dependency inside)
_LOG = logging.getLogger("fast_stt.ui")
if not _LOG.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
# Reuse the same engine instance as the API to avoid double loads / CUDA contention
try:
    from .api import ENGINE as ENGINE  # type: ignore
except Exception:
    from .engine import FastSTTEngine  # local fallback (should not happen in serve.py)

    ENGINE = FastSTTEngine()


def _preload_status(_: float | None = None) -> str:
    if ENGINE.preload_done.is_set():
        return "✅ **Model status:** ready"
    if ENGINE.preload_started.is_set():
        return "⏳ **Model status:** preloading…"
    return "⌛ **Model status:** queued"


def start_session(request: gr.Request) -> str:
    session_dir = Path(tempfile.gettempdir()) / request.session_hash
    session_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info(f"Session with hash {request.session_hash} started at {session_dir}.")
    return session_dir.as_posix()


def end_session(request: gr.Request) -> None:
    session_dir = Path(tempfile.gettempdir()) / request.session_hash
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    _LOG.info(f"Session ended. Temporary directory {session_dir} removed.")


def get_audio_segment(
    audio_path: str, start_second: float, end_second: float
) -> Tuple[int, np.ndarray] | None:
    if not audio_path or not Path(audio_path).exists():
        return None
    try:
        start_ms = max(0, int(start_second * 1000))
        end_ms = max(start_ms + 100, int(end_second * 1000))
        audio = AudioSegment.from_file(audio_path)
        clipped_audio = audio[start_ms:end_ms]
        samples = np.array(clipped_audio.get_array_of_samples())
        if clipped_audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1).astype(samples.dtype)
        frame_rate = clipped_audio.frame_rate or audio.frame_rate
        if samples.size == 0:
            return None
        return (frame_rate, samples)
    except Exception:
        return None


@GPU_DECORATOR
def get_transcripts_and_raw_times(
    audio_path: str,
    session_dir: str,
    device_choice: str = "auto",
    vad_enabled: bool = False,
    chunking_enabled: bool = True,
    compile_enabled: bool = False,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not audio_path:
        gr.Error("No audio file path provided for transcription.", duration=None)
        return (
            [["N/A", "N/A", "Processing failed"]],
            [[0.0, 0.0]],
            None,
            gr.DownloadButton(label="Download Transcript (CSV)", visible=False),
            gr.DownloadButton(label="Download Transcript (SRT)", visible=False),
            gr.DownloadButton(label="Download Transcript (TXT)", visible=False),
            gr.DownloadButton(label="Download Transcript (MD)", visible=False),
        )

    vis_data = [["N/A", "N/A", "Processing failed"]]
    raw_times_data = [[0.0, 0.0]]
    processed_audio_path = None
    csv_button_update = gr.DownloadButton(
        label="Download Transcript (CSV)", visible=False
    )
    srt_button_update = gr.DownloadButton(
        label="Download Transcript (SRT)", visible=False
    )
    txt_button_update = gr.DownloadButton(
        label="Download Transcript (TXT)", visible=False
    )
    md_button_update = gr.DownloadButton(
        label="Download Transcript (MD)", visible=False
    )

    original_path_name = Path(audio_path).name
    audio_name = Path(audio_path).stem

    try:
        progress(0.0, desc="Loading audio")
        try:
            gr.Info(f"Loading audio: {original_path_name}", duration=2)
            audio = AudioSegment.from_file(audio_path)
            duration_sec = audio.duration_seconds
        except Exception as load_e:
            gr.Error(
                f"Failed to load audio file {original_path_name}: {load_e}",
                duration=None,
            )
            return (
                vis_data,
                raw_times_data,
                audio_path,
                csv_button_update,
                srt_button_update,
                txt_button_update,
                md_button_update,
            )

        progress(0.15, desc="Preprocessing (resample/mono/VAD)")
        resampled = False
        mono = False
        target_sr = 16000
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
            resampled = True
        if audio.channels == 2:
            audio = audio.set_channels(1)
            mono = True
        elif audio.channels > 2:
            gr.Error(
                f"Audio has {audio.channels} channels. Only mono (1) or stereo (2) supported.",
                duration=None,
            )
            return (
                vis_data,
                raw_times_data,
                audio_path,
                csv_button_update,
                srt_button_update,
                txt_button_update,
                md_button_update,
            )

        # Optional VAD pass (UI toggle)
        if vad_enabled:
            try:
                from .engine import maybe_apply_vad as _vad

                audio = _vad(audio)
            except Exception:
                pass

        if resampled or mono or vad_enabled:
            processed_audio_path = Path(session_dir, f"{audio_name}_resampled.wav")
            audio.export(processed_audio_path, format="wav")
            transcribe_path = processed_audio_path.as_posix()
            info_path_name = f"{original_path_name} (processed)"
        else:
            transcribe_path = audio_path
            info_path_name = original_path_name

        try:
            progress(0.35, desc="Loading model")
            # Ensure engine is loaded with current toggles
            ENGINE.load_model(
                device=device_choice, compile=compile_enabled, chunking=chunking_enabled
            )
            gr.Info(f"Transcribing {info_path_name} on {ENGINE.device}...", duration=2)

            progress(0.6, desc="Transcribing")
            result = ENGINE.transcribe_file(
                transcribe_path,
                device=device_choice,
                vad=False,  # already applied above
                chunking=chunking_enabled,
                compile=compile_enabled,
            )
            output_segments = result["segments"]
            if not output_segments:
                gr.Error(
                    "Transcription failed or produced unexpected output format.",
                    duration=None,
                )
                return (
                    vis_data,
                    raw_times_data,
                    audio_path,
                    csv_button_update,
                    srt_button_update,
                    txt_button_update,
                    md_button_update,
                )

            segment_timestamps = output_segments
            vis_data = [
                [f"{ts['start']:.2f}", f"{ts['end']:.2f}", ts["segment"]]
                for ts in segment_timestamps
            ]
            raw_times_data = [[ts["start"], ts["end"]] for ts in segment_timestamps]

            progress(0.85, desc="Generating exports")
            # CSV
            try:
                csv_file_path = Path(session_dir, f"transcription_{audio_name}.csv")
                with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Start (s)", "End (s)", "Segment"])
                    writer.writerows(vis_data)
                csv_button_update = gr.DownloadButton(
                    value=csv_file_path, visible=True, label="Download Transcript (CSV)"
                )
            except Exception as csv_e:
                gr.Error(
                    f"Failed to create transcript CSV file: {csv_e}", duration=None
                )

            # SRT
            if segment_timestamps:
                try:
                    srt_content = result["srt"]
                    srt_file_path = Path(session_dir, f"transcription_{audio_name}.srt")
                    srt_file_path.write_text(srt_content, encoding="utf-8")
                    srt_button_update = gr.DownloadButton(
                        value=srt_file_path,
                        visible=True,
                        label="Download Transcript (SRT)",
                    )
                except Exception as srt_e:
                    gr.Warning(
                        f"Failed to create transcript SRT file: {srt_e}", duration=5
                    )
                # TXT
                try:
                    txt_content = result["text"]
                    txt_file_path = Path(session_dir, f"transcription_{audio_name}.txt")
                    txt_file_path.write_text(txt_content, encoding="utf-8")
                    txt_button_update = gr.DownloadButton(
                        value=txt_file_path,
                        visible=True,
                        label="Download Transcript (TXT)",
                    )
                except Exception as txt_e:
                    gr.Warning(
                        f"Failed to create transcript TXT file: {txt_e}", duration=5
                    )
                # MD
                try:
                    md_content = result["md"]
                    md_file_path = Path(session_dir, f"transcription_{audio_name}.md")
                    md_file_path.write_text(md_content, encoding="utf-8")
                    md_button_update = gr.DownloadButton(
                        value=md_file_path,
                        visible=True,
                        label="Download Transcript (MD)",
                    )
                except Exception as md_e:
                    gr.Warning(
                        f"Failed to create transcript MD file: {md_e}", duration=5
                    )

            progress(1.0, desc="Done")
            gr.Info("Transcription complete.", duration=2)
            return (
                vis_data,
                raw_times_data,
                audio_path,
                csv_button_update,
                srt_button_update,
                txt_button_update,
                md_button_update,
            )

        except FileNotFoundError:
            gr.Error(
                f"Audio file for transcription not found: {Path(transcribe_path).name}.",
                duration=None,
            )
            return (
                vis_data,
                raw_times_data,
                audio_path,
                csv_button_update,
                srt_button_update,
                txt_button_update,
                md_button_update,
            )

        except Exception as e:
            try:
                import torch

                if isinstance(e, torch.cuda.OutOfMemoryError):
                    msg = "CUDA out of memory. Please try a shorter audio or reduce GPU load."
                    gr.Error(msg, duration=None)
                    return (
                        [["OOM", "OOM", msg]],
                        [[0.0, 0.0]],
                        audio_path,
                        csv_button_update,
                        srt_button_update,
                        txt_button_update,
                        md_button_update,
                    )
            except Exception:
                pass
            msg = f"Transcription failed: {e}"
            gr.Error(msg, duration=None)
            return (
                [["Error", "Error", msg]],
                [[0.0, 0.0]],
                audio_path,
                csv_button_update,
                srt_button_update,
                txt_button_update,
                md_button_update,
            )

        finally:
            try:
                if processed_audio_path and os.path.exists(processed_audio_path):
                    os.remove(processed_audio_path)
                gc.collect()
                ENGINE.maybe_empty_cuda_cache()
            except Exception as cleanup_e:
                gr.Warning(f"Issue during model cleanup: {cleanup_e}", duration=5)

    finally:
        pass


def play_segment(evt: gr.SelectData, raw_ts_list, current_audio_path):
    """Play the selected segment from the DataFrame."""
    if not isinstance(raw_ts_list, list) or not current_audio_path:
        return gr.Audio(value=None, label="Selected Segment")
    idx = (
        evt.index[0]
        if hasattr(evt, "index") and isinstance(evt.index, (list, tuple))
        else -1
    )
    if idx < 0 or idx >= len(raw_ts_list):
        return gr.Audio(value=None, label="Selected Segment")
    if not isinstance(raw_ts_list[idx], (list, tuple)) or len(raw_ts_list[idx]) != 2:
        return gr.Audio(value=None, label="Selected Segment")
    start_time_s, end_time_s = raw_ts_list[idx]
    seg = get_audio_segment(current_audio_path, float(start_time_s), float(end_time_s))
    if not seg:
        return gr.Audio(value=None, label="Selected Segment")
    return gr.Audio(
        value=seg,
        autoplay=True,
        label=f"Segment: {float(start_time_s):.2f}s - {float(end_time_s):.2f}s",
        interactive=False,
    )


def transcribe_many_files(
    audio_paths: list[str] | None,
    session_dir: str,
    device_choice: str = "auto",
    vad_enabled: bool = False,
    chunking_enabled: bool = True,
    compile_enabled: bool = False,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not audio_paths:
        return [], gr.DownloadButton(visible=False)
    rows: list[list[str | float]] = []
    zip_path = Path(session_dir) / "batch_transcripts.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, ap in enumerate(audio_paths, 1):
            progress(i / max(len(audio_paths), 1), desc=f"Transcribing {Path(ap).name}")
            try:
                res = ENGINE.transcribe_file(
                    ap,
                    device=device_choice,
                    vad=vad_enabled,
                    chunking=chunking_enabled,
                    compile=compile_enabled,
                )
                for s in res["segments"]:
                    rows.append(
                        [
                            Path(ap).name,
                            f"{s['start']:.2f}",
                            f"{s['end']:.2f}",
                            s["segment"],
                        ]
                    )
                stem = Path(ap).stem
                # write artifacts into zip
                zf.writestr(f"{stem}.txt", res["text"] or "")
                zf.writestr(f"{stem}.srt", res["srt"] or "")
                zf.writestr(f"{stem}.md", res["md"] or "")
            except Exception as e:
                rows.append([Path(ap).name, "0.00", "0.00", f"ERROR: {e}"])
    return rows, gr.DownloadButton(
        value=str(zip_path), label="Download All (ZIP)", visible=True
    )


# --- UI THEME / LAYOUT ---
article = (
    "<div style='background: linear-gradient(90deg, #76B900, #5A9200); color: white; "
    "padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;'>"
    "Fast, local speech-to-text powered by NVIDIA Parakeet."
    "</div>"
)
# Try packaged example asset; fall back to none
_examples: list[list[str]] = []
try:
    # packaged under fast_stt/data (optional)
    data_dir = resources.files("fast_stt").joinpath("data")
    ex = data_dir.joinpath("example-yt_saTD1u8PorI.mp3")
    if ex.is_file():
        _examples = [[str(ex)]]
except Exception:
    pass
examples = _examples

nvidia_theme = gr_themes.Default(
    primary_hue=gr_themes.Color(
        c50="#E6F1D9",
        c100="#CEE3B3",
        c200="#B5D58C",
        c300="#9CC766",
        c400="#84B940",
        c500="#76B900",
        c600="#68A600",
        c700="#5A9200",
        c800="#4C7E00",
        c900="#3E6A00",
        c950="#2F5600",
    ),
    neutral_hue="gray",
    font=[gr_themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set()

with gr.Blocks(theme=nvidia_theme, delete_cache=(900, 3600)) as demo:
    model_display_name = ENGINE.model_name_short
    gr.Markdown(
        f"<h1 style='text-align: center; margin: 0 auto;'>Speech Transcription with {model_display_name}</h1>"
    )
    gr.HTML(article)

    status_md = gr.Markdown("⌛ **Model status:** queued")
    t = gr.Timer(0.75)
    t.tick(_preload_status, None, status_md)

    current_audio_path_state = gr.State(None)
    raw_timestamps_list_state = gr.State([])

    session_dir_path = gr.State()
    demo.load(start_session, outputs=[session_dir_path])
    demo.load(lambda: ENGINE.start_preload(), outputs=None)

    # Device & options
    device_choice = gr.Dropdown(
        choices=["auto", "cpu", "cuda"], value="auto", label="Device"
    )
    with gr.Row():
        vad_checkbox = gr.Checkbox(
            label="Voice Activity Detection (skip silence)", value=False
        )
        chunking_checkbox = gr.Checkbox(label="Memory-friendly chunking", value=True)
        compile_checkbox = gr.Checkbox(
            label="Speed up with torch.compile (CUDA)", value=False
        )

    with gr.Tabs():
        with gr.TabItem("Audio File"):
            file_input = gr.Audio(
                sources=["upload"], type="filepath", label="Upload Audio File"
            )
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=[file_input],
                    label="Example Audio Files (Click to Load)",
                )
            file_transcribe_btn = gr.Button(
                "Transcribe Uploaded File", variant="primary"
            )

        with gr.TabItem("Microphone"):
            mic_input = gr.Audio(
                sources=["microphone"], type="filepath", label="Record Audio"
            )
            mic_transcribe_btn = gr.Button(
                "Transcribe Microphone Input", variant="primary"
            )

        with gr.TabItem("Batch Files"):
            files_input = gr.Files(label="Select multiple audio files")
            batch_btn = gr.Button("Transcribe Files (Batch)", variant="primary")
            batch_df = gr.DataFrame(
                headers=["File", "Start (s)", "End (s)", "Segment"],
                datatype=["str", "number", "number", "str"],
                wrap=True,
                label="Batch Segments",
            )
            batch_zip = gr.DownloadButton(label="Download All (ZIP)", visible=False)

    gr.Markdown("---")
    gr.Markdown(
        "<p><strong style='color: #FF0000; font-size: 1.2em;'>Transcription Results (Click row to play segment)</strong></p>"
    )

    with gr.Row():
        download_btn_csv = gr.DownloadButton(
            label="Download Transcript (CSV)", visible=False
        )
        download_btn_srt = gr.DownloadButton(
            label="Download Transcript (SRT)", visible=False
        )
        download_btn_txt = gr.DownloadButton(
            label="Download Transcript (TXT)", visible=False
        )
        download_btn_md = gr.DownloadButton(
            label="Download Transcript (MD)", visible=False
        )

    vis_timestamps_df = gr.DataFrame(
        headers=["Start (s)", "End (s)", "Segment"],
        datatype=["number", "number", "str"],
        wrap=True,
        label="Transcription Segments",
    )

    selected_segment_player = gr.Audio(label="Selected Segment", interactive=False)

    mic_transcribe_btn.click(
        fn=get_transcripts_and_raw_times,
        inputs=[
            mic_input,
            session_dir_path,
            device_choice,
            vad_checkbox,
            chunking_checkbox,
            compile_checkbox,
        ],
        outputs=[
            vis_timestamps_df,
            raw_timestamps_list_state,
            current_audio_path_state,
            download_btn_csv,
            download_btn_srt,
            download_btn_txt,
            download_btn_md,
        ],
        api_name="transcribe_mic",
        show_progress="full",
    )

    file_transcribe_btn.click(
        fn=get_transcripts_and_raw_times,
        inputs=[
            file_input,
            session_dir_path,
            device_choice,
            vad_checkbox,
            chunking_checkbox,
            compile_checkbox,
        ],
        outputs=[
            vis_timestamps_df,
            raw_timestamps_list_state,
            current_audio_path_state,
            download_btn_csv,
            download_btn_srt,
            download_btn_txt,
            download_btn_md,
        ],
        api_name="transcribe_file",
        show_progress="full",
    )

    batch_btn.click(
        fn=transcribe_many_files,
        inputs=[
            files_input,
            session_dir_path,
            device_choice,
            vad_checkbox,
            chunking_checkbox,
            compile_checkbox,
        ],
        outputs=[batch_df, batch_zip],
        api_name="transcribe_batch",
        show_progress="full",
    )

    vis_timestamps_df.select(
        fn=play_segment,
        inputs=[raw_timestamps_list_state, current_audio_path_state],
        outputs=[selected_segment_player],
    )

    demo.unload(end_session)

# NOTE: entry point & browser auto-open are in fast_stt.__main__.py


# Expose an internal helper to configure a Gradio queue safely when mounted
def _configure_queue(_demo) -> None:
    """
    Mount-safe queue setup. Older/newer Gradio versions have different
    Blocks.queue() signatures; we only pass universally-supported args.
    """
    try:
        _demo.queue(max_size=32)  # type: ignore[attr-defined]
    except TypeError:
        try:
            _demo.queue()  # type: ignore[attr-defined]
        except Exception:
            pass
