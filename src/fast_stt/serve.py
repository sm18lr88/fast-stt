from __future__ import annotations
import os, threading, time, webbrowser
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router as api_router, ENGINE as API_ENGINE


def make_app(open_url: str | None = None, include_ui: bool = True) -> FastAPI:
    app = FastAPI(title="fast-stt", version="0.1.0")

    # Optional CORS
    if os.environ.get("FAST_STT_CORS", "").lower() in ("1", "true", "yes"):
        origins = [
            o.strip() for o in os.environ.get("FAST_STT_CORS_ORIGINS", "*").split(",")
        ]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(api_router, prefix="/api", tags=["fast-stt"])

    # Mount Gradio UI at root (optional; avoids importing gradio for API-only mode)
    if include_ui:
        from .app import demo, _configure_queue  # lazy import to keep API lean
        import gradio as gr  # local import

        _configure_queue(demo)
        # Use empty path to avoid double-slash redirects like "//assets/..."
        gr.mount_gradio_app(app, demo, path="")

    @app.on_event("startup")
    async def _startup():
        # preload core model (no Gradio required)
        API_ENGINE.start_preload()
        # Open browser once server is starting (optional)
        if (
            include_ui
            and open_url
            and os.environ.get("FAST_STT_OPEN_BROWSER", "1")
            not in (
                "0",
                "false",
                "no",
            )
        ):

            def _open():
                time.sleep(0.8)
                try:
                    webbrowser.open(open_url)
                except Exception:
                    pass

            threading.Thread(target=_open, daemon=True).start()

    return app
