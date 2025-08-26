from __future__ import annotations
import os
import uvicorn
from .serve import make_app


def main() -> None:
    host = os.environ.get("FAST_STT_HOST", "127.0.0.1")
    port = int(os.environ.get("FAST_STT_PORT", "7860"))
    app = make_app(open_url=None, include_ui=False)
    print("Launching fast-stt API only (endpoints at /api)...")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
