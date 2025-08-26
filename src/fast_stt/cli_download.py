from __future__ import annotations
from pathlib import Path
import os
from .paths import get_model_file


def main() -> None:
    model_name = os.environ.get("PARAKEET_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
    target = Path(
        os.environ.get("PARAKEET_MODEL_PATH", str(get_model_file(model_name)))
    ).resolve()
    try:
        from nemo.collections.asr.models import ASRModel
    except Exception as e:
        print(f"[download] NeMo not installed? {e}")
        raise SystemExit(1)

    if target.exists():
        # Verify integrity; if corrupted, force re-download.
        try:
            _ = ASRModel.restore_from(str(target), map_location="cpu")
            print(f"[download] Model already present: {target}")
            return
        except Exception as e:
            print(
                f"[download] Existing model at {target} is invalid ({e}); re-downloading…"
            )
            try:
                target.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"[download] Fetching model from HF: {model_name}")
    model = ASRModel.from_pretrained(model_name=model_name, map_location="cpu")
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(target))
    print(f"[download] Saved to {target}")


if __name__ == "__main__":
    main()
