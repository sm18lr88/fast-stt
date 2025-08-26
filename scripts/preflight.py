from __future__ import annotations
import shutil, subprocess, sys, json
from pathlib import Path


def cmd_ok(cmd: list[str]) -> bool:
    try:
        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return True
    except Exception:
        return False


def main():
    report: dict[str, object] = {}

    # Python / uv
    report["python"] = sys.version
    report["uv"] = shutil.which("uv") or "missing"

    # ffmpeg
    ff = shutil.which("ffmpeg")
    if ff:
        try:
            out = subprocess.check_output([ff, "-version"]).decode().splitlines()[0]
        except Exception:
            out = ff
        report["ffmpeg"] = out
    else:
        report["ffmpeg"] = "missing"

    # NVIDIA
    if cmd_ok(["nvidia-smi", "--help"]):
        try:
            out = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version",
                        "--format=csv,noheader",
                    ]
                )
                .decode()
                .strip()
            )
            report["nvidia"] = out.splitlines()
        except Exception as e:
            report["nvidia"] = f"detected but query failed: {e}"
    else:
        report["nvidia"] = "not detected"

    # basic disk space check (>=5GB free)
    try:
        import shutil as _sh

        root = Path.cwd()
        # Windows: prefer current drive root (e.g. "C:\\"); POSIX: "/"
        root_str = (
            str(getattr(root, "drive", "") + "\\")
            if getattr(root, "drive", "")
            else "/"
        )
        total, used, free = _sh.disk_usage(root_str)
        free_gb = free / (1024**3)
        report["disk_free_gb"] = round(free_gb, 2)
        if free_gb < 5:
            print(
                json.dumps(
                    {"disk_free_gb": report["disk_free_gb"], "error": "low disk"},
                    indent=2,
                )
            )
            sys.exit(3)
    except Exception:
        pass
    print(json.dumps(report, indent=2))
    # exit with non-zero if a hard requirement is missing
    if report["ffmpeg"] == "missing":
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
