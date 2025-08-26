from __future__ import annotations
import datetime
from typing import Any, List


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    delta = datetime.timedelta(seconds=seconds)
    total = int(delta.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    ms = delta.microseconds // 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt_content(segment_timestamps: List[dict[str, Any]]) -> str:
    out: List[str] = []
    for i, ts in enumerate(segment_timestamps, start=1):
        out.append(str(i))
        out.append(f"{format_srt_time(ts['start'])} --> {format_srt_time(ts['end'])}")
        out.append(ts["segment"])
        out.append("")
    return "\n".join(out)


def generate_txt_content(segment_timestamps: List[dict[str, Any]]) -> str:
    return "\n".join([ts["segment"] for ts in segment_timestamps])


def generate_md_content(segment_timestamps: List[dict[str, Any]]) -> str:
    return "\n".join([f"- {ts['segment']}" for ts in segment_timestamps])
