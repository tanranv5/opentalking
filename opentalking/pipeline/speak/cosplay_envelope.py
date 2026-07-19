"""Cosplay Brain 展示/TTS envelope 与前置动作切片工具。"""

from __future__ import annotations

import json


COSPLAY_ENVELOPE_MARKER = '{"_cosplay_display_tts"'


def parse_cosplay_envelope(raw: str) -> tuple[str, str, str | None, str | None, str | None]:
    """解析 Brain envelope；非 envelope 时 display/tts 均回退原文。

    Returns:
        (display_text, tts_text, action, assistant_turn_id, action_timing)
        action_timing 为 pre/post；仅 pre 应由 OT 前置播放。
    """
    text = (raw or "").strip()
    try:
        data = json.loads(text)
    except Exception:
        return text, text, None, None, None
    if not isinstance(data, dict) or data.get("_cosplay_display_tts") is not True:
        return text, text, None, None, None
    display = str(data.get("display_text") or "").strip()
    tts = str(data.get("tts_text") or "").strip()
    if not tts:
        tts = display
    if not display:
        display = tts
    action = str(data.get("action") or "").strip() or None
    assistant_turn_id = str(data.get("assistant_turn_id") or "").strip() or None
    timing_raw = str(data.get("action_timing") or "").strip().lower()
    if timing_raw in {"pre", "before"}:
        action_timing = "pre"
    elif timing_raw in {"post", "after"}:
        action_timing = "post"
    else:
        # 缺字段时：有 action 视为 pre（兼容旧 Brain）；无 action 为 None
        action_timing = "pre" if action else None
    # post 动作不应由 OT 前置；双保险（Brain 本应不下发 post action）
    if action_timing != "pre":
        action = None
    return display, tts, action, assistant_turn_id, action_timing


def pre_action_slice_lengths(
    total_frames: int,
    fps: float,
    sample_rate: int,
    max_ms: int,
) -> tuple[int, int]:
    """计算不超过 max_ms 的视频帧数与严格对齐的音频采样数。"""
    if total_frames <= 0 or fps <= 0 or sample_rate <= 0 or max_ms <= 0:
        return 0, 0
    frame_count = min(total_frames, max(0, int(fps * max_ms / 1000.0)))
    sample_count = int(round(frame_count * sample_rate / fps))
    return frame_count, max(0, sample_count)
