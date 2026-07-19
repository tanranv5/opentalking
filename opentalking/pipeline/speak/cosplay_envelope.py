"""Cosplay Brain 展示/TTS envelope 与前置动作切片工具。"""

from __future__ import annotations

from dataclasses import dataclass
import json


COSPLAY_ENVELOPE_MARKER = '{"_cosplay_display_tts"'


@dataclass(frozen=True)
class ParsedCosplayEnvelope:
    """Brain envelope 解析结果；动作协议错误与台词结果分离。"""

    display_text: str
    tts_text: str
    action: str | None = None
    assistant_turn_id: str | None = None
    action_timing: str | None = None
    action_error: str | None = None


def parse_cosplay_envelope(raw: str) -> ParsedCosplayEnvelope:
    """解析 Brain envelope；非 envelope 时 display/tts 均使用原文。

    动作属于可选媒体。action 元数据非法时保留 display/tts，拒绝动作并返回
    action_error，由调用方记录错误，不让动作协议错误覆盖合法台词。
    """
    text = (raw or "").strip()
    try:
        data = json.loads(text)
    except Exception:
        return ParsedCosplayEnvelope(text, text)
    if not isinstance(data, dict) or data.get("_cosplay_display_tts") is not True:
        return ParsedCosplayEnvelope(text, text)
    display = str(data.get("display_text") or "").strip()
    tts = str(data.get("tts_text") or "").strip()
    if not tts:
        tts = display
    if not display:
        display = tts
    action = str(data.get("action") or "").strip() or None
    assistant_turn_id = str(data.get("assistant_turn_id") or "").strip() or None
    timing_raw = str(data.get("action_timing") or "").strip().lower()
    if not action:
        return ParsedCosplayEnvelope(display, tts, assistant_turn_id=assistant_turn_id)
    if not timing_raw:
        return ParsedCosplayEnvelope(
            display,
            tts,
            assistant_turn_id=assistant_turn_id,
            action_error="action_timing is required when action is present",
        )
    if timing_raw in {"post", "after"}:
        return ParsedCosplayEnvelope(
            display,
            tts,
            assistant_turn_id=assistant_turn_id,
            action_error="post action must not be sent to OpenTalking",
        )
    if timing_raw not in {"pre", "before"}:
        return ParsedCosplayEnvelope(
            display,
            tts,
            assistant_turn_id=assistant_turn_id,
            action_error=f"invalid action_timing: {timing_raw}",
        )
    return ParsedCosplayEnvelope(display, tts, action, assistant_turn_id, "pre")


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
