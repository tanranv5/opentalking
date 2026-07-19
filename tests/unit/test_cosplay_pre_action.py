from __future__ import annotations

import pytest

from opentalking.pipeline.speak.cosplay_envelope import (
    parse_cosplay_envelope,
    pre_action_slice_lengths,
)


def test_pre_action_slice_lengths_caps_video_and_aligns_audio() -> None:
    assert pre_action_slice_lengths(125, 25.0, 16000, 1800) == (45, 28800)
    assert pre_action_slice_lengths(20, 25.0, 16000, 1800) == (20, 12800)
    assert pre_action_slice_lengths(125, 25.0, 16000, 0) == (0, 0)


def test_parse_cosplay_envelope_pre_action() -> None:
    raw = (
        '{"_cosplay_display_tts":true,"display_text":"中文","tts_text":"English",'
        '"action":"N03","action_timing":"pre","assistant_turn_id":"turn-1"}'
    )
    display, tts, action, turn_id, timing = parse_cosplay_envelope(raw)
    assert display == "中文"
    assert tts == "English"
    assert action == "N03"
    assert turn_id == "turn-1"
    assert timing == "pre"


def test_parse_cosplay_envelope_rejects_post_action() -> None:
    raw = (
        '{"_cosplay_display_tts":true,"display_text":"中文","tts_text":"English",'
        '"action":"N13","action_timing":"post","assistant_turn_id":"turn-2"}'
    )
    with pytest.raises(ValueError, match="post action must not be sent to OpenTalking"):
        parse_cosplay_envelope(raw)


def test_parse_cosplay_envelope_rejects_action_without_timing() -> None:
    raw = (
        '{"_cosplay_display_tts":true,"display_text":"中文","tts_text":"English",'
        '"action":"N05","assistant_turn_id":"turn-3"}'
    )
    with pytest.raises(ValueError, match="action_timing is required"):
        parse_cosplay_envelope(raw)


def test_parse_cosplay_envelope_rejects_invalid_action_timing() -> None:
    raw = (
        '{"_cosplay_display_tts":true,"display_text":"中文","tts_text":"English",'
        '"action":"N05","action_timing":"during","assistant_turn_id":"turn-4"}'
    )
    with pytest.raises(ValueError, match="invalid action_timing"):
        parse_cosplay_envelope(raw)


def test_parse_non_envelope_falls_back() -> None:
    display, tts, action, turn_id, timing = parse_cosplay_envelope("plain text")
    assert display == "plain text"
    assert tts == "plain text"
    assert action is None
    assert turn_id is None
    assert timing is None
