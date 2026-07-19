from __future__ import annotations

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


def test_parse_cosplay_envelope_post_action_not_for_ot() -> None:
    raw = (
        '{"_cosplay_display_tts":true,"display_text":"中文","tts_text":"English",'
        '"action":"N13","action_timing":"post","assistant_turn_id":"turn-2"}'
    )
    display, tts, action, turn_id, timing = parse_cosplay_envelope(raw)
    assert display == "中文"
    assert tts == "English"
    assert action is None
    assert turn_id == "turn-2"
    assert timing == "post"


def test_parse_cosplay_envelope_legacy_action_defaults_to_pre() -> None:
    raw = (
        '{"_cosplay_display_tts":true,"display_text":"中文","tts_text":"English",'
        '"action":"N05","assistant_turn_id":"turn-3"}'
    )
    _, _, action, turn_id, timing = parse_cosplay_envelope(raw)
    assert action == "N05"
    assert turn_id == "turn-3"
    assert timing == "pre"


def test_parse_non_envelope_falls_back() -> None:
    display, tts, action, turn_id, timing = parse_cosplay_envelope("plain text")
    assert display == "plain text"
    assert tts == "plain text"
    assert action is None
    assert turn_id is None
    assert timing is None
