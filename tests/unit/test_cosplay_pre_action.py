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
    parsed = parse_cosplay_envelope(raw)
    assert parsed.display_text == "中文"
    assert parsed.tts_text == "English"
    assert parsed.action == "N03"
    assert parsed.assistant_turn_id == "turn-1"
    assert parsed.action_timing == "pre"
    assert parsed.action_error is None


def test_parse_cosplay_envelope_preserves_speech_when_post_action_is_rejected() -> None:
    raw = (
        '{"_cosplay_display_tts":true,"display_text":"中文","tts_text":"English",'
        '"action":"N13","action_timing":"post","assistant_turn_id":"turn-2"}'
    )
    parsed = parse_cosplay_envelope(raw)
    assert parsed.display_text == "中文"
    assert parsed.tts_text == "English"
    assert parsed.action is None
    assert parsed.action_timing is None
    assert parsed.assistant_turn_id == "turn-2"
    assert parsed.action_error == "post action must not be sent to OpenTalking"


def test_parse_cosplay_envelope_preserves_speech_when_action_timing_is_missing() -> None:
    raw = (
        '{"_cosplay_display_tts":true,"display_text":"中文","tts_text":"English",'
        '"action":"N05","assistant_turn_id":"turn-3"}'
    )
    parsed = parse_cosplay_envelope(raw)
    assert parsed.display_text == "中文"
    assert parsed.tts_text == "English"
    assert parsed.action is None
    assert parsed.action_error == "action_timing is required when action is present"


def test_parse_cosplay_envelope_preserves_speech_when_action_timing_is_invalid() -> None:
    raw = (
        '{"_cosplay_display_tts":true,"display_text":"中文","tts_text":"English",'
        '"action":"N05","action_timing":"during","assistant_turn_id":"turn-4"}'
    )
    parsed = parse_cosplay_envelope(raw)
    assert parsed.display_text == "中文"
    assert parsed.tts_text == "English"
    assert parsed.action is None
    assert parsed.action_error == "invalid action_timing: during"


def test_parse_non_envelope_falls_back() -> None:
    parsed = parse_cosplay_envelope("plain text")
    assert parsed.display_text == "plain text"
    assert parsed.tts_text == "plain text"
    assert parsed.action is None
    assert parsed.assistant_turn_id is None
    assert parsed.action_timing is None
    assert parsed.action_error is None
