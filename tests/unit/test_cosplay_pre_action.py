from __future__ import annotations

from opentalking.pipeline.speak.cosplay_envelope import pre_action_slice_lengths


def test_pre_action_slice_lengths_caps_video_and_aligns_audio() -> None:
    assert pre_action_slice_lengths(125, 25.0, 16000, 1800) == (45, 28800)
    assert pre_action_slice_lengths(20, 25.0, 16000, 1800) == (20, 12800)
    assert pre_action_slice_lengths(125, 25.0, 16000, 0) == (0, 0)
