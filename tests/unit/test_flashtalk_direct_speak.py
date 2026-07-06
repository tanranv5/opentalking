from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import opentalking.pipeline.speak.synthesis_runner as synthesis_runner
from opentalking.core.types.frames import AudioChunk
from opentalking.pipeline.speak.synthesis_runner import FlashTalkRunner


class FakeDirectTTS:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    async def synthesize_stream(self, text: str, voice: str | None = None):
        self.captured["tts_text"] = text
        self.captured["tts_voice_arg"] = voice
        yield AudioChunk(
            data=np.array([1, -2, 3], dtype=np.int16),
            sample_rate=16000,
            duration_ms=0.2,
        )

    async def aclose(self) -> None:
        self.captured["closed"] = True


@pytest.mark.asyncio
async def test_flashtalk_runner_direct_speak_synthesizes_text_to_uploaded_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    runner = FlashTalkRunner.__new__(FlashTalkRunner)
    runner.session_id = "sess_direct"
    runner.model_type = "quicktalk"
    runner.speech_tasks = set()
    runner._tts_settings = SimpleNamespace()

    monkeypatch.delenv("OPENTALKING_QUICKTALK_RENDER_CHUNK_MS", raising=False)

    def fake_build_tts_adapter(**kwargs: object) -> FakeDirectTTS:
        captured["tts_kwargs"] = kwargs
        return FakeDirectTTS(captured)

    async def fake_speak_uploaded_pcm(
        pcm: np.ndarray,
        *,
        enqueue_unix: float | None = None,
        speech_text: str | None = None,
    ) -> None:
        captured["pcm"] = np.asarray(pcm, dtype=np.int16).copy()
        captured["enqueue_unix"] = enqueue_unix
        captured["speech_text"] = speech_text

    monkeypatch.setattr(synthesis_runner, "build_tts_adapter", fake_build_tts_adapter, raising=False)
    runner.speak_uploaded_pcm = fake_speak_uploaded_pcm  # type: ignore[method-assign]

    task = runner.create_direct_speak_task(
        "  开场白  ",
        tts_voice="voice-a",
        tts_provider="local_cosyvoice",
        tts_model="cosyvoice3",
        enqueue_unix=123.0,
    )
    await task

    assert captured["tts_kwargs"] == {
        "sample_rate": 16000,
        "chunk_ms": 500.0,
        "settings": runner._tts_settings,
        "default_voice": "voice-a",
        "tts_provider": "local_cosyvoice",
        "tts_model": "cosyvoice3",
    }
    assert captured["tts_text"] == "开场白"
    assert np.array_equal(captured["pcm"], np.array([1, -2, 3], dtype=np.int16))
    assert captured["speech_text"] == "开场白"
    assert captured["enqueue_unix"] == 123.0
    assert captured["closed"] is True
