from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import opentalking.pipeline.session.runner as session_runner_module
import opentalking.pipeline.speak.clip_tasks as clip_tasks_module
import opentalking.pipeline.speak.synthesis_runner as synthesis_runner_module
from opentalking.pipeline.session.runner import SessionRunner
from opentalking.pipeline.speak.clip_tasks import ExternalClipTaskMixin
from opentalking.pipeline.speak.synthesis_runner import FlashTalkRunner


class _DummyClipRunner(ExternalClipTaskMixin):
    def __init__(self) -> None:
        self.session_id = "clip-session"
        self.redis = object()
        self._closed = False
        self._speaking = False
        self.speech_tasks: set[asyncio.Task[None]] = set()
        self._active_clip_task: asyncio.Task[None] | None = None
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release: dict[str, asyncio.Event] = {}

    async def play_clip(self, clip_id: str) -> bool:
        self.release[clip_id] = asyncio.Event()
        await self.started.put(clip_id)
        await self.release[clip_id].wait()
        return True


class _PendingTask:
    def done(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_external_clip_task_is_cancelled_by_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []

    async def capture_event(_redis: object, _sid: str, name: str, data: dict[str, object]) -> None:
        events.append({"name": name, **data})

    monkeypatch.setattr(clip_tasks_module, "publish_event", capture_event)
    runner = _DummyClipRunner()

    assert runner.start_clip("N01") is True
    assert await asyncio.wait_for(runner.started.get(), timeout=0.2) == "N01"

    await runner.cancel_active_clip("speech_started")

    assert runner._active_clip_task is None
    assert events == [{
        "name": "clip.ended",
        "session_id": "clip-session",
        "clip_id": "N01",
        "played": False,
        "reason": "speech_started",
    }]


@pytest.mark.asyncio
async def test_new_external_clip_supersedes_previous_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []

    async def capture_event(_redis: object, _sid: str, name: str, data: dict[str, object]) -> None:
        events.append({"name": name, **data})

    monkeypatch.setattr(clip_tasks_module, "publish_event", capture_event)
    runner = _DummyClipRunner()

    assert runner.start_clip("N01") is True
    assert await asyncio.wait_for(runner.started.get(), timeout=0.2) == "N01"
    assert runner.start_clip("N02") is True
    assert await asyncio.wait_for(runner.started.get(), timeout=0.2) == "N02"

    runner.release["N02"].set()
    active = runner._active_clip_task
    assert active is not None
    await active

    assert events == [
        {
            "name": "clip.ended",
            "session_id": "clip-session",
            "clip_id": "N01",
            "played": False,
            "reason": "superseded",
        },
        {
            "name": "clip.ended",
            "session_id": "clip-session",
            "clip_id": "N02",
            "played": True,
        },
    ]


def test_external_clip_is_rejected_when_speech_task_is_pending() -> None:
    runner = _DummyClipRunner()
    pending = _PendingTask()
    runner.speech_tasks = {pending}  # type: ignore[assignment]

    assert runner.start_clip("N03") is False
    assert runner._active_clip_task is None


def test_both_realtime_runners_use_external_clip_task_mixin() -> None:
    assert issubclass(FlashTalkRunner, ExternalClipTaskMixin)
    assert issubclass(SessionRunner, ExternalClipTaskMixin)


@pytest.mark.asyncio
async def test_flashtalk_external_clip_does_not_publish_speech_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(FlashTalkRunner)
    runner.session_id = "flash-clip"
    runner.avatar_id = "avatar"
    runner.redis = object()
    runner.flashtalk = SimpleNamespace(fps=25, sample_rate=16000)
    runner._speak_lock = asyncio.Lock()
    runner._interrupt = asyncio.Event()
    runner._closed = False
    runner._speaking = False
    runner._speech_media_active = False
    runner._resolve_clip_path = lambda _clip_id: "/tmp/N01.mp4"  # type: ignore[method-assign]
    runner._load_clip_video = lambda _path: [np.zeros((2, 2, 3), dtype=np.uint8)]  # type: ignore[method-assign]
    runner._load_clip_audio = lambda _path, samples, _rate: np.zeros(samples, dtype=np.int16)  # type: ignore[method-assign]
    runner._ensure_media_clock_started = lambda: None  # type: ignore[method-assign]
    captured: dict[str, object] = {}

    async def queue_chunk(_pcm: object, _frames: object, *, speech_media: bool) -> None:
        captured["speech_media"] = speech_media

    async def set_state(_redis: object, _sid: str, _state: str) -> None:
        return None

    async def no_wait(_seconds: float) -> None:
        return None

    runner._queue_av_chunk = queue_chunk  # type: ignore[method-assign]
    monkeypatch.setattr(synthesis_runner_module, "set_session_state", set_state)
    monkeypatch.setattr(synthesis_runner_module.asyncio, "sleep", no_wait)

    assert await runner.play_clip("N01") is True
    assert captured == {"speech_media": False}


@pytest.mark.asyncio
async def test_session_runner_pre_action_waits_for_clip_duration_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(SessionRunner)
    runner.session_id = "session-pre-action"
    runner.avatar_id = "avatar"
    runner.webrtc = SimpleNamespace(
        draining=False,
        clear_media_queues=lambda: None,
        reset_clocks=lambda: None,
    )
    runner.avatar_state = SimpleNamespace(manifest=SimpleNamespace(fps=25, sample_rate=16000))
    runner._interrupt = asyncio.Event()
    # full clip (max_ms=0), wait on
    runner._read_int_env = lambda name, default: {  # type: ignore[method-assign]
        "OPENTALKING_PRE_ACTION_MAX_MS": 0,
        "OPENTALKING_PRE_ACTION_WAIT": 1,
    }.get(name, default)
    runner._resolve_clip_path = lambda _clip_id: SimpleNamespace(__str__=lambda self: "/tmp/N01.mp4")  # type: ignore[method-assign]
    runner._load_clip_video = lambda _path: [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]  # type: ignore[method-assign]
    runner._load_clip_audio = lambda _path, samples, _rate: np.zeros(samples, dtype=np.int16)  # type: ignore[method-assign]

    async def video_sink(_frame: object, *, speech_media: bool = True) -> None:
        assert speech_media is False

    async def audio_sink(_pcm: object, _rate: int, *, speech_media: bool = True) -> None:
        assert speech_media is False

    waits: list[float] = []

    async def track_wait_for(awaitable, timeout=None):  # type: ignore[no-untyped-def]
        waits.append(float(timeout or 0.0))
        # drain the wait() coroutine if any
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError

    runner._video_sink = video_sink  # type: ignore[method-assign]
    runner._audio_sink = audio_sink  # type: ignore[method-assign]
    monkeypatch.setattr(session_runner_module.asyncio, "wait_for", track_wait_for)

    assert await runner._play_pre_action("N01", "turn-1") is True
    # 2 frames @ 25fps => 0.08s wait
    assert waits, "pre-action should wait for playback duration when WAIT=1"
    assert waits[0] == pytest.approx(2 / 25.0)


@pytest.mark.asyncio
async def test_session_runner_pre_action_can_skip_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(SessionRunner)
    runner.session_id = "session-pre-action-nowait"
    runner.avatar_id = "avatar"
    runner.webrtc = SimpleNamespace(
        draining=False,
        clear_media_queues=lambda: None,
        reset_clocks=lambda: None,
    )
    runner.avatar_state = SimpleNamespace(manifest=SimpleNamespace(fps=25, sample_rate=16000))
    runner._interrupt = asyncio.Event()
    runner._read_int_env = lambda name, default: {  # type: ignore[method-assign]
        "OPENTALKING_PRE_ACTION_MAX_MS": 1800,
        "OPENTALKING_PRE_ACTION_WAIT": 0,
    }.get(name, default)
    runner._resolve_clip_path = lambda _clip_id: SimpleNamespace(__str__=lambda self: "/tmp/N01.mp4")  # type: ignore[method-assign]
    runner._load_clip_video = lambda _path: [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]  # type: ignore[method-assign]
    runner._load_clip_audio = lambda _path, samples, _rate: np.zeros(samples, dtype=np.int16)  # type: ignore[method-assign]

    async def video_sink(_frame: object, *, speech_media: bool = True) -> None:
        assert speech_media is False

    async def audio_sink(_pcm: object, _rate: int, *, speech_media: bool = True) -> None:
        assert speech_media is False

    async def fail_wait_for(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError("pre-action must not wait when WAIT=0")

    runner._video_sink = video_sink  # type: ignore[method-assign]
    runner._audio_sink = audio_sink  # type: ignore[method-assign]
    monkeypatch.setattr(session_runner_module.asyncio, "wait_for", fail_wait_for)

    assert await runner._play_pre_action("N01", "turn-1") is True
