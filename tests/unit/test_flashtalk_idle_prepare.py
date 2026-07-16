from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

import opentalking.pipeline.speak.synthesis_runner as synthesis_runner
from opentalking.pipeline.speak.synthesis_runner import FlashTalkRunner


class _FakePeerConnection:
    connectionState = "new"

    def on(self, _event: str):
        def _register(callback):
            return callback

        return _register


class _FakeWebRTC:
    def __init__(self, *, fps: float, sample_rate: int) -> None:
        self.pc = _FakePeerConnection()


class _FakeAudio2VideoClient:
    fps = 25
    sample_rate = 16000
    width = 4
    height = 4
    frame_num = 1
    motion_frames_num = 1
    slice_len = 1
    audio_chunk_samples = 640

    async def connect(self) -> None:
        return None


class _FakeVideoCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


def _build_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_type: str,
    allow_background_idle_cache: bool = False,
) -> tuple[FlashTalkRunner, Path, Path]:
    avatars_root = tmp_path / "avatars"
    avatar_dir = avatars_root / "avatar"
    avatar_dir.mkdir(parents=True)
    reference_path = avatar_dir / "reference.png"
    Image.new("RGB", (4, 4), (32, 64, 96)).save(reference_path)

    runner = FlashTalkRunner.__new__(FlashTalkRunner)
    runner.session_id = f"sess-{model_type}"
    runner.avatar_id = "avatar"
    runner.model_type = model_type
    runner.avatars_root = avatars_root
    runner._custom_ref_image_path = ""
    runner.flashtalk = _FakeAudio2VideoClient()
    runner.audio2video = runner.flashtalk
    runner._closed = False
    runner._idle_task = object()
    runner.ready_event = asyncio.Event()
    runner._reference_frame = None
    runner._last_frame = None
    runner._idle_frames = []
    runner._idle_playback_indices = []
    runner._idle_frame_idx = 0
    runner._allow_background_idle_cache = allow_background_idle_cache
    runner._dynamic_idle_prepare_task = None

    async def _init_flashtalk_session(_ref_image_path: Path) -> None:
        return None

    runner._init_flashtalk_session = _init_flashtalk_session
    runner._tts_opener_enabled_for_model = lambda: False
    monkeypatch.setattr(synthesis_runner, "WebRTCSession", _FakeWebRTC)
    return runner, avatar_dir, reference_path


@pytest.mark.parametrize("model_type", ["quicktalk", "flashhead"])
@pytest.mark.asyncio
async def test_prepare_loads_prerecorded_idle_frames_in_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> None:
    runner, avatar_dir, _ = _build_runner(tmp_path, monkeypatch, model_type=model_type)
    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]
    loaded_paths: list[Path] = []

    def _load_idle_video(path: Path) -> list[np.ndarray]:
        loaded_paths.append(path)
        return frames

    monkeypatch.setattr(runner, "_load_idle_video", _load_idle_video)

    await runner.prepare()

    assert runner.ready_event.is_set()
    assert loaded_paths == []
    await asyncio.sleep(0)
    assert loaded_paths == [avatar_dir]
    assert runner._idle_frames is frames


@pytest.mark.parametrize("model_type", ["quicktalk", "flashhead"])
@pytest.mark.asyncio
async def test_prepare_keeps_idle_frames_empty_when_idle_video_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> None:
    runner, _, _ = _build_runner(tmp_path, monkeypatch, model_type=model_type)
    monkeypatch.setattr(runner, "_load_idle_video", lambda _avatar_dir: None)

    await runner.prepare()
    await asyncio.sleep(0)

    assert runner._idle_frames == []


@pytest.mark.asyncio
async def test_prerecorded_idle_loader_failure_is_non_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner, avatar_dir, _ = _build_runner(tmp_path, monkeypatch, model_type="quicktalk")
    caplog.set_level("INFO", logger=synthesis_runner.__name__)

    def _raise_decode_error(_avatar_dir: Path) -> None:
        raise RuntimeError("decode failed")

    monkeypatch.setattr(runner, "_load_idle_video", _raise_decode_error)

    await runner._prepare_prerecorded_idle_video(avatar_dir)

    assert runner._idle_frames == []
    assert "load failed (non-fatal)" in caplog.text


@pytest.mark.parametrize(
    ("idle_enabled", "expected_call"),
    [(True, "dynamic"), (False, "background")],
)
@pytest.mark.asyncio
async def test_flashtalk_idle_cache_gates_are_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_enabled: bool,
    expected_call: str,
) -> None:
    runner, _, reference_path = _build_runner(
        tmp_path,
        monkeypatch,
        model_type="flashtalk",
        allow_background_idle_cache=True,
    )
    calls: list[tuple[str, Path]] = []

    async def _prepare_dynamic_idle_cache(path: Path) -> None:
        calls.append(("dynamic", path))

    async def _prepare_idle_cache_background(path: Path) -> None:
        calls.append(("background", path))

    async def _prepare_prerecorded_idle_video(path: Path) -> None:
        calls.append(("prerecorded", path))

    monkeypatch.setattr(runner, "_prepare_dynamic_idle_cache", _prepare_dynamic_idle_cache)
    monkeypatch.setattr(runner, "_prepare_idle_cache_background", _prepare_idle_cache_background)
    monkeypatch.setattr(runner, "_prepare_prerecorded_idle_video", _prepare_prerecorded_idle_video)
    monkeypatch.setattr(
        synthesis_runner,
        "get_settings",
        lambda: SimpleNamespace(flashtalk_idle_enable=idle_enabled),
    )

    await runner.prepare()
    await asyncio.sleep(0)

    assert calls == [(expected_call, reference_path)]
    assert (runner._dynamic_idle_prepare_task is not None) is idle_enabled


def test_idle_video_loader_truncates_frames_at_memory_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    avatar_dir = tmp_path / "avatar"
    avatar_dir.mkdir()
    (avatar_dir / "idle.mp4").touch()
    decoded_frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(4)]
    capture = _FakeVideoCapture(decoded_frames)
    runner = FlashTalkRunner.__new__(FlashTalkRunner)
    runner.flashtalk = _FakeAudio2VideoClient()

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(synthesis_runner, "_IDLE_VIDEO_MAX_SECONDS", 0.08)

    frames = runner._load_idle_video(avatar_dir)

    assert frames is not None
    assert len(frames) == 2
    assert capture.read_count == 3
    assert capture.released
    assert "Idle video truncated at memory limit" in caplog.text
