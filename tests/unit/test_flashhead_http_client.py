from __future__ import annotations

import json
from pathlib import Path

from opentalking.providers.synthesis.flashhead.http_client import FlashHeadHTTPClient
from opentalking.providers.synthesis.flashhead.ws_client import FlashHeadWSClient


class FakeFlashHeadWebSocket:
    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    async def send(self, message: str) -> None:
        self.sent_messages.append(message)

    async def recv(self) -> str:
        return json.dumps({
            "type": "session.created",
            "session_id": "sess_ws",
            "audio": {"sample_rate": 16000, "chunk_samples": 16000},
            "video": {
                "fps": 25,
                "width": 416,
                "height": 704,
                "frame_count": 25,
                "motion_frames_num": 0,
            },
        })


def test_flashhead_payload_matches_omnirt_schema(tmp_path: Path) -> None:
    client = FlashHeadHTTPClient(
        base_url="http://example.test",
        model="soulx-flashhead-1.3b",
        shared_local_dir=str(tmp_path / "local"),
        shared_remote_dir="/mnt/flashhead",
        fps=25,
        sample_rate=16000,
        chunk_samples=16000,
    )
    client._ref_image_remote_path = "/mnt/flashhead/reference.png"

    payload = client._build_generate_payload("/mnt/flashhead/chunk.wav")

    assert payload["task"] == "audio2video"
    assert payload["model"] == "soulx-flashhead-1.3b"
    assert payload["inputs"] == {
        "image": "/mnt/flashhead/reference.png",
        "audio": "/mnt/flashhead/chunk.wav",
    }
    assert payload["config"]["fps"] == 25


def test_flashhead_maps_shared_paths(tmp_path: Path) -> None:
    local = tmp_path / "shared"
    client = FlashHeadHTTPClient(
        base_url="http://example.test",
        shared_local_dir=str(local),
        shared_remote_dir="/mnt/shared",
    )

    mapped = client._map_remote_path_to_local("/mnt/shared/out/result.mp4")

    assert mapped == (local / "out" / "result.mp4").resolve()


async def test_flashhead_init_session_ignores_runner_metadata(tmp_path: Path) -> None:
    client = FlashHeadHTTPClient(
        base_url="http://example.test",
        shared_local_dir=str(tmp_path / "shared"),
        shared_remote_dir="/mnt/shared",
    )

    init = await client.init_session(
        b"fake-image",
        wav2lip_postprocess_mode="basic",
        mouth_metadata={"version": 1},
        video_config={"fps": 25},
    )

    assert init["model"] == "soulx-flashhead-1.3b"
    assert client._ref_image_remote_path.endswith("/reference.png")
    await client.close()


async def test_flashhead_ws_init_session_ignores_runner_metadata() -> None:
    client = FlashHeadWSClient(ws_url="ws://example.test/avatar")
    fake_ws = FakeFlashHeadWebSocket()
    client._ws = fake_ws

    init = await client.init_session(
        b"fake-image",
        wav2lip_postprocess_mode="basic",
        mouth_metadata={"version": 1},
        video_config={"fps": 25},
    )

    sent = json.loads(fake_ws.sent_messages[0])
    assert init["type"] == "session.created"
    assert sent["type"] == "session.create"
    assert sent["inputs"]["image_b64"]
    assert "wav2lip_postprocess_mode" not in sent
    assert "mouth_metadata" not in sent
    assert "video_config" not in sent
