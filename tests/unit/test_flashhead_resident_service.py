from __future__ import annotations

import json
import struct

import numpy as np

from scripts import flashhead_http_service_resident as service


def test_pack_video_payload_uses_flashtalk_binary_protocol() -> None:
    payload = service._pack_video_payload([b"jpeg-a", b"jpeg-b"])

    assert payload[:4] == service.MAGIC_VIDEO
    assert struct.unpack("<I", payload[4:8])[0] == 2
    first_len = struct.unpack("<I", payload[8:12])[0]
    assert first_len == len(b"jpeg-a")
    assert payload[12:12 + first_len] == b"jpeg-a"


def test_session_created_payload_reports_audio_and_video_contract() -> None:
    payload = service._session_created_payload(
        session_id="sess_ws",
        sample_rate=16000,
        chunk_samples=17920,
        fps=25,
        width=512,
        height=512,
        frame_count=29,
        motion_frames_num=5,
    )

    assert payload == {
        "type": "session.created",
        "session_id": "sess_ws",
        "audio": {
            "sample_rate": 16000,
            "chunk_samples": 17920,
        },
        "video": {
            "fps": 25,
            "width": 512,
            "height": 512,
            "frame_count": 29,
            "motion_frames_num": 5,
        },
    }


def test_decode_session_create_writes_reference_image(tmp_path) -> None:
    image_bytes = b"fake-png"
    message = json.dumps(
        {
            "type": "session.create",
            "inputs": {"image_b64": service.base64.b64encode(image_bytes).decode("ascii")},
            "config": {"seed": 123, "audio_encode_mode": "stream"},
        }
    )

    session = service._decode_session_create(message, tmp_path)

    assert session.reference_image.read_bytes() == image_bytes
    assert session.seed == 123
    assert session.audio_encode_mode == "stream"


def test_jpeg_encoder_accepts_rgb_frames() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    jpeg = service._encode_jpeg_frame(frame)

    assert jpeg.startswith(b"\xff\xd8")
