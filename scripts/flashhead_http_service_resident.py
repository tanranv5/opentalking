from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import struct
import subprocess
import sys
import time
import traceback
import uuid
import wave
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

MAGIC_AUDIO = b"AUDI"
MAGIC_VIDEO = b"VIDX"

REPO_DIR = Path(os.environ.get("FLASHHEAD_REPO_DIR", "/root/test/model-repos/SoulX-FlashHead")).resolve()
CKPT_DIR = os.environ.get("FLASHHEAD_CKPT_DIR", "/root/test/models/SoulX-FlashHead-1_3B")
WAV2VEC_DIR = os.environ.get("FLASHHEAD_WAV2VEC_DIR", "/root/test/models/wav2vec2-base-960h")
OUTPUT_DIR = Path(os.environ.get("FLASHHEAD_OUTPUT_DIR", "/tmp/opentalking_flashhead_io/outputs")).resolve()
SESSION_DIR = Path(os.environ.get("FLASHHEAD_SESSION_DIR", "/tmp/opentalking_flashhead_io/sessions")).resolve()
MODEL_TYPE = os.environ.get("FLASHHEAD_MODEL_TYPE", "pro")
PRELOAD = os.environ.get("FLASHHEAD_PRELOAD", "1").strip().lower() in {"1", "true", "yes", "on"}
BASE_CACHE_MAX_ITEMS = max(0, int(os.environ.get("FLASHHEAD_BASE_CACHE_MAX_ITEMS", "8")))
JPEG_QUALITY = max(1, min(100, int(os.environ.get("FLASHHEAD_WS_JPEG_QUALITY", "80"))))

_pipeline = None
_loaded_key: tuple[str, str, str] | None = None
_loaded_at = 0.0
_lock = Lock()
_base_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()

_runtime_loaded = False
imageio = None
librosa = None
np = None
torch = None
get_audio_embedding = None
get_base_data = None
get_infer_params = None
get_pipeline = None
run_pipeline = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if PRELOAD:
        await asyncio.to_thread(_ensure_pipeline)
    yield


app = FastAPI(title="FlashHead resident HTTP/WS adapter", version="0.3.0", lifespan=_lifespan)


class GenerateRequest(BaseModel):
    task: str = Field(default="audio2video")
    model: str = Field(default="soulx-flashhead-1.3b")
    inputs: dict[str, Any]
    config: dict[str, Any] = Field(default_factory=dict)


class PrewarmRequest(BaseModel):
    inputs: dict[str, Any]
    config: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RealtimeSession:
    session_id: str
    session_dir: Path
    reference_image: Path
    seed: int
    use_face_crop: bool
    audio_encode_mode: str
    sample_rate: int = 16000
    fps: int = 25
    width: int = 512
    height: int = 512
    frame_count: int = 29
    motion_frames_num: int = 5
    chunk_index: int = 0

    @property
    def chunk_samples(self) -> int:
        slice_len = max(1, self.frame_count - self.motion_frames_num)
        return int(slice_len * self.sample_rate // max(1, self.fps))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "resident",
        "protocols": ["http", "websocket"],
        "ws_path": "/v1/avatar/realtime",
        "repo_dir": str(REPO_DIR),
        "ckpt_dir": CKPT_DIR,
        "wav2vec_dir": WAV2VEC_DIR,
        "model_type": MODEL_TYPE,
        "pipeline_loaded": _pipeline is not None,
        "loaded_at": _loaded_at,
        "base_cache_items": len(_base_cache),
        "base_cache_max_items": BASE_CACHE_MAX_ITEMS,
    }


@app.post("/v1/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    if req.task != "audio2video":
        raise HTTPException(status_code=400, detail=f"unsupported task: {req.task}")
    image = _required_path(req.inputs, "image")
    audio = _required_path(req.inputs, "audio")
    if not image.is_file():
        raise HTTPException(status_code=400, detail=f"image not found: {image}")
    if not audio.is_file():
        raise HTTPException(status_code=400, detail=f"audio not found: {audio}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file = OUTPUT_DIR / f"flashhead_resident_{int(time.time())}_{uuid.uuid4().hex[:12]}.mp4"
    seed = int(req.config.get("seed") or req.config.get("base_seed") or 999)
    audio_encode_mode = str(req.config.get("audio_encode_mode") or "stream")
    use_face_crop = bool(req.config.get("use_face_crop") or False)

    started = time.monotonic()
    with _lock:
        pipeline = _ensure_pipeline()
        try:
            metrics = _run_generation(
                pipeline=pipeline,
                cond_image=str(image),
                audio_path=str(audio),
                save_file=str(save_file),
                audio_encode_mode=audio_encode_mode,
                seed=seed,
                use_face_crop=use_face_crop,
            )
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"FlashHead resident generation failed: {exc}") from exc
    elapsed = time.monotonic() - started
    return {
        "outputs": [{"path": str(save_file), "type": "video/mp4"}],
        "artifact": {"path": str(save_file)},
        "elapsed_sec": elapsed,
        "model": req.model,
        "mode": "resident",
        "base_cache": metrics.get("base_cache"),
    }


@app.post("/v1/prewarm")
def prewarm(req: PrewarmRequest) -> dict[str, Any]:
    images = _prewarm_images(req.inputs)
    seed = int(req.config.get("seed") or req.config.get("base_seed") or 999)
    use_face_crop = bool(req.config.get("use_face_crop") or False)

    results = []
    started = time.monotonic()
    with _lock:
        pipeline = _ensure_pipeline()
        for image in images:
            if not image.is_file():
                raise HTTPException(status_code=400, detail=f"image not found: {image}")
            try:
                status = _prepare_base_data_cached(
                    pipeline=pipeline,
                    cond_image=str(image),
                    seed=seed,
                    use_face_crop=use_face_crop,
                )
            except Exception as exc:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"FlashHead resident prewarm failed: {exc}") from exc
            results.append({"image": str(image), "base_cache": status})
    return {
        "ok": True,
        "mode": "resident",
        "elapsed_sec": time.monotonic() - started,
        "results": results,
        "base_cache_items": len(_base_cache),
    }


@app.websocket("/v1/avatar/realtime")
async def realtime_avatar(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        session = _decode_session_create(raw, SESSION_DIR / f"ws_{uuid.uuid4().hex[:12]}")
        await asyncio.to_thread(_prepare_realtime_session, session)
        await websocket.send_text(json.dumps(_session_created_payload(
            session_id=session.session_id,
            sample_rate=session.sample_rate,
            chunk_samples=session.chunk_samples,
            fps=session.fps,
            width=session.width,
            height=session.height,
            frame_count=session.frame_count,
            motion_frames_num=session.motion_frames_num,
        ), ensure_ascii=False))

        while True:
            message = await websocket.receive()
            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                return
            text = message.get("text")
            if text is not None:
                if _is_session_close_message(text):
                    await websocket.send_text(json.dumps({"type": "session.closed", "session_id": session.session_id}))
                    return
                await websocket.send_text(json.dumps({"type": "error", "message": "unexpected text message"}))
                continue
            payload = message.get("bytes")
            if payload is None:
                continue
            if len(payload) < 4 or payload[:4] != MAGIC_AUDIO:
                await websocket.send_text(json.dumps({"type": "error", "message": "invalid audio payload"}))
                continue
            video_payload = await asyncio.to_thread(_generate_realtime_video_payload, session, payload[4:])
            await websocket.send_bytes(video_payload)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False))
        except Exception:
            return


def _load_runtime_modules() -> None:
    global _runtime_loaded
    global imageio, librosa, np, torch
    global get_audio_embedding, get_base_data, get_infer_params, get_pipeline, run_pipeline
    if _runtime_loaded:
        return
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)
    import imageio as imageio_module
    import librosa as librosa_module
    import numpy as np_module
    import torch as torch_module
    from flash_head.inference import (
        get_audio_embedding as runtime_get_audio_embedding,
        get_base_data as runtime_get_base_data,
        get_infer_params as runtime_get_infer_params,
        get_pipeline as runtime_get_pipeline,
        run_pipeline as runtime_run_pipeline,
    )

    imageio = imageio_module
    librosa = librosa_module
    np = np_module
    torch = torch_module
    get_audio_embedding = runtime_get_audio_embedding
    get_base_data = runtime_get_base_data
    get_infer_params = runtime_get_infer_params
    get_pipeline = runtime_get_pipeline
    run_pipeline = runtime_run_pipeline
    _runtime_loaded = True


def _ensure_pipeline():
    global _pipeline, _loaded_key, _loaded_at
    _load_runtime_modules()
    key = (CKPT_DIR, WAV2VEC_DIR, MODEL_TYPE)
    if _pipeline is None or _loaded_key != key:
        _pipeline = get_pipeline(world_size=1, ckpt_dir=CKPT_DIR, model_type=MODEL_TYPE, wav2vec_dir=WAV2VEC_DIR)
        _loaded_key = key
        _loaded_at = time.time()
    return _pipeline


def _run_generation(
    *,
    pipeline,
    cond_image: str,
    audio_path: str,
    save_file: str,
    audio_encode_mode: str,
    seed: int,
    use_face_crop: bool,
) -> dict[str, str]:
    frames_list, base_cache_status, fps = _generate_frame_arrays(
        pipeline=pipeline,
        cond_image=cond_image,
        audio_path=audio_path,
        audio_encode_mode=audio_encode_mode,
        seed=seed,
        use_face_crop=use_face_crop,
    )
    _save_video(frames_list, save_file, audio_path, fps=fps)
    return {"base_cache": base_cache_status}


def _generate_frame_arrays(
    *,
    pipeline,
    cond_image: str,
    audio_path: str,
    audio_encode_mode: str,
    seed: int,
    use_face_crop: bool,
) -> tuple[list[Any], str, int]:
    _load_runtime_modules()
    base_cache_status = _prepare_base_data_cached(
        pipeline=pipeline,
        cond_image=cond_image,
        seed=seed,
        use_face_crop=use_face_crop,
    )
    infer_params = get_infer_params()
    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num

    human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    human_speech_array_frame_num = frame_num * sample_rate // tgt_fps
    generated_list = []

    if audio_encode_mode == "once":
        remainder = (len(human_speech_array_all) - human_speech_array_frame_num) % human_speech_array_slice_len
        if remainder > 0:
            human_speech_array_all = np.concatenate([
                human_speech_array_all,
                np.zeros(human_speech_array_slice_len - remainder, dtype=human_speech_array_all.dtype),
            ])
        audio_embedding_all = get_audio_embedding(pipeline, human_speech_array_all)
        chunks = [
            audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous()
            for i in range((audio_embedding_all.shape[1] - frame_num) // slice_len)
        ]
        for chunk_idx, audio_embedding_chunk in enumerate(chunks):
            torch.cuda.synchronize()
            video = run_pipeline(pipeline, audio_embedding_chunk)
            if chunk_idx != 0:
                video = video[motion_frames_num:]
            torch.cuda.synchronize()
            generated_list.append(video.cpu().numpy().astype(np.uint8))
    elif audio_encode_mode == "stream":
        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num
        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
        remainder = len(human_speech_array_all) % human_speech_array_slice_len
        if remainder > 0:
            human_speech_array_all = np.concatenate([
                human_speech_array_all,
                np.zeros(human_speech_array_slice_len - remainder, dtype=human_speech_array_all.dtype),
            ])
        slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)
        for human_speech_array in slices:
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
            torch.cuda.synchronize()
            video = run_pipeline(pipeline, audio_embedding)
            video = video[motion_frames_num:]
            torch.cuda.synchronize()
            generated_list.append(video.cpu().numpy().astype(np.uint8))
    else:
        raise ValueError(f"unsupported audio_encode_mode: {audio_encode_mode}")

    return generated_list, base_cache_status, int(tgt_fps)


def _prepare_base_data_cached(*, pipeline, cond_image: str, seed: int, use_face_crop: bool) -> str:
    _load_runtime_modules()
    if BASE_CACHE_MAX_ITEMS <= 0:
        get_base_data(pipeline, cond_image_path_or_dir=cond_image, base_seed=seed, use_face_crop=use_face_crop)
        return "disabled"

    key = _base_cache_key(cond_image, seed=seed, use_face_crop=use_face_crop)
    cached = _base_cache.get(key)
    if cached is not None:
        _base_cache.move_to_end(key)
        _restore_base_data(pipeline, cached, seed=seed)
        return "hit"

    get_base_data(pipeline, cond_image_path_or_dir=cond_image, base_seed=seed, use_face_crop=use_face_crop)
    _base_cache[key] = _capture_base_data(pipeline)
    _base_cache.move_to_end(key)
    while len(_base_cache) > BASE_CACHE_MAX_ITEMS:
        _base_cache.popitem(last=False)
    return "miss"


def _base_cache_key(cond_image: str, *, seed: int, use_face_crop: bool) -> tuple[Any, ...]:
    image = Path(cond_image).resolve()
    stat = image.stat()
    infer_params = get_infer_params()
    return (
        str(image),
        stat.st_mtime_ns,
        stat.st_size,
        int(seed),
        bool(use_face_crop),
        MODEL_TYPE,
        int(infer_params["height"]),
        int(infer_params["width"]),
        int(infer_params["frame_num"]),
        int(infer_params["motion_frames_num"]),
    )


def _capture_base_data(pipeline) -> dict[str, Any]:
    return {
        "cond_image_dict": pipeline.cond_image_dict,
        "frame_num": pipeline.frame_num,
        "motion_frames_num": pipeline.motion_frames_num,
        "color_correction_strength": pipeline.color_correction_strength,
        "target_h": pipeline.target_h,
        "target_w": pipeline.target_w,
        "lat_h": pipeline.lat_h,
        "lat_w": pipeline.lat_w,
        "timesteps": pipeline.timesteps,
        "cond_image_tensor_dict": pipeline.cond_image_tensor_dict,
        "ref_img_latent_dict": pipeline.ref_img_latent_dict,
        "person_name": pipeline.person_name,
        "original_color_reference": pipeline.original_color_reference,
        "ref_img_latent": pipeline.ref_img_latent,
        "initial_latent_motion_frames": pipeline.ref_img_latent[:, :1].clone(),
    }


def _restore_base_data(pipeline, cached: dict[str, Any], *, seed: int) -> None:
    pipeline.cond_image_dict = cached["cond_image_dict"]
    pipeline.frame_num = cached["frame_num"]
    pipeline.motion_frames_num = cached["motion_frames_num"]
    pipeline.color_correction_strength = cached["color_correction_strength"]
    pipeline.target_h = cached["target_h"]
    pipeline.target_w = cached["target_w"]
    pipeline.lat_h = cached["lat_h"]
    pipeline.lat_w = cached["lat_w"]
    pipeline.generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    pipeline.timesteps = cached["timesteps"]
    pipeline.cond_image_tensor_dict = cached["cond_image_tensor_dict"]
    pipeline.ref_img_latent_dict = cached["ref_img_latent_dict"]
    pipeline.person_name = cached["person_name"]
    pipeline.original_color_reference = cached["original_color_reference"]
    pipeline.ref_img_latent = cached["ref_img_latent"]
    pipeline.latent_motion_frames = cached["initial_latent_motion_frames"].clone()


def _save_video(frames_list: list[Any], video_path: str, audio_path: str, fps: int) -> None:
    if not frames_list:
        raise ValueError("FlashHead generated no frames")
    temp_video_path = video_path.replace(".mp4", "_tmp.mp4")
    with imageio.get_writer(
        temp_video_path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0"],
    ) as writer:
        for frames in frames_list:
            for i in range(frames.shape[0]):
                writer.append_data(frames[i, :, :, :])
    cmd = ["ffmpeg", "-i", temp_video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", video_path, "-y"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg merge failed: {exc.stdout}") from exc
    os.remove(temp_video_path)


def _prepare_realtime_session(session: RealtimeSession) -> None:
    with _lock:
        pipeline = _ensure_pipeline()
        _prepare_base_data_cached(
            pipeline=pipeline,
            cond_image=str(session.reference_image),
            seed=session.seed,
            use_face_crop=session.use_face_crop,
        )
        params = get_infer_params()
        session.sample_rate = int(params["sample_rate"])
        session.fps = int(params["tgt_fps"])
        session.width = int(params["width"])
        session.height = int(params["height"])
        session.frame_count = int(params["frame_num"])
        session.motion_frames_num = int(params["motion_frames_num"])


def _generate_realtime_video_payload(session: RealtimeSession, pcm_bytes: bytes) -> bytes:
    audio_path = _write_pcm_wav(session, pcm_bytes)
    with _lock:
        pipeline = _ensure_pipeline()
        frames_list, _, _ = _generate_frame_arrays(
            pipeline=pipeline,
            cond_image=str(session.reference_image),
            audio_path=str(audio_path),
            audio_encode_mode=session.audio_encode_mode,
            seed=session.seed,
            use_face_crop=session.use_face_crop,
        )
    jpeg_parts: list[bytes] = []
    for frames in frames_list:
        for i in range(frames.shape[0]):
            jpeg_parts.append(_encode_jpeg_frame(frames[i, :, :, :]))
    return _pack_video_payload(jpeg_parts)


def _write_pcm_wav(session: RealtimeSession, pcm_bytes: bytes) -> Path:
    session.chunk_index += 1
    path = session.session_dir / f"chunk_{session.chunk_index:08d}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(session.sample_rate)
        wf.writeframes(pcm_bytes)
    return path


def _pack_video_payload(jpeg_parts: list[bytes]) -> bytes:
    chunks = [MAGIC_VIDEO, struct.pack("<I", len(jpeg_parts))]
    for jpeg in jpeg_parts:
        chunks.append(struct.pack("<I", len(jpeg)))
        chunks.append(jpeg)
    return b"".join(chunks)


def _encode_jpeg_frame(frame: Any) -> bytes:
    import cv2
    import numpy as np_module

    arr = np_module.asarray(frame, dtype=np_module.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected RGB frame with shape HxWx3, got {arr.shape!r}")
    bgr = np_module.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def _session_created_payload(
    *,
    session_id: str,
    sample_rate: int,
    chunk_samples: int,
    fps: int,
    width: int,
    height: int,
    frame_count: int,
    motion_frames_num: int,
) -> dict[str, Any]:
    return {
        "type": "session.created",
        "session_id": session_id,
        "audio": {
            "sample_rate": sample_rate,
            "chunk_samples": chunk_samples,
        },
        "video": {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "motion_frames_num": motion_frames_num,
        },
    }


def _decode_session_create(raw_message: str, session_dir: Path) -> RealtimeSession:
    msg = json.loads(raw_message)
    if not isinstance(msg, dict) or msg.get("type") != "session.create":
        raise ValueError("expected session.create message")
    inputs = msg.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("missing inputs")
    image_b64 = str(inputs.get("image_b64") or "").strip()
    if not image_b64:
        raise ValueError("missing inputs.image_b64")
    config = msg.get("config")
    if not isinstance(config, dict):
        config = {}
    session_dir.mkdir(parents=True, exist_ok=True)
    reference_image = session_dir / "reference.png"
    reference_image.write_bytes(base64.b64decode(image_b64))
    return RealtimeSession(
        session_id=str(msg.get("session_id") or uuid.uuid4().hex),
        session_dir=session_dir,
        reference_image=reference_image,
        seed=int(config.get("seed") or config.get("base_seed") or 999),
        use_face_crop=bool(config.get("use_face_crop") or False),
        audio_encode_mode=str(config.get("audio_encode_mode") or "stream"),
    )


def _is_session_close_message(raw_message: str) -> bool:
    try:
        msg = json.loads(raw_message)
    except json.JSONDecodeError:
        return False
    return isinstance(msg, dict) and msg.get("type") == "session.close"


def _prewarm_images(inputs: dict[str, Any]) -> list[Path]:
    raw_images = inputs.get("images")
    if isinstance(raw_images, list):
        paths = [Path(str(item)).expanduser().resolve() for item in raw_images if str(item).strip()]
        if paths:
            return paths
    return [_required_path(inputs, "image")]


def _required_path(inputs: dict[str, Any], key: str) -> Path:
    raw = inputs.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(status_code=400, detail=f"missing inputs.{key}")
    return Path(raw).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashHead resident HTTP/WS adapter")
    parser.add_argument("--host", default=os.environ.get("FLASHHEAD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FLASHHEAD_PORT", "8766")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
