from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import threading
import wave
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile as StarletteUploadFile

from opentalking.core.config import get_settings
from opentalking.providers.tts.factory import build_tts_adapter
from opentalking.providers.tts.indextts_config import normalize_indextts_config
from opentalking.providers.tts.providers import normalize_tts_provider
from opentalking.providers.tts.qwen_tts_voices import sanitize_qwen_model

router = APIRouter(prefix="/tts", tags=["tts"])
logger = logging.getLogger(__name__)

MAX_PREVIEW_TEXT_CHARS = 1000
LOCAL_COSYVOICE_PREVIEW_SECONDS = 8.0
LOCAL_COSYVOICE_PROVIDER = "local_cosyvoice"
COSYVOICE3_END_OF_PROMPT = "<|endofprompt|>"
LOCAL_COSYVOICE3_PREVIEW_INSTRUCTION = "用标准普通话中文自然朗读，不要翻译，不要使用外语。"
_INDEXTTS_PROVIDERS = {"indextts", "local_indextts", "omnirt_indextts"}
PreviewUploadFile = UploadFile | StarletteUploadFile
_LOCAL_COSYVOICE_PREVIEW_LOCK = threading.Lock()


class TTSPreviewRequest(BaseModel):
    text: Annotated[str, Field(max_length=MAX_PREVIEW_TEXT_CHARS)]
    voice: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    indextts_config: dict[str, Any] | None = None


def _preview_sample_limit(provider: str | None, sample_rate: int) -> int | None:
    if provider == LOCAL_COSYVOICE_PROVIDER:
        return max(1, int(sample_rate * LOCAL_COSYVOICE_PREVIEW_SECONDS))
    return None


def _local_cosyvoice_preview_service_url() -> str:
    return os.environ.get("OPENTALKING_TTS_LOCAL_COSYVOICE_PREVIEW_SERVICE_URL", "").strip()


def _prepare_preview_text(text: str, provider: str | None, model: str | None) -> str:
    return text


def _wav_bytes(chunks: list[np.ndarray], sample_rate: int) -> bytes:
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    pcm = np.asarray(pcm, dtype="<i2").reshape(-1)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out.getvalue()


def _trim_preview_edge_silence(
    pcm: np.ndarray,
    sample_rate: int,
    *,
    threshold: int = 300,
    frame_ms: float = 20.0,
    padding_ms: float = 80.0,
) -> np.ndarray:
    data = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if data.size == 0:
        return data
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    pad = max(0, int(sample_rate * padding_ms / 1000.0))

    start = 0
    for idx in range(0, data.size, frame):
        chunk = data[idx : idx + frame]
        if chunk.size and int(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) >= threshold:
            start = max(0, idx - pad)
            break
    else:
        return data

    end = data.size
    for idx in range(data.size, 0, -frame):
        chunk = data[max(0, idx - frame) : idx]
        if chunk.size and int(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) >= threshold:
            end = min(data.size, idx + pad)
            break
    return data[start:end]


def _normalize_preview_request(
    body: TTSPreviewRequest,
    *,
    indextts_emotion_audio_path: Path | None = None,
) -> tuple[str, str | None, str | None, str | None, dict[str, Any] | None]:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")

    try:
        provider = normalize_tts_provider(body.tts_provider, default=None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model = body.tts_model.strip() if body.tts_model and body.tts_model.strip() else None
    if provider in {"dashscope", "bailian", "qwen", "qwen_tts", "local_cosyvoice", "local_qwen3_tts"} and model:
        model = sanitize_qwen_model(model)

    voice = body.voice.strip() if body.voice and body.voice.strip() else None
    indextts_config: dict[str, Any] | None = None
    if provider in _INDEXTTS_PROVIDERS and (body.indextts_config or indextts_emotion_audio_path is not None):
        raw_config = dict(body.indextts_config or {})
        if indextts_emotion_audio_path is not None:
            raw_config["emotion_mode"] = "audio"
            raw_config["emo_audio_prompt"] = str(indextts_emotion_audio_path)
        try:
            indextts_config = normalize_indextts_config(raw_config) or None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return text, voice, provider, model, indextts_config


def _config_from_form_value(raw: object) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="indextts_config must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(status_code=422, detail="indextts_config must be a JSON object")
    return decoded


async def _preview_request_from_http(request: Request) -> tuple[TTSPreviewRequest, PreviewUploadFile | None]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type in {"multipart/form-data", "application/x-www-form-urlencoded"}:
        form = await request.form()
        upload = form.get("indextts_emotion_audio_file")
        return (
            TTSPreviewRequest(
                text=str(form.get("text") or ""),
                voice=str(form.get("voice") or "") or None,
                tts_provider=str(form.get("tts_provider") or "") or None,
                tts_model=str(form.get("tts_model") or "") or None,
                indextts_config=_config_from_form_value(form.get("indextts_config")),
            ),
            upload if isinstance(upload, (UploadFile, StarletteUploadFile)) else None,
        )
    try:
        data = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="request body must be valid JSON") from exc
    return TTSPreviewRequest(**data), None


async def _save_preview_emotion_audio(upload: PreviewUploadFile | None) -> Path | None:
    if upload is None:
        return None
    body = await upload.read()
    if not body:
        raise HTTPException(status_code=400, detail="empty IndexTTS emotion audio")
    suffix = Path(upload.filename or "emotion.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(body)
        return Path(tmp.name)


def _indextts_emotion_mode_for_log(
    *,
    raw_config: dict[str, Any] | None,
    normalized_config: dict[str, Any] | None,
    emotion_audio_path: Path | None,
) -> str:
    if not normalized_config and emotion_audio_path is None:
        return "none"
    if emotion_audio_path is not None or (normalized_config and "emo_audio_prompt" in normalized_config):
        return "audio"

    raw_mode = str((raw_config or {}).get("emotion_mode") or (raw_config or {}).get("mode") or "").strip().lower()
    if raw_mode in {"vector", "manual", "emo_vector", "normalized_vector"}:
        return raw_mode
    if raw_mode in {"text", "emo_text", "emotion_text"}:
        return "text"
    if raw_mode in {"voice", "follow_voice", "none", ""}:
        pass
    elif raw_mode:
        return raw_mode

    if normalized_config and "emo_vector" in normalized_config:
        return "vector"
    if normalized_config and normalized_config.get("use_emo_text"):
        return "text"
    return "voice"


def _log_tts_preview_request(
    *,
    provider: str | None,
    voice: str | None,
    model: str | None,
    raw_indextts_config: dict[str, Any] | None,
    indextts_config: dict[str, Any] | None,
    emotion_audio_path: Path | None,
) -> None:
    fields: list[str] = [
        f"provider={provider or 'default'}",
        f"voice_id={voice or 'default'}",
        f"model={model or 'default'}",
    ]
    if provider in _INDEXTTS_PROVIDERS:
        fields.extend(
            [
                "indextts_emotion_mode="
                + _indextts_emotion_mode_for_log(
                    raw_config=raw_indextts_config,
                    normalized_config=indextts_config,
                    emotion_audio_path=emotion_audio_path,
                ),
                f"indextts_emo_alpha={(indextts_config or {}).get('emo_alpha', 'default')}",
                f"indextts_emo_vector={(indextts_config or {}).get('emo_vector', 'default')}",
                f"indextts_use_random={(indextts_config or {}).get('use_random', 'default')}",
                f"indextts_emotion_audio_uploaded={emotion_audio_path is not None}",
            ]
        )
    logger.info("tts preview requested | %s", " ".join(fields))


@router.post("/preview", response_class=Response)
async def preview_tts(request: Request) -> Response:
    body, emotion_audio_upload = await _preview_request_from_http(request)
    emotion_audio_path = await _save_preview_emotion_audio(emotion_audio_upload)
    text, voice, provider, model, indextts_config = _normalize_preview_request(
        body,
        indextts_emotion_audio_path=emotion_audio_path,
    )
    _log_tts_preview_request(
        provider=provider,
        voice=voice,
        model=model,
        raw_indextts_config=body.indextts_config,
        indextts_config=indextts_config,
        emotion_audio_path=emotion_audio_path,
    )
    settings = get_settings()
    sample_rate = int(settings.tts_sample_rate)
    local_cosyvoice_lock_acquired = False
    if provider == LOCAL_COSYVOICE_PROVIDER:
        local_cosyvoice_lock_acquired = _LOCAL_COSYVOICE_PREVIEW_LOCK.acquire(blocking=False)
        if not local_cosyvoice_lock_acquired:
            raise HTTPException(status_code=503, detail="local_cosyvoice preview busy")
    tts = None
    chunks: list[np.ndarray] = []
    effective_sample_rate = sample_rate
    sample_limit = _preview_sample_limit(provider, sample_rate)
    drain_after_limit = provider == LOCAL_COSYVOICE_PROVIDER
    preview_text = _prepare_preview_text(text, provider, model)
    total_samples = 0
    try:
        preview_service_url = _local_cosyvoice_preview_service_url() if provider == LOCAL_COSYVOICE_PROVIDER else ""
        if preview_service_url:
            from opentalking.providers.tts.local_cosyvoice.adapter import LocalCosyVoiceTTSAdapter

            tts = LocalCosyVoiceTTSAdapter(
                default_voice=voice,
                sample_rate=sample_rate,
                chunk_ms=40.0,
                model=model,
                service_url=preview_service_url,
            )
        else:
            tts = build_tts_adapter(
                sample_rate=sample_rate,
                chunk_ms=40.0,
                default_voice=voice,
                tts_provider=provider,
                tts_model=model,
                indextts_config=indextts_config,
            )
        async for chunk in tts.synthesize_stream(preview_text, voice=voice):
            arr = np.asarray(chunk.data, dtype=np.int16).reshape(-1)
            if arr.size:
                if provider == LOCAL_COSYVOICE_PROVIDER:
                    chunks.append(arr.copy())
                elif sample_limit is None or total_samples < sample_limit:
                    if sample_limit is not None and total_samples + int(arr.size) > sample_limit:
                        arr = arr[: sample_limit - total_samples]
                    chunks.append(arr.copy())
                total_samples += int(arr.size)
            effective_sample_rate = int(chunk.sample_rate or effective_sample_rate)
            if sample_limit is not None and total_samples >= sample_limit and not drain_after_limit:
                break
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS preview failed: {exc}") from exc
    finally:
        close = getattr(tts, "aclose", None) if tts is not None else None
        if close is not None:
            await close()
        if local_cosyvoice_lock_acquired:
            _LOCAL_COSYVOICE_PREVIEW_LOCK.release()
        if emotion_audio_path is not None:
            emotion_audio_path.unlink(missing_ok=True)

    if provider == LOCAL_COSYVOICE_PROVIDER and chunks:
        pcm = _trim_preview_edge_silence(np.concatenate(chunks), effective_sample_rate)
        sample_limit = _preview_sample_limit(provider, effective_sample_rate)
        if sample_limit is not None and pcm.size > sample_limit:
            pcm = pcm[:sample_limit]
        chunks = [pcm] if pcm.size else []

    if not chunks:
        raise HTTPException(status_code=502, detail="TTS preview returned no audio")

    return Response(
        content=_wav_bytes(chunks, effective_sample_rate),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )
