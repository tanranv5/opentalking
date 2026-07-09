from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import numpy as np

from opentalking.core.types.frames import AudioChunk
from opentalking.providers.tts._ws_stream import is_ws_url, ws_pcm_stream


END_OF_PROMPT = "<|endofprompt|>"


def _settings_value(name: str, default: str = "") -> str:
    try:
        from opentalking.core.config import get_settings

        value = getattr(get_settings(), name, default)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return default


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if pcm.size == 0 or src_sr == dst_sr:
        return pcm.copy()
    pcm_f = pcm.astype(np.float32) / 32768.0
    n_dst = max(1, int(round(pcm.size * dst_sr / src_sr)))
    xi = np.linspace(0.0, pcm.size - 1.0, num=n_dst)
    out = np.interp(xi, np.arange(pcm.size), pcm_f)
    return np.clip(np.round(out * 32768.0), -32768, 32767).astype(np.int16)


def _split_pcm_chunks(pcm: np.ndarray, sr: int, chunk_ms: float) -> list[AudioChunk]:
    samples_per_chunk = max(1, int(sr * (chunk_ms / 1000.0)))
    out: list[AudioChunk] = []
    for i in range(0, len(pcm), samples_per_chunk):
        part = pcm[i : i + samples_per_chunk]
        if part.size == 0:
            continue
        out.append(AudioChunk(data=part.astype(np.int16), sample_rate=sr, duration_ms=1000.0 * part.size / sr))
    return out


def _source_sample_rate_from_headers(headers: Any, fallback: int) -> int:
    direct = str(headers.get("x-audio-sample-rate", "") or "").strip()
    if direct.isdigit():
        return int(direct)
    content_type = str(headers.get("content-type", "") or "")
    for part in content_type.split(";")[1:]:
        key, sep, value = part.strip().partition("=")
        if sep and key.strip().lower() == "rate" and value.strip().isdigit():
            return int(value.strip())
    return fallback


def _split_instruction(text: str) -> tuple[str, str]:
    """把 Brain 传来的 ``情绪指令<|endofprompt|>台词`` 拆成 (台词, 情绪指令)。

    无分隔符时整段视为台词、无 instruct。CustomVoice 的 instruct 是自然语言语气描述，
    与 CosyVoice 的 ``用XX语气说`` 格式一致，故可直接复用同一约定。
    """
    raw = text.strip()
    if END_OF_PROMPT in raw:
        instruction, _, reply = raw.partition(END_OF_PROMPT)
        instruction = instruction.strip()
        reply = reply.strip() or raw
        return reply, instruction
    return raw, ""


class LocalQwen3TTSAdapter:
    """Same-host local Qwen3-TTS (CustomVoice) adapter with HTTP/WS transport.

    只支持 CustomVoice 预设音色 + instruct 情绪；不做声音复刻（Qwen 复刻走 Base，
    与 instruct 互斥，见 scripts/local_qwen3_tts_service.py 说明）。
    传输按 service_url scheme 自动选择：ws(s):// 走 WebSocket，否则 HTTP 分块流式。
    """

    def __init__(
        self,
        default_voice: str | None = None,
        sample_rate: int = 16000,
        chunk_ms: float = 20.0,
        *,
        model: str | None = None,
        language: str | None = None,
    ) -> None:
        self.default_voice = default_voice or "Vivian"
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.model = (
            model
            or os.environ.get("OPENTALKING_TTS_LOCAL_QWEN3_TTS_MODEL")
            or os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_MODEL")
            or _settings_value("local_qwen3_tts_model", "")
            or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ).strip()
        self.language = (
            (language or "").strip()
            or os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_LANGUAGE", "").strip()
            or _settings_value("local_qwen3_tts_language", "")
            or "Chinese"
        )
        self.service_url = (
            os.environ.get("OPENTALKING_TTS_LOCAL_QWEN3_TTS_SERVICE_URL", "").strip()
            or os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_SERVICE_URL", "").strip()
            or _settings_value("local_qwen3_tts_service_url", "")
        )

    def _build_payload(self, text: str, voice: str | None) -> dict[str, Any] | None:
        reply, instruction = _split_instruction(text)
        try:
            from opentalking.pipeline.speak.text_sanitize import sanitize_tts_text

            tts_text = sanitize_tts_text(reply)
        except Exception:
            tts_text = reply
        if not tts_text:
            return None
        payload: dict[str, Any] = {
            "text": tts_text,
            "voice": voice or self.default_voice,
            "model": self.model,
            "language": self.language,
            "sample_rate": self.sample_rate,
        }
        if instruction:
            payload["instruct"] = instruction
        return payload

    async def synthesize_stream(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        if not text.strip():
            return
        if not self.service_url:
            raise RuntimeError(
                "Local Qwen3-TTS requires OPENTALKING_LOCAL_QWEN3_TTS_SERVICE_URL. "
                "Run scripts/local_qwen3_tts_service.py and point this variable at it "
                "(http(s):// for HTTP chunked, ws(s):// for WebSocket)."
            )
        payload = self._build_payload(text, voice)
        if payload is None:
            return
        if is_ws_url(self.service_url):
            async for chunk in ws_pcm_stream(self.service_url, payload, self.sample_rate, self.chunk_ms):
                yield chunk
            return
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        pending = b""
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", self.service_url, json=payload) as resp:
                resp.raise_for_status()
                source_sr = _source_sample_rate_from_headers(resp.headers, self.sample_rate)
                async for data in resp.aiter_bytes():
                    if not data:
                        continue
                    data = pending + data
                    if len(data) % 2:
                        pending = data[-1:]
                        data = data[:-1]
                    else:
                        pending = b""
                    if not data:
                        continue
                    pcm = np.frombuffer(data, dtype="<i2").astype(np.int16, copy=False)
                    pcm = _resample_linear(pcm, source_sr, self.sample_rate)
                    for chunk in _split_pcm_chunks(pcm, self.sample_rate, self.chunk_ms):
                        yield chunk
