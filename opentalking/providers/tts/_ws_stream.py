"""Reusable local WebSocket PCM streaming client for TTS adapters.

Protocol (local same-host services, not DashScope cloud):
  1. Client connects to ``ws_url`` and sends a single JSON text frame = the
     synthesize request payload (same shape as the HTTP ``/synthesize`` body).
  2. Server MAY reply with a leading JSON text frame carrying stream metadata,
     e.g. ``{"sample_rate": 24000}``, to declare the source PCM sample rate.
  3. Server streams raw little-endian int16 PCM as binary frames.
  4. Server ends the stream by sending a JSON text frame ``{"event": "done"}``
     or by closing the connection.

The client resamples to ``sample_rate`` and yields fixed ``chunk_ms`` AudioChunks,
matching the HTTP path so the speak pipeline sees identical output either way.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from opentalking.core.types.frames import AudioChunk


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
        out.append(
            AudioChunk(
                data=part.astype(np.int16),
                sample_rate=sr,
                duration_ms=1000.0 * part.size / sr,
            )
        )
    return out


def is_ws_url(url: str | None) -> bool:
    """Return True when ``url`` selects the WebSocket transport (ws:// or wss://)."""
    value = (url or "").strip().lower()
    return value.startswith("ws://") or value.startswith("wss://")


def _source_sr_from_meta(meta: Any, fallback: int) -> int:
    if not isinstance(meta, dict):
        return fallback
    for key in ("sample_rate", "source_sample_rate", "rate"):
        value = meta.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return fallback


async def ws_pcm_stream(
    ws_url: str,
    payload: dict[str, Any],
    sample_rate: int,
    chunk_ms: float,
) -> AsyncIterator[AudioChunk]:
    """Stream PCM AudioChunks from a local TTS WebSocket service.

    :param ws_url: ``ws(s)://host:port/path`` of the local synthesize endpoint.
    :param payload: JSON-serializable synthesize request (text/voice/instruct/...).
    :param sample_rate: target output sample rate.
    :param chunk_ms: emitted AudioChunk duration in milliseconds.
    """
    import websockets

    source_sr = sample_rate
    pending = b""
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps(payload, ensure_ascii=False))
        async for message in ws:
            if isinstance(message, str):
                try:
                    meta = json.loads(message)
                except (ValueError, TypeError):
                    continue
                if isinstance(meta, dict) and meta.get("event") == "done":
                    break
                if isinstance(meta, dict) and meta.get("error"):
                    raise RuntimeError(f"WS TTS error: {meta.get('error')}")
                source_sr = _source_sr_from_meta(meta, source_sr)
                continue
            data = pending + bytes(message)
            if len(data) % 2:
                pending = data[-1:]
                data = data[:-1]
            else:
                pending = b""
            if not data:
                continue
            pcm = np.frombuffer(data, dtype="<i2").astype(np.int16, copy=False)
            pcm = _resample_linear(pcm, source_sr, sample_rate)
            for chunk in _split_pcm_chunks(pcm, sample_rate, chunk_ms):
                yield chunk
    if pending:
        # 尾部残留奇数字节丢弃（半个采样点无意义），保持与 HTTP 路径一致。
        pending = b""
