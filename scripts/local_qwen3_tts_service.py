"""Local Qwen3-TTS HTTP/WS service (CustomVoice + instruct 情绪).

设计约定（与 local_cosyvoice_service 对齐）：
- 只用 CustomVoice 模型（预设音色 + 自然语言 instruct 情绪控制）。
  Qwen 官方把"声音复刻"与"指令控制"拆到互斥模型（Base vs CustomVoice / vc vs instruct），
  同一次合成无法克隆+情绪共存，故本服务不做复刻，只做预设音色+情绪。
- 输出统一为 little-endian int16 PCM 流（audio/L16），HTTP 分块 + WS 二进制帧两条路，
  客户端按 PCM 解码，无需再判 wav/mp3。
- 真流式（stream_generate_pcm）待 GPU 上 qwen_tts 版本确认后再接；当前先整段生成再切 PCM 帧。
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


DEFAULT_MODEL_DIRNAME = "Qwen__Qwen3-TTS-12Hz-1.7B-CustomVoice"


class SynthesizeRequest(BaseModel):
    text: str
    # voice 即 CustomVoice 预设 speaker 名（Vivian/Serena/...）。
    voice: str | None = None
    instruct: str | None = None
    language: str | None = None
    model: str | None = None
    sample_rate: int | None = None


def _audio_to_i16(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int16)
    if np.max(np.abs(arr)) > 1.5:
        return np.clip(arr, -32768, 32767).astype(np.int16)
    return np.clip(np.round(arr * 32768.0), -32768, 32767).astype(np.int16)


class Qwen3TTSService:
    def __init__(
        self,
        *,
        model_dir: str,
        device: str,
        dtype: str,
        default_speaker: str,
        language: str,
        max_new_tokens: int,
        chunk_ms: float,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.default_speaker = default_speaker
        self.language = language
        self.max_new_tokens = max_new_tokens
        self.chunk_ms = chunk_ms
        self._model: Any | None = None

    def _torch_dtype(self) -> torch.dtype:
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        return torch.bfloat16

    def model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "qwen_tts is not installed. Install the local-qwen3-tts-service extra in a separate venv."
            ) from exc
        kwargs: dict[str, Any] = {
            "device_map": self.device,
            "dtype": self._torch_dtype(),
            "attn_implementation": "sdpa",
        }
        t0 = time.perf_counter()
        self._model = Qwen3TTSModel.from_pretrained(self.model_dir, **kwargs)
        print(
            f"loaded qwen3_tts model={self.model_dir} device={self.device} seconds={time.perf_counter() - t0:.3f}",
            flush=True,
        )
        return self._model

    def _generate_pcm(self, req: SynthesizeRequest) -> tuple[np.ndarray, int]:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        speaker = (req.voice or self.default_speaker).strip()
        if not speaker:
            raise HTTPException(status_code=400, detail="voice (CustomVoice speaker) is required")
        instruct = (req.instruct or "").strip()
        kwargs: dict[str, Any] = {
            "text": text,
            "language": req.language or self.language,
            "speaker": speaker,
            "max_new_tokens": self.max_new_tokens,
        }
        # instruct 为空时不传，走自然语速；非空则由 CustomVoice 1.7B 做情绪/风格控制。
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sr = self.model().generate_custom_voice(**kwargs)
        pcm = _audio_to_i16(wavs[0])
        return pcm, int(sr)

    def synthesize_pcm_stream(self, req: SynthesizeRequest) -> tuple[Iterator[bytes], int]:
        """整段生成 → 切成 chunk_ms 的 PCM 帧迭代器。真流式后续接入 stream_generate_pcm。"""
        t0 = time.perf_counter()
        pcm, sr = self._generate_pcm(req)
        samples_per_chunk = max(1, int(sr * (self.chunk_ms / 1000.0)))
        print(
            f"synth chars={len(req.text.strip())} sr={sr} samples={pcm.size} seconds={time.perf_counter() - t0:.3f}",
            flush=True,
        )

        def generate() -> Iterator[bytes]:
            for i in range(0, pcm.size, samples_per_chunk):
                part = pcm[i : i + samples_per_chunk]
                if part.size == 0:
                    continue
                yield part.astype("<i2", copy=False).tobytes()

        return generate(), sr


def create_app(service: Qwen3TTSService) -> FastAPI:
    app = FastAPI(title="OpenTalking Local Qwen3-TTS Service")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_dir": service.model_dir,
            "device": service.device,
            "loaded": service._model is not None,
        }

    @app.post("/synthesize")
    def synthesize(req: SynthesizeRequest) -> StreamingResponse:
        try:
            stream, sr = service.synthesize_pcm_stream(req)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"qwen3_tts synth failed: {type(exc).__name__}: {exc}",
            ) from exc
        return StreamingResponse(
            stream,
            media_type=f"audio/L16; rate={sr}; channels=1",
            headers={"X-Audio-Sample-Rate": str(sr)},
        )

    @app.websocket("/synthesize/ws")
    async def synthesize_ws(ws: WebSocket) -> None:
        # WS 协议同 local_cosyvoice：payload → {"sample_rate":sr} → PCM 帧 → {"event":"done"}。
        await ws.accept()
        try:
            raw = await ws.receive_text()
            req = SynthesizeRequest(**json.loads(raw))
        except (WebSocketDisconnect, json.JSONDecodeError, TypeError, ValueError) as exc:
            await _ws_send_error(ws, f"invalid request: {exc}")
            return
        try:
            stream, sr = await asyncio.to_thread(service.synthesize_pcm_stream, req)
        except HTTPException as exc:
            await _ws_send_error(ws, str(exc.detail))
            return
        except Exception as exc:
            await _ws_send_error(ws, f"qwen3_tts synth failed: {type(exc).__name__}: {exc}")
            return
        await ws.send_text(json.dumps({"sample_rate": sr}))
        iterator = iter(stream)
        try:
            while True:
                frame = await asyncio.to_thread(next, iterator, None)
                if frame is None:
                    break
                await ws.send_bytes(frame)
            await ws.send_text(json.dumps({"event": "done"}))
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            await _ws_send_error(ws, f"stream failed: {type(exc).__name__}: {exc}")

    return app


async def _ws_send_error(ws: WebSocket, message: str) -> None:
    try:
        await ws.send_text(json.dumps({"error": message}))
    except Exception:
        pass


def build_service_from_env() -> Qwen3TTSService:
    return Qwen3TTSService(
        model_dir=os.environ.get(
            "OPENTALKING_LOCAL_QWEN3_TTS_MODEL_DIR",
            str(
                Path(os.environ.get("OPENTALKING_LOCAL_AUDIO_MODEL_ROOT", "./models/local-audio")).expanduser()
                / DEFAULT_MODEL_DIRNAME
            ),
        ),
        device=os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_DEVICE", "cuda:0"),
        dtype=os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_DTYPE", "bfloat16"),
        default_speaker=os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_SPEAKER", "Vivian"),
        language=os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_LANGUAGE", "Chinese"),
        max_new_tokens=int(os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_MAX_NEW_TOKENS", "2048")),
        chunk_ms=float(os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_CHUNK_MS", "20")),
    )


service = build_service_from_env()
app = create_app(service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Qwen3-TTS HTTP/WS service.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "19091")))
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
