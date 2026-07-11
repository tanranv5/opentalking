"""Local Qwen3-TTS HTTP/WS service (CustomVoice + instruct) — faster-qwen3-tts edition.

Uses faster_qwen3_tts.FasterQwen3TTS with CUDA graph acceleration and
true streaming generation via generate_custom_voice_streaming().
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
from typing import Any, Generator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


DEFAULT_MODEL_DIRNAME = "Qwen__Qwen3-TTS-12Hz-1.7B-CustomVoice"


class SynthesizeRequest(BaseModel):
    text: str
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
        streaming_chunk_size: int,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.default_speaker = default_speaker
        self.language = language
        self.max_new_tokens = max_new_tokens
        self.chunk_ms = chunk_ms
        self.streaming_chunk_size = streaming_chunk_size
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
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as exc:
            raise RuntimeError(
                "faster_qwen3_tts is not installed."
            ) from exc
        t0 = time.perf_counter()
        self._model = FasterQwen3TTS.from_pretrained(
            self.model_dir,
            device=self.device,
            dtype=self._torch_dtype(),
            attn_implementation="sdpa",
        )
        elapsed = time.perf_counter() - t0
        print(
            f"loaded faster_qwen3_tts model={self.model_dir} device={self.device} seconds={elapsed:.3f}",
            flush=True,
        )
        # Warmup: first inference captures CUDA graphs (~10-20s)
        print("warming up CUDA graphs (first inference)...", flush=True)
        t1 = time.perf_counter()
        self._model.generate_custom_voice(
            text="warmup",
            language="Chinese",
            speaker=self.default_speaker,
        )
        print(
            f"warmup done seconds={time.perf_counter() - t1:.3f}",
            flush=True,
        )
        return self._model

    def synthesize_pcm_stream(self, req: SynthesizeRequest) -> tuple[Generator[bytes, None, None], int]:
        """True streaming: yield PCM i16 frames as they are generated."""
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        speaker = (req.voice or self.default_speaker).strip()
        if not speaker:
            raise HTTPException(status_code=400, detail="voice (CustomVoice speaker) is required")
        instruct = (req.instruct or "").strip() or None

        t0 = time.perf_counter()
        chunk_count = 0
        total_samples = 0
        detected_sr = [0]

        def generate() -> Generator[bytes, None, None]:
            nonlocal chunk_count, total_samples
            kwargs: dict[str, Any] = {
                "text": text,
                "language": req.language or self.language,
                "speaker": speaker,
                "max_new_tokens": self.max_new_tokens,
                "chunk_size": self.streaming_chunk_size,
            }
            if instruct:
                kwargs["instruct"] = instruct

            first_chunk_logged = False
            for audio_chunk, sr, timing in self.model().generate_custom_voice_streaming(**kwargs):
                detected_sr[0] = int(sr)
                pcm = _audio_to_i16(audio_chunk)
                if pcm.size == 0:
                    continue
                total_samples += pcm.size
                chunk_count += 1
                if not first_chunk_logged:
                    first_chunk_logged = True
                    first_byte_ms = (time.perf_counter() - t0) * 1000
                    print(
                        f"streaming first_byte_ms={first_byte_ms:.0f} prefill_ms={timing.get('prefill_ms', 0):.0f}",
                        flush=True,
                    )
                # Sub-chunk into chunk_ms sized PCM frames for WS/HTTP compatibility
                samples_per_frame = max(1, int(sr * (self.chunk_ms / 1000.0)))
                for i in range(0, pcm.size, samples_per_frame):
                    part = pcm[i : i + samples_per_frame]
                    if part.size > 0:
                        yield part.astype("<i2", copy=False).tobytes()

            elapsed = time.perf_counter() - t0
            print(
                f"synth chars={len(text)} sr={detected_sr[0]} samples={total_samples} "
                f"chunks={chunk_count} seconds={elapsed:.3f}",
                flush=True,
            )

        # We need to know SR before we start streaming. The model's sample rate is
        # deterministic (24000 for Qwen3-TTS 12Hz), so we read it from the loaded model.
        sr = getattr(self.model(), "sample_rate", 24000)
        return generate(), int(sr)


def create_app(service: Qwen3TTSService) -> FastAPI:
    app = FastAPI(title="OpenTalking Local Qwen3-TTS Service (faster-qwen3-tts)")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_dir": service.model_dir,
            "device": service.device,
            "loaded": service._model is not None,
            "backend": "faster_qwen3_tts",
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
        try:
            while True:
                frame = await asyncio.to_thread(next, stream, None)
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
        streaming_chunk_size=int(os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_STREAMING_CHUNK_SIZE", "8")),
    )


service = build_service_from_env()
app = create_app(service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Qwen3-TTS HTTP/WS service (faster-qwen3-tts).")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "19091")))
    args = parser.parse_args()
    # 启动即加载模型并捕获 CUDA graph，避免首次会话开场白承担 15s+ 冷启动首包。
    # 端口在预热完成后才打开，start_stack 的 /health 等待因此天然覆盖预热窗口。
    if os.environ.get("OPENTALKING_LOCAL_QWEN3_TTS_EAGER_LOAD", "1") != "0":
        service.model()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
