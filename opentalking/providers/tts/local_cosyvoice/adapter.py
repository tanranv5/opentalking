from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
import wave
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from opentalking.core.types.frames import AudioChunk


COSYVOICE3_END_OF_PROMPT = "<|endofprompt|>"
DEFAULT_COSYVOICE3_INSTRUCTION = "请始终使用标准普通话中文自然清晰地朗读，不要翻译，不要使用英文、外文、颜文字或表情符号。"


def _settings_value(name: str, default: str = "") -> str:
    try:
        from opentalking.core.config import get_settings

        value = getattr(get_settings(), name, default)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return default


def _model_aliases(model: str | None) -> set[str]:
    value = str(model or "").strip().strip("/").replace("\\", "/")
    if not value:
        return set()
    aliases = {value, value.replace("/", "__")}
    name = Path(value).name
    if name:
        aliases.add(name)
        if "__" in name:
            aliases.add(name.replace("__", "/"))
    return aliases


def _models_match(left: str | None, right: str | None) -> bool:
    return bool(_model_aliases(left) & _model_aliases(right))


def _parse_service_url_map(raw: str) -> dict[str, str]:
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("local CosyVoice service URL map must be a JSON object")
        return {str(k).strip(): str(v).strip() for k, v in parsed.items() if str(k).strip() and str(v).strip()}
    out: dict[str, str] = {}
    for entry in text.replace("\n", ",").replace(";", ",").split(","):
        item = entry.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"invalid local CosyVoice service URL map entry: {item!r}")
        key = key.strip()
        value = value.strip()
        if key and value:
            out[key] = value
    return out


def _service_url_map() -> dict[str, str]:
    raw = (
        os.environ.get("OPENTALKING_TTS_LOCAL_COSYVOICE_SERVICE_URLS", "").strip()
        or _settings_value("tts_local_cosyvoice_service_urls", "")
    )
    return _parse_service_url_map(raw)


def _resolve_service_url_for_model(model: str, default_model: str, default_url: str) -> str:
    urls = _service_url_map()
    if not urls:
        return default_url
    for key, url in urls.items():
        if _models_match(model, key):
            return url
    if default_url and _models_match(model, default_model):
        return default_url
    raise RuntimeError(
        f"No local CosyVoice service URL configured for model {model!r}. "
        "Set OPENTALKING_TTS_LOCAL_COSYVOICE_SERVICE_URLS or omit tts_model to use the default local model."
    )


def _model_root() -> Path:
    raw = (
        os.environ.get("OPENTALKING_LOCAL_AUDIO_MODEL_ROOT", "").strip()
        or _settings_value("local_audio_model_root", "")
        or "./models/local-audio"
    )
    return Path(raw).expanduser()


def _resolve_model_path(model: str) -> str:
    path = Path(model).expanduser()
    if path.exists():
        return str(path)
    return str(_model_root() / model.replace("/", "__"))


def _resolve_default_local_voice_prompt() -> dict[str, str] | None:
    for base in (_model_root() / "voices" / "clones", _model_root() / "voices" / "system"):
        if not base.is_dir():
            continue
        for voice_dir in sorted(base.iterdir()):
            voice_prompt = _load_local_voice_prompt(voice_dir)
            if voice_prompt is not None:
                return voice_prompt
    return None


def _load_local_voice_prompt(voice_dir: Path) -> dict[str, str] | None:
    prompt_audio = voice_dir / "prompt.wav"
    prompt_text = voice_dir / "prompt.txt"
    if not prompt_audio.is_file() or not prompt_text.is_file():
        return None
    result = {"prompt_audio": str(prompt_audio)}
    text = prompt_text.read_text(encoding="utf-8").strip()
    if text:
        result["prompt_text"] = text
    meta_path = voice_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        for key in ("mode", "instruction"):
            value = str(meta.get(key) or "").strip()
            if value:
                result[key] = value
    return result if (result.get("prompt_text") or result.get("mode") or result.get("instruction")) else None


def _resolve_local_voice_prompt(voice: str | None) -> dict[str, str] | None:
    voice_id = (voice or "").strip()
    if not voice_id or voice_id == "local-default":
        return _resolve_default_local_voice_prompt()
    if not all(ch.isalnum() or ch in {"_", "-"} for ch in voice_id):
        return _resolve_default_local_voice_prompt()
    for base in (_model_root() / "voices" / "clones", _model_root() / "voices" / "system"):
        voice_prompt = _load_local_voice_prompt(base / voice_id)
        if voice_prompt is not None:
            return voice_prompt
    return _resolve_default_local_voice_prompt()


def _split_cosyvoice3_instruction(text: str) -> tuple[str, str]:
    raw = text.strip()
    if COSYVOICE3_END_OF_PROMPT in raw:
        instruction, _, reply = raw.partition(COSYVOICE3_END_OF_PROMPT)
        instruction = instruction.strip() or DEFAULT_COSYVOICE3_INSTRUCTION
        reply = reply.strip() or raw
        return reply, f"{instruction}{COSYVOICE3_END_OF_PROMPT}"
    return raw, ""


def _env_device() -> str:
    return (
        os.environ.get("OPENTALKING_TTS_LOCAL_COSYVOICE_DEVICE", "").strip()
        or _settings_value("tts_local_cosyvoice_device", "")
        or os.environ.get("OPENTALKING_LOCAL_TTS_DEVICE", "").strip()
        or os.environ.get("OPENTALKING_LOCAL_AUDIO_DEVICE", "").strip()
        or _settings_value("local_audio_device", "auto")
        or "auto"
    )


def _local_cosyvoice_bool(field: str, settings_name: str, default: bool) -> bool:
    raw = os.environ.get(f"OPENTALKING_TTS_LOCAL_COSYVOICE_{field}", "").strip()
    if not raw:
        raw = _settings_value(settings_name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _local_cosyvoice_int(field: str, settings_name: str, default: int) -> int:
    raw = os.environ.get(f"OPENTALKING_TTS_LOCAL_COSYVOICE_{field}", "").strip()
    if not raw:
        raw = _settings_value(settings_name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _local_cosyvoice_fp16(device: str) -> bool:
    raw = (
        os.environ.get("OPENTALKING_TTS_LOCAL_COSYVOICE_FP16", "").strip()
        or _settings_value("tts_local_cosyvoice_fp16", "")
        or "auto"
    ).lower()
    if raw == "auto":
        return device.startswith("cuda")
    return raw in {"1", "true", "yes", "on"}


def _instantiate_cosyvoice_runtime(cls: Any, model_dir: str, kwargs: dict[str, Any]) -> Any:
    runtime_kwargs = dict(kwargs)
    optional_keys = ("load_vllm", "trt_concurrent", "load_jit", "load_trt", "fp16")
    while True:
        try:
            return cls(model_dir, **runtime_kwargs)
        except TypeError as exc:
            text = str(exc)
            unsupported = next((key for key in optional_keys if key in runtime_kwargs and key in text), None)
            if unsupported is None:
                raise
            runtime_kwargs.pop(unsupported)


def _resolve_cosyvoice_runtime_class(cosyvoice_module: Any, preferred_name: str) -> Any:
    cls = getattr(cosyvoice_module, preferred_name, None)
    if cls is not None:
        return cls
    cls = getattr(cosyvoice_module, "AutoModel", None)
    if cls is not None:
        return cls
    raise RuntimeError(f"CosyVoice runtime module exposes neither {preferred_name} nor AutoModel.")


def _audio_format_from_content_type(content_type: str | None) -> str | None:
    value = (content_type or "").split(";", 1)[0].strip().lower()
    if value in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return "wav"
    if value in {"audio/l16", "audio/pcm", "application/octet-stream"}:
        return "pcm"
    if value in {"audio/mpeg", "audio/mp3"}:
        return "mp3"
    return None


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if pcm.size == 0 or src_sr == dst_sr:
        return pcm.copy()
    pcm_f = pcm.astype(np.float32) / 32768.0
    n_dst = max(1, int(round(pcm.size * dst_sr / src_sr)))
    xi = np.linspace(0.0, pcm.size - 1.0, num=n_dst)
    out = np.interp(xi, np.arange(pcm.size), pcm_f)
    return np.clip(np.round(out * 32768.0), -32768, 32767).astype(np.int16)


def _float_audio_to_i16(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int16)
    if np.max(np.abs(arr)) > 1.5:
        return np.clip(arr, -32768, 32767).astype(np.int16)
    return np.clip(np.round(arr * 32768.0), -32768, 32767).astype(np.int16)


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


class LocalCosyVoiceTTSAdapter:
    """Local CosyVoice adapter.

    Supports either a same-host local service URL or an in-process CosyVoice
    Python runtime when that package is installed.
    """

    def __init__(
        self,
        default_voice: str | None = None,
        sample_rate: int = 16000,
        chunk_ms: float = 20.0,
        *,
        model: str | None = None,
        service_url: str | None = None,
    ) -> None:
        self.default_voice = default_voice or "local-default"
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.default_model = (
            os.environ.get("OPENTALKING_TTS_LOCAL_COSYVOICE_MODEL")
            or _settings_value("tts_local_cosyvoice_model", "")
            or "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        ).strip()
        self.model = (model or self.default_model).strip()
        self.model_dir = (
            os.environ.get("OPENTALKING_TTS_LOCAL_COSYVOICE_MODEL_DIR", "").strip()
            or _settings_value("tts_local_cosyvoice_model_dir", "")
            or _resolve_model_path(self.model)
        )
        self.runtime_dir = (
            os.environ.get("OPENTALKING_TTS_LOCAL_COSYVOICE_RUNTIME_DIR", "").strip()
            or _settings_value("tts_local_cosyvoice_runtime_dir", "")
        )
        explicit_service_url = (service_url or "").strip()
        default_service_url = explicit_service_url or (
            os.environ.get("OPENTALKING_TTS_LOCAL_COSYVOICE_SERVICE_URL", "").strip()
            or _settings_value("tts_local_cosyvoice_service_url", "")
        )
        self.service_url = (
            explicit_service_url
            or _resolve_service_url_for_model(
                self.model,
                self.default_model,
                default_service_url,
            )
        )
        self.device = _env_device()
        self.fp16 = _local_cosyvoice_fp16(self.device)
        self.load_jit = _local_cosyvoice_bool("LOAD_JIT", "tts_local_cosyvoice_load_jit", False)
        self.load_trt = _local_cosyvoice_bool("LOAD_TRT", "tts_local_cosyvoice_load_trt", False)
        self.load_vllm = _local_cosyvoice_bool("LOAD_VLLM", "tts_local_cosyvoice_load_vllm", False)
        self.trt_concurrent = max(
            1,
            _local_cosyvoice_int("TRT_CONCURRENT", "tts_local_cosyvoice_trt_concurrent", 1),
        )
        self._engine: Any | None = None

    async def synthesize_stream(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        if not text.strip():
            return
        if self.service_url:
            async for chunk in self._synthesize_via_service(text, voice=voice):
                yield chunk
            return
        chunks = await asyncio.to_thread(self._synthesize_in_process, text, voice or self.default_voice)
        for chunk in chunks:
            yield chunk

    async def _synthesize_via_service(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        raw_text = text.strip()
        instruction_text = ""
        if "cosyvoice3" in self.model.lower():
            raw_text, instruction_text = _split_cosyvoice3_instruction(raw_text)
        try:
            from opentalking.pipeline.speak.text_sanitize import sanitize_tts_text

            tts_text = sanitize_tts_text(raw_text)
        except Exception:
            tts_text = raw_text
        if not tts_text:
            return
        payload = {
            "text": tts_text,
            "voice": voice or self.default_voice,
            "model": self.model,
            "sample_rate": self.sample_rate,
        }
        local_prompt = _resolve_local_voice_prompt(voice or self.default_voice)
        if local_prompt is not None:
            payload.update(local_prompt)
        if instruction_text:
            payload["mode"] = "instruct"
            payload["instruction"] = instruction_text
        elif "cosyvoice3" in self.model.lower():
            prompt_text = str(payload.get("prompt_text") or "").strip()
            if prompt_text and COSYVOICE3_END_OF_PROMPT not in prompt_text:
                payload["prompt_text"] = f"{DEFAULT_COSYVOICE3_INSTRUCTION}{COSYVOICE3_END_OF_PROMPT}{prompt_text}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", self.service_url, json=payload) as resp:
                resp.raise_for_status()
                input_format = _audio_format_from_content_type(resp.headers.get("content-type"))
                if input_format == "pcm":
                    source_sr = _source_sample_rate_from_headers(resp.headers, self.sample_rate)
                    pending = b""
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
                    return
                if input_format == "wav":
                    raw = await resp.aread()
                    with wave.open(io.BytesIO(raw), "rb") as wf:
                        source_sr = int(wf.getframerate())
                        channels = int(wf.getnchannels())
                        sample_width = int(wf.getsampwidth())
                        pcm_bytes = wf.readframes(wf.getnframes())
                    if sample_width != 2:
                        raise RuntimeError(f"Unsupported WAV sample width for local CosyVoice: {sample_width}")
                    pcm = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.int16, copy=False)
                    if channels > 1:
                        frame_count = pcm.size // channels
                        pcm = (
                            pcm[: frame_count * channels]
                            .reshape(frame_count, channels)
                            .mean(axis=1)
                            .astype(np.int16)
                        )
                    pcm = _resample_linear(pcm, source_sr, self.sample_rate)
                    for chunk in _split_pcm_chunks(pcm, self.sample_rate, self.chunk_ms):
                        yield chunk
                    return

                from opentalking.providers.tts.edge.adapter import _stream_decode_audio_to_pcm_chunks

                async def _audio_iter() -> AsyncIterator[bytes]:
                    async for data in resp.aiter_bytes():
                        if data:
                            yield data

                async for chunk in _stream_decode_audio_to_pcm_chunks(
                    _audio_iter(),
                    self.sample_rate,
                    self.chunk_ms,
                    input_format=input_format,
                ):
                    yield chunk

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            cosyvoice_module = importlib.import_module("cosyvoice.cli.cosyvoice")
        except ImportError as exc:
            raise RuntimeError(
                "Local CosyVoice requires a local service URL or the CosyVoice Python package. "
                "Set OPENTALKING_TTS_LOCAL_COSYVOICE_SERVICE_URL, or install the local-audio runtime."
            ) from exc
        model_dir = self.model_dir or _resolve_model_path(self.model)
        kwargs: dict[str, Any] = {
            "load_jit": self.load_jit,
            "load_trt": self.load_trt,
            "load_vllm": self.load_vllm,
            "fp16": self.fp16,
            "trt_concurrent": self.trt_concurrent,
        }
        model_lower = self.model.lower()
        if "cosyvoice3" in model_lower:
            cls = _resolve_cosyvoice_runtime_class(cosyvoice_module, "CosyVoice3")
        elif "cosyvoice2" in model_lower:
            cls = _resolve_cosyvoice_runtime_class(cosyvoice_module, "CosyVoice2")
        else:
            cls = _resolve_cosyvoice_runtime_class(cosyvoice_module, "CosyVoice")
        self._engine = _instantiate_cosyvoice_runtime(cls, model_dir, kwargs)
        return self._engine

    def _available_voice(self, engine: Any, requested: str) -> str:
        if requested and requested != "local-default":
            return requested
        list_spks = getattr(engine, "list_available_spks", None)
        if callable(list_spks):
            speakers = list_spks()
            if isinstance(speakers, Iterable):
                for item in speakers:
                    if item:
                        return str(item)
        return requested or "中文女"

    def _synthesize_in_process(self, text: str, voice: str) -> list[AudioChunk]:
        engine = self._load_engine()
        sr = int(getattr(engine, "sample_rate", 22050) or 22050)
        pcm_parts: list[np.ndarray] = []
        local_prompt = _resolve_local_voice_prompt(voice)
        infer_instruct2 = getattr(engine, "inference_instruct2", None)
        if callable(infer_instruct2) and local_prompt is not None:
            tts_text, instruct_text = _split_cosyvoice3_instruction(text)
            iterator = infer_instruct2(
                tts_text,
                instruct_text,
                local_prompt["prompt_audio"],
                stream=False,
                text_frontend=True,
            )
        else:
            spk_id = self._available_voice(engine, voice)
            infer_sft = getattr(engine, "inference_sft", None)
            if not callable(infer_sft):
                raise RuntimeError("CosyVoice runtime does not expose an available synthesis method.")
            iterator = infer_sft(text, spk_id, stream=False)
        for item in iterator:
            speech = item.get("tts_speech") if isinstance(item, dict) else item
            if hasattr(speech, "detach"):
                speech = speech.detach().cpu().numpy()
            pcm_parts.append(_float_audio_to_i16(speech))
        if not pcm_parts:
            return []
        pcm = np.concatenate(pcm_parts).astype(np.int16, copy=False)
        pcm = _resample_linear(pcm, sr, self.sample_rate)
        return _split_pcm_chunks(pcm, self.sample_rate, self.chunk_ms)
