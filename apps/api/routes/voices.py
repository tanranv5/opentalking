"""音色目录（SQLite）与百炼声音复刻。"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import shutil
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from opentalking.providers.tts.dashscope_qwen import clone as bailian_clone
from opentalking.voice.store import delete_entry, get_entry, init_voice_store, insert_clone, list_voices

log = logging.getLogger(__name__)

router = APIRouter(tags=["voices"])

_UPLOAD_DIR = Path("data/voice_uploads")
LOCAL_COSYVOICE_SAMPLE_TEXT = "你好，今天阳光很好，我正在用自然清晰的声音，记录这一段音色。"
LOCAL_COSYVOICE_MIN_ACTIVE_SEC = 2.0
LOCAL_COSYVOICE_MIN_RMS_DBFS = -45.0
LOCAL_COSYVOICE_MIN_RECOGNIZED_CHARS = 4
LOCAL_COSYVOICE_MIN_TEXT_OVERLAP = 0.45
INDEXTTS_PROVIDER = "indextts"
INDEXTTS_LEGACY_PROVIDERS = {"local_indextts", "omnirt_indextts"}
INDEXTTS_PROVIDERS = {INDEXTTS_PROVIDER, *INDEXTTS_LEGACY_PROVIDERS}


class VoiceItem(TypedDict):
    id: int
    user_id: int
    provider: str
    voice_id: str
    display_label: str
    target_model: str | None
    source: str


def _voice_profile(provider: str) -> str | None:
    if provider.strip().lower() == "xiaomi_mimo":
        return "xiaomi_mimo"
    return None


def _upload_dir() -> Path:
    d = _UPLOAD_DIR.resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _local_audio_model_root() -> Path:
    raw = os.environ.get("OPENTALKING_LOCAL_AUDIO_MODEL_ROOT", "").strip()
    try:
        from opentalking.core.config import get_settings

        raw = raw or (get_settings().local_audio_model_root or "").strip()
    except Exception:
        pass
    if not raw:
        raw = "./models/local-audio"
    return Path(raw).expanduser().resolve()


def _safe_local_voice_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", label.strip().lower())
    slug = slug.strip("-")[:32] or "voice"
    return f"local-{slug}-{uuid.uuid4().hex[:8]}"


def _clean_local_cosyvoice_prompt_text(prompt_text: str) -> str:
    text = (prompt_text or "").strip()
    if "<|endofprompt|>" in text:
        text = text.rsplit("<|endofprompt|>", 1)[-1].strip()
    return text or LOCAL_COSYVOICE_SAMPLE_TEXT


def _write_local_cosyvoice_prompt(
    *,
    voice_id: str,
    wav: bytes,
    prompt_text: str,
    display_label: str,
    target_model: str,
    validation: dict[str, Any] | None = None,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", voice_id):
        raise ValueError("invalid local voice id")
    voice_dir = _local_audio_model_root() / "voices" / "clones" / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    clean_prompt_text = _clean_local_cosyvoice_prompt_text(prompt_text)
    (voice_dir / "prompt.wav").write_bytes(wav)
    (voice_dir / "prompt.txt").write_text(clean_prompt_text, encoding="utf-8")
    (voice_dir / "meta.json").write_text(
        json.dumps(
            {
                "voice_id": voice_id,
                "display_label": display_label,
                "provider": "local_cosyvoice",
                "target_model": target_model,
                "prompt_audio": str(voice_dir / "prompt.wav"),
                "prompt_text": clean_prompt_text,
                "validation": validation or {},
                "source": "clone",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return voice_dir


def _write_local_indextts_prompt(
    *,
    provider: str,
    voice_id: str,
    wav: bytes,
    prompt_text: str,
    display_label: str,
    target_model: str,
    validation: dict[str, Any] | None = None,
) -> Path:
    if provider not in INDEXTTS_PROVIDERS:
        raise ValueError("invalid IndexTTS provider")
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", voice_id):
        raise ValueError("invalid local voice id")
    voice_dir = _local_audio_model_root() / "voices" / "clones" / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    clean_prompt_text = prompt_text.strip()
    (voice_dir / "prompt.wav").write_bytes(wav)
    if clean_prompt_text:
        (voice_dir / "prompt.txt").write_text(clean_prompt_text, encoding="utf-8")
    (voice_dir / "meta.json").write_text(
        json.dumps(
            {
                "voice_id": voice_id,
                "display_label": display_label,
                "provider": provider,
                "target_model": target_model,
                "prompt_audio": str(voice_dir / "prompt.wav"),
                "prompt_text": clean_prompt_text,
                "validation": validation or {},
                "source": "clone",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return voice_dir


def _wav_audio_stats(wav: bytes) -> dict[str, float]:
    with wave.open(io.BytesIO(wav), "rb") as wf:
        sample_rate = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
        frames = int(wf.getnframes())
        raw = wf.readframes(frames)
    if sample_width != 2:
        raise HTTPException(status_code=400, detail=f"本地 CosyVoice 只支持 16-bit WAV，当前 sample_width={sample_width}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1 and pcm.size:
        pcm = pcm[: (pcm.size // channels) * channels].reshape(-1, channels).mean(axis=1)
    duration_sec = frames / sample_rate if sample_rate else 0.0
    if pcm.size == 0:
        rms_dbfs = -120.0
        active_sec = 0.0
    else:
        rms = float(np.sqrt(np.mean(np.square(pcm / 32768.0))))
        rms_dbfs = 20.0 * np.log10(max(rms, 1e-8))
        active_mask = np.abs(pcm) > 500.0
        active_sec = float(active_mask.sum() / sample_rate) if sample_rate else 0.0
    return {
        "sample_rate": float(sample_rate),
        "duration_sec": float(duration_sec),
        "active_sec": float(active_sec),
        "rms_dbfs": float(rms_dbfs),
    }


def _validate_local_indextts_prompt(wav: bytes) -> dict[str, Any]:
    stats = _wav_audio_stats(wav)
    if stats["duration_sec"] < 1.0:
        raise HTTPException(status_code=400, detail="IndexTTS 参考音频过短，请录制 3-15 秒清晰人声。")
    if stats["active_sec"] < 0.5 or stats["rms_dbfs"] < LOCAL_COSYVOICE_MIN_RMS_DBFS:
        raise HTTPException(status_code=400, detail="IndexTTS 参考音频声音太小或静音太多，请靠近麦克风重录。")
    return stats


def _text_overlap_ratio(left: str, right: str) -> float:
    left_chars = [ch for ch in left if not ch.isspace() and ch not in "，。！？、,.!?;；:："]
    right_set = {ch for ch in right if not ch.isspace() and ch not in "，。！？、,.!?;；:："}
    if not left_chars or not right_set:
        return 0.0
    return sum(1 for ch in left_chars if ch in right_set) / len(left_chars)


async def _validate_local_cosyvoice_prompt(wav: bytes, prompt_text: str) -> dict[str, Any]:
    text = prompt_text.strip() or LOCAL_COSYVOICE_SAMPLE_TEXT
    stats = _wav_audio_stats(wav)
    if stats["duration_sec"] < 3.0:
        raise HTTPException(status_code=400, detail="本地 CosyVoice 参考音频过短，请录制 5-15 秒清晰人声。")
    if stats["active_sec"] < LOCAL_COSYVOICE_MIN_ACTIVE_SEC or stats["rms_dbfs"] < LOCAL_COSYVOICE_MIN_RMS_DBFS:
        raise HTTPException(status_code=400, detail="本地 CosyVoice 参考音频声音太小或静音太多，请靠近麦克风重录。")

    try:
        from opentalking.providers.stt.factory import transcribe_wav_path_sync
    except Exception:
        return {**stats, "recognized_text": ""}

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav)
        wav_path = Path(tmp.name)
    try:
        recognized, stt_ms = await asyncio.to_thread(transcribe_wav_path_sync, wav_path)
    except Exception as exc:
        log.warning("local cosyvoice prompt STT validation failed: %s", exc)
        return {**stats, "recognized_text": "", "stt_error": str(exc)}
    finally:
        wav_path.unlink(missing_ok=True)

    recognized = recognized.strip()
    if len(recognized) < LOCAL_COSYVOICE_MIN_RECOGNIZED_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"本地 CosyVoice 参考音频识别内容过短（识别为“{recognized or '空'}”），请完整朗读提示文本后重录。",
        )
    overlap = _text_overlap_ratio(recognized, text)
    if overlap < LOCAL_COSYVOICE_MIN_TEXT_OVERLAP:
        raise HTTPException(
            status_code=400,
            detail=f"本地 CosyVoice 参考音频与参考文本不一致（识别为“{recognized}”），请让文本和录音内容一致后重录。",
        )
    return {**stats, "recognized_text": recognized, "stt_ms": float(stt_ms), "expected_text": text, "text_overlap": overlap}


def _remove_local_prompt(voice_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", voice_id):
        return
    voice_dir = _local_audio_model_root() / "voices" / "clones" / voice_id
    root = (_local_audio_model_root() / "voices" / "clones").resolve()
    target = voice_dir.resolve()
    if root not in target.parents:
        return
    shutil.rmtree(target, ignore_errors=True)


def _remove_local_cosyvoice_prompt(voice_id: str) -> None:
    _remove_local_prompt(voice_id)


def _bundled_system_voice_root() -> Path:
    return Path(__file__).resolve().parents[3] / "opentalking" / "assets" / "voices" / "system"


def _system_voice_roots() -> list[Path]:
    roots = [_local_audio_model_root() / "voices" / "system", _bundled_system_voice_root()]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(root)
    return out


def _local_cosyvoice_voice_items(source: str) -> list[VoiceItem]:
    if source not in {"system", "clones"}:
        return []
    root = _local_audio_model_root() / "voices" / source
    if not root.is_dir():
        return []
    items: list[VoiceItem] = []
    for idx, voice_dir in enumerate(sorted(p for p in root.iterdir() if p.is_dir()), start=1):
        voice_id = voice_dir.name
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", voice_id):
            continue
        if not (voice_dir / "prompt.wav").is_file() or not (voice_dir / "prompt.txt").is_file():
            continue
        label = voice_id
        target_model: str | None = None
        meta_path = voice_dir / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                label = str(meta.get("display_label") or meta.get("label") or label)
                tm = str(meta.get("target_model") or "").strip()
                target_model = tm or None
            except Exception:
                pass
        items.append(
            {
                "id": -idx,
                "user_id": 1,
                "provider": "local_cosyvoice",
                "voice_id": voice_id,
                "display_label": label,
                "target_model": target_model,
                "source": "clone" if source == "clones" else "system",
            }
        )
    return items


def _local_cosyvoice_system_voice_items() -> list[VoiceItem]:
    return _local_cosyvoice_voice_items("system")


def _local_cosyvoice_clone_voice_items() -> list[VoiceItem]:
    return _local_cosyvoice_voice_items("clones")


def _local_indextts_voice_items(provider: str, source: str) -> list[VoiceItem]:
    if provider not in INDEXTTS_PROVIDERS or source not in {"system", "clones"}:
        return []
    roots = [_local_audio_model_root() / "voices" / source]
    if source == "system":
        roots = _system_voice_roots()
    items: list[VoiceItem] = []
    idx = 0
    for root in roots:
        if not root.is_dir():
            continue
        for voice_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            idx += 1
            voice_id = voice_dir.name
            if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", voice_id):
                continue
            if not (voice_dir / "prompt.wav").is_file():
                continue
            label = voice_id
            target_model: str | None = None
            meta_path = voice_dir / "meta.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    raw_providers = meta.get("providers")
                    requested = {provider, *INDEXTTS_LEGACY_PROVIDERS} if provider == INDEXTTS_PROVIDER else {provider}
                    if isinstance(raw_providers, list):
                        allowed = {str(item).strip().lower() for item in raw_providers}
                        if allowed and not (allowed & requested):
                            continue
                    elif str(meta.get("provider") or "").strip().lower() not in {"", *INDEXTTS_PROVIDERS}:
                        continue
                    label = str(meta.get("display_label") or meta.get("label") or label)
                    tm = str(meta.get("target_model") or "").strip()
                    target_model = tm or None
                except Exception:
                    pass
            items.append(
                {
                    "id": -idx,
                    "user_id": 1,
                    "provider": provider,
                    "voice_id": voice_id,
                    "display_label": label,
                    "target_model": target_model,
                    "source": "clone" if source == "clones" else "system",
                }
            )
    return items


def _public_base(request: Request) -> str:
    try:
        from opentalking.core.config import get_settings

        raw = (get_settings().public_base_url or "").strip()
    except Exception:
        raw = ""
    if raw:
        return raw.rstrip("/")
    return str(request.base_url).rstrip("/")


def _dedupe_display_label(label: str, *, provider: str, target_model: str | None) -> str:
    normalized = label.strip() or "我的复刻音色"
    existing = {
        r.display_label
        for r in list_voices(provider=provider)
        if r.source == "clone" and (r.target_model or "") == (target_model or "")
    }
    if normalized not in existing:
        return normalized
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    return f"{normalized}-{stamp}"


@router.get("/voices", response_model=None)
async def get_voices(provider: str | None = None) -> JSONResponse:
    init_voice_store()
    p = provider.strip().lower() if provider else None
    public_p = INDEXTTS_PROVIDER if p in INDEXTTS_PROVIDERS else p
    row_provider = None if public_p is None else public_p
    rows = list_voices(provider=row_provider)
    if public_p == INDEXTTS_PROVIDER:
        rows = [*rows, *list_voices(provider="local_indextts"), *list_voices(provider="omnirt_indextts")]
    elif public_p is None:
        rows = [*rows, *list_voices(provider="local_indextts"), *list_voices(provider="omnirt_indextts")]
    items: list[VoiceItem] = []
    existing = set()
    for r in rows:
        item_provider = INDEXTTS_PROVIDER if r.provider in INDEXTTS_PROVIDERS else r.provider
        key = (item_provider, r.voice_id)
        if key in existing:
            continue
        existing.add(key)
        item: VoiceItem = {
            "id": r.id,
            "user_id": r.user_id,
            "provider": item_provider,
            "voice_id": r.voice_id,
            "display_label": r.display_label,
            "target_model": r.target_model,
            "source": r.source,
        }
        profile = _voice_profile(r.provider)
        if profile:
            item["profile"] = profile  # type: ignore[typeddict-unknown-key]
        items.append(item)
    if public_p in {None, "local_cosyvoice"}:
        for item in [*_local_cosyvoice_system_voice_items(), *_local_cosyvoice_clone_voice_items()]:
            key = (item["provider"], item["voice_id"])
            if key not in existing:
                items.append(item)
                existing.add(key)
    if public_p is None or public_p == INDEXTTS_PROVIDER:
        for source in ("system", "clones"):
            for item in _local_indextts_voice_items(INDEXTTS_PROVIDER, source):
                key = (item["provider"], item["voice_id"])
                if key not in existing:
                    items.append(item)
                    existing.add(key)
    return JSONResponse({"items": items})


@router.get("/voice-uploads/{token}")
async def serve_voice_upload(token: str) -> FileResponse:
    """供百炼服务端拉取 CosyVoice 复刻用音频（须公网可达）。"""
    safe = re.sub(r"[^a-f0-9]", "", token.lower())
    if len(safe) != 32:
        raise HTTPException(status_code=404, detail="invalid token")
    path = _upload_dir() / f"{safe}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="sample not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{safe}.wav")


@router.post("/voices/clone", response_model=None)
async def post_voice_clone(
    request: Request,
    background_tasks: BackgroundTasks,
    provider: str = Form(..., description="local_cosyvoice、indextts、cosyvoice、dashscope 或 xiaomi_mimo"),
    target_model: str = Form(...),
    display_label: str = Form(...),
    audio: UploadFile = File(...),
    prefix: str = Form(""),
    preferred_name: str = Form(""),
    prompt_text: str = Form(LOCAL_COSYVOICE_SAMPLE_TEXT),
) -> JSONResponse:
    """
    上传一段朗读音频完成复刻。
    - 本地 CosyVoice：保存为本地 prompt 音频和文本，不调用云端。
    - IndexTTS：保存为本地 prompt 音频，可供 IndexTTS local / OmniRT 后端共同使用。
    - 云端 CosyVoice：需要本服务地址对公网可达（或配置 OPENTALKING_PUBLIC_BASE_URL）。
    - 千问（dashscope）：使用 base64，无需公网 URL。
    - 小米 MiMo：保存参考音频 data URI，合成时传给 mimo-v2.5-tts-voiceclone。
    """
    init_voice_store()
    prov = provider.strip().lower()
    if prov in {"xiaomi", "mimo"}:
        prov = "xiaomi_mimo"
    if prov not in {"local_cosyvoice", "cosyvoice", "dashscope", "xiaomi_mimo", *INDEXTTS_PROVIDERS}:
        raise HTTPException(
            status_code=400,
            detail="provider 须为 local_cosyvoice、indextts、cosyvoice、dashscope 或 xiaomi_mimo",
        )

    raw = await audio.read()
    if len(raw) < 256:
        raise HTTPException(status_code=400, detail="音频过短")
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="音频文件过大（>12MB）")

    suffix = Path(audio.filename or "upload.bin").suffix.lower() or ".webm"
    try:
        wav = bailian_clone.convert_audio_to_wav_24k_mono(raw, suffix)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    tm = (target_model or "").strip()
    label = _dedupe_display_label(
        (display_label or "").strip() or "我的复刻音色",
        provider=INDEXTTS_PROVIDER if prov in INDEXTTS_PROVIDERS else prov,
        target_model=tm,
    )

    try:
        if prov in INDEXTTS_PROVIDERS:
            voice_id = _safe_local_voice_id(label)
            effective_model = tm or "IndexTeam/IndexTTS-2"
            validation = _validate_local_indextts_prompt(wav)
            _write_local_indextts_prompt(
                provider=INDEXTTS_PROVIDER,
                voice_id=voice_id,
                wav=wav,
                prompt_text=(prompt_text or "").strip(),
                display_label=label,
                target_model=effective_model,
                validation=validation,
            )
            eid = insert_clone(
                provider=INDEXTTS_PROVIDER,
                voice_id=voice_id,
                display_label=label,
                target_model=effective_model,
            )
            return JSONResponse(
                {
                    "ok": True,
                    "entry_id": eid,
                    "voice_id": voice_id,
                    "display_label": label,
                    "provider": INDEXTTS_PROVIDER,
                    "target_model": effective_model,
                    "message": "IndexTTS 复刻音色已保存，可用于 IndexTTS 合成。",
                }
            )

        if prov == "local_cosyvoice":
            voice_id = _safe_local_voice_id(label)
            clean_prompt_text = _clean_local_cosyvoice_prompt_text(prompt_text)
            validation = await _validate_local_cosyvoice_prompt(wav, clean_prompt_text)
            _write_local_cosyvoice_prompt(
                voice_id=voice_id,
                wav=wav,
                prompt_text=clean_prompt_text,
                display_label=label,
                target_model=tm,
                validation=validation,
            )
            eid = insert_clone(
                provider="local_cosyvoice",
                voice_id=voice_id,
                display_label=label,
                target_model=tm,
            )
            return JSONResponse(
                {
                    "ok": True,
                    "entry_id": eid,
                    "voice_id": voice_id,
                    "display_label": label,
                    "provider": "local_cosyvoice",
                    "target_model": tm,
                    "message": "本地 CosyVoice 音色已保存，可直接用于本地合成。",
                }
            )

        if prov == "cosyvoice":
            token = uuid.uuid4().hex
            up = _upload_dir() / f"{token}.wav"
            up.write_bytes(wav)
            base = _public_base(request)
            public_url = f"{base}/voice-uploads/{token}"
            pf = (prefix or "").strip().lower() or f"c{token[:6]}"
            try:
                vid = bailian_clone.clone_cosyvoice_voice(
                    target_model=tm,
                    prefix=pf,
                    public_audio_url=public_url,
                )
            except Exception as e:
                up.unlink(missing_ok=True)
                raise HTTPException(status_code=502, detail=f"CosyVoice 复刻失败: {e}") from e

            eid = insert_clone(
                provider="cosyvoice",
                voice_id=vid,
                display_label=label,
                target_model=tm,
            )

            async def _remove_upload_after_delay(p: Path, seconds: float = 300.0) -> None:
                await asyncio.sleep(seconds)
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

            background_tasks.add_task(_remove_upload_after_delay, up)
            return JSONResponse(
                {
                    "ok": True,
                    "entry_id": eid,
                    "voice_id": vid,
                    "display_label": label,
                    "provider": "cosyvoice",
                    "target_model": tm,
                    "message": "CosyVoice 复刻成功，请在合成时选择相同 CosyVoice 模型。",
                }
            )

        if prov == "xiaomi_mimo":
            voice_id = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
            effective_model = tm or "mimo-v2.5-tts-voiceclone"
            eid = insert_clone(
                provider="xiaomi_mimo",
                voice_id=voice_id,
                display_label=label,
                target_model=effective_model,
            )
            return JSONResponse(
                {
                    "ok": True,
                    "entry_id": eid,
                    "voice_id": voice_id,
                    "display_label": label,
                    "provider": "xiaomi_mimo",
                    "profile": "xiaomi_mimo",
                    "target_model": effective_model,
                    "message": "小米 MiMo 复刻音色已保存，请使用 mimo-v2.5-tts-voiceclone 合成。",
                }
            )

        # 千问声音复刻（与实时 qwen3-tts-flash-realtime 不同系，请选文档中的 VC / 复刻专用模型）
        pref = (preferred_name or "").strip().lower() or f"u{uuid.uuid4().hex[:8]}"
        try:
            vid = bailian_clone.clone_qwen_voice(
                wav_bytes=wav,
                target_model=tm,
                preferred_name=pref,
                audio_mime="audio/wav",
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        eid = insert_clone(
            provider="dashscope",
            voice_id=vid,
            display_label=label,
            target_model=tm,
        )
        return JSONResponse(
            {
                "ok": True,
                "entry_id": eid,
                "voice_id": vid,
                "display_label": label,
                "provider": "dashscope",
                "target_model": tm,
                "message": "千问复刻成功：合成时请选用与复刻相同的 target_model。",
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.delete("/voices/{entry_id}", response_model=None)
async def delete_voice_entry(entry_id: int) -> JSONResponse:
    init_voice_store()
    row = get_entry(entry_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    if row.get("source") != "clone":
        raise HTTPException(status_code=400, detail="不能删除系统预设音色")
    if delete_entry(entry_id):
        if row.get("provider") in {"local_cosyvoice", *INDEXTTS_PROVIDERS}:
            _remove_local_prompt(str(row.get("voice_id") or ""))
        return JSONResponse({"ok": True})
    raise HTTPException(status_code=404, detail="not found")
