"""百炼 DashScope 语音识别（Paraformer 实时模型）。

- **文件路径**：``Recognition.call(wav)``（SDK 内部按块走 WebSocket）。
- **真流式**：``Recognition.start()`` → ``send_audio_frame(pcm)`` → ``stop()``，
  适用于浏览器经 WebSocket 下发 **PCM s16le mono 16kHz** 分块。
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

log = logging.getLogger(__name__)


class _NoopRecognitionCallback(RecognitionCallback):
    """Recognition 构造要求 callback；同步 call() 路径在内部汇总结果，此处仅占位。"""

    def on_event(self, result: RecognitionResult) -> None:
        pass


class _StreamingTextCollector(RecognitionCallback):
    """流式识别文本收集器。

    DashScope 流式回调中 ``result.get_sentence()`` 返回 **单个 dict**（当前句），
    仅同步 ``Recognition.call()`` 路径才返回 list。按 SDK 的
    ``RecognitionResult.is_sentence_end(sentence)``（``end_time`` 非空）判定句尾：
    句尾追加进 ``_segments`` 并清空 partial，否则仅覆盖当前句 partial。
    ``combined_text()`` = 已定稿分句 + 尾部未定稿 partial，保证长语音不丢前段/尾段。

    事件协议（经 ``event_queue`` 发往 WS 前端）：
    - ``transcript.partial``：中间结果，text 为累计全文预览
    - ``transcript.segment_final``：单句定稿，text 为累计全文（整段未结束，is_final=False）
    - 整段唯一终态 ``transcript.final`` 由 API 层在识别全部结束后发送，本类不发。
    """

    def __init__(self, event_queue: "queue.Queue[dict] | None" = None) -> None:
        self._segments: list[str] = []
        # 已消费句尾的 (begin_time, end_time)，防 SDK 重发同一句尾导致重复拼接。
        # 不能按文本去重：用户可能真的连说两遍相同的话。
        self._consumed_sentence_keys: set[tuple] = set()
        self._last_partial = ""
        self._lock = threading.Lock()
        self.error_message: str | None = None
        self._event_queue = event_queue
        self._last_emitted_partial = ""

    @staticmethod
    def _iter_sentences(result: RecognitionResult) -> list[dict]:
        """将流式（dict）与同步（list）两种返回统一为句 dict 列表。"""
        sentences = result.get_sentence()
        if isinstance(sentences, dict):
            return [sentences]
        if isinstance(sentences, list):
            return [s for s in sentences if isinstance(s, dict)]
        return []

    def on_event(self, result: RecognitionResult) -> None:
        with self._lock:
            handled = False
            for item in self._iter_sentences(result):
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                handled = True
                if RecognitionResult.is_sentence_end(item):
                    key = (item.get("begin_time"), item.get("end_time"))
                    if key in self._consumed_sentence_keys:
                        continue
                    self._consumed_sentence_keys.add(key)
                    self._segments.append(text)
                    self._last_partial = ""
                    self._emit("transcript.segment_final", self._merged_locked(), False)
                else:
                    self._last_partial = text
                    self._emit("transcript.partial", self._merged_locked(), False)
            if not handled:
                t = recognition_result_to_text(result)
                if t:
                    self._last_partial = t
                    self._emit("transcript.partial", self._merged_locked(), False)

    def on_error(self, result: RecognitionResult) -> None:
        with self._lock:
            self.error_message = getattr(result, "message", None) or str(result)
            self._emit("error", self.error_message, True)

    def _merged_locked(self) -> str:
        """调用方须已持有 ``_lock``：已定稿分句 + 尾部 partial。"""
        return ("".join(self._segments) + self._last_partial).strip()

    def combined_text(self) -> str:
        with self._lock:
            return self._merged_locked()

    def _emit(self, event_type: str, text: str | None, is_final: bool) -> None:
        if self._event_queue is None:
            return
        payload = (text or "").strip()
        if not payload:
            return
        # 仅对 partial 做同文本去重；segment_final 即使文本与上一条 partial 相同也要发出
        if event_type == "transcript.partial":
            if payload == self._last_emitted_partial:
                return
            self._last_emitted_partial = payload
        self._event_queue.put({
            "type": event_type,
            "text": payload,
            "is_final": is_final,
        })


def _dashscope_api_key() -> str:
    direct = os.environ.get("OPENTALKING_STT_DASHSCOPE_API_KEY", "").strip()
    if direct:
        return direct
    try:
        from opentalking.core.config import get_settings

        settings = get_settings()
        return getattr(settings, "stt_dashscope_api_key", "").strip()
    except Exception:
        return ""


def _ffmpeg_bin() -> str:
    return os.environ.get("OPENTALKING_FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"


def _stt_model() -> str:
    direct = os.environ.get("OPENTALKING_STT_DASHSCOPE_MODEL", "").strip()
    if direct:
        return direct
    try:
        from opentalking.core.config import get_settings

        settings = get_settings()
        return getattr(settings, "stt_dashscope_model", "").strip() or "paraformer-realtime-v2"
    except Exception:
        return "paraformer-realtime-v2"


def _language_hints() -> list[str] | None:
    raw = os.environ.get("OPENTALKING_STT_LANGUAGE_HINTS", "").strip()
    if not raw:
        return ["zh", "en"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


def recognition_result_to_text(result: RecognitionResult) -> str:
    """从 Recognition.call 返回值解析中文/英文拼接文本。"""
    sentences = result.get_sentence()
    if sentences is None:
        out = getattr(result, "output", None)
        if isinstance(out, dict):
            s = out.get("sentence")
            if isinstance(s, dict) and s.get("text"):
                return str(s["text"]).strip()
        return ""

    if isinstance(sentences, dict):
        return str(sentences.get("text", "")).strip()

    parts: list[str] = []
    for item in sentences:
        if isinstance(item, dict) and item.get("text"):
            parts.append(str(item["text"]))
    merged = "".join(parts).strip()
    if merged:
        return merged

    out = getattr(result, "output", None)
    if isinstance(out, dict):
        s = out.get("sentence")
        if isinstance(s, dict) and s.get("text"):
            return str(s["text"]).strip()
    return ""


async def ensure_wav_16k_mono(src_path: Path, wav_out: Path) -> None:
    """将浏览器上传的 webm/ogg/mp4 等转为 16kHz 单声道 WAV（Paraformer 常用格式）。"""
    ff = _ffmpeg_bin()
    log.debug("STT ffmpeg: %s -> %s (src size=%s)", src_path, wav_out, src_path.stat().st_size)
    proc = await asyncio.create_subprocess_exec(
        ff,
        "-y",
        "-i",
        str(src_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_out),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        msg = stderr.decode(errors="replace")[-800:]
        log.warning("STT ffmpeg failed rc=%s: %s", proc.returncode, msg)
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {msg}")


async def decode_audio_file_to_pcm_i16(src_path: Path) -> np.ndarray:
    """将任意 ffmpeg 可读音频解码为 16kHz mono PCM int16。"""
    ff = _ffmpeg_bin()
    proc = await asyncio.create_subprocess_exec(
        ff,
        "-y",
        "-i",
        str(src_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        msg = stderr.decode(errors="replace")[-800:]
        log.warning("audio decode ffmpeg failed rc=%s: %s", proc.returncode, msg)
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {msg}")
    return np.frombuffer(stdout, dtype=np.int16)


def _recognize_wav_sync(wav_path: Path) -> tuple[str, float]:
    """调用百炼 Recognition；返回 (文本, Recognition.call 墙钟毫秒)。"""
    import dashscope

    api_key = _dashscope_api_key()
    if not api_key:
        raise RuntimeError(
            "缺少 DashScope STT API Key：请设置 OPENTALKING_STT_DASHSCOPE_API_KEY。"
        )

    dashscope.api_key = api_key
    hints = _language_hints()
    kwargs: dict = {}
    if hints:
        kwargs["language_hints"] = hints

    model = _stt_model()
    rc = Recognition(
        model=model,
        callback=_NoopRecognitionCallback(),
        format="wav",
        sample_rate=16000,
        **kwargs,
    )
    t_call = time.perf_counter()
    try:
        result = rc.call(str(wav_path))
    except Exception as e:
        log.exception("STT Recognition.call raised")
        raise RuntimeError(f"DashScope STT 调用异常: {e}") from e

    call_ms = (time.perf_counter() - t_call) * 1000.0

    sc = getattr(result, "status_code", None)
    try:
        sc_int = int(sc) if sc is not None else None
    except (TypeError, ValueError):
        sc_int = None
    if sc_int != 200:
        msg = getattr(result, "message", None) or getattr(result, "code", None)
        req = getattr(result, "request_id", None)
        log.warning(
            "STT DashScope 非成功: status_code=%s message=%s code=%s request_id=%s",
            sc,
            getattr(result, "message", None),
            getattr(result, "code", None),
            req,
        )
        detail = msg or "STT failed"
        raise RuntimeError(
            f"百炼语音识别失败（HTTP {sc}）: {detail}。"
            "请核对 API Key、模型名 OPENTALKING_STT_DASHSCOPE_MODEL、账号是否开通语音识别。"
        )

    text = recognition_result_to_text(result)
    if not text.strip():
        log.warning(
            "STT 返回空文本: output=%s",
            getattr(result, "output", None),
        )
    return text, call_ms


def transcribe_pcm_chunk_queue_sync(
    chunk_queue: "queue.Queue[bytes | None]",
    *,
    event_queue: "queue.Queue[dict] | None" = None,
) -> tuple[str, float]:
    """PCM s16le mono 16kHz 分块流式识别。

    ``chunk_queue`` 中依次放入音频 ``bytes``；放入 ``None`` 表示本段音频已结束，
    随后将调用 ``Recognition.stop()``。
    """
    import dashscope

    api_key = _dashscope_api_key()
    if not api_key:
        raise RuntimeError(
            "缺少 DashScope STT API Key：请设置 OPENTALKING_STT_DASHSCOPE_API_KEY。"
        )

    dashscope.api_key = api_key
    hints = _language_hints()
    kwargs: dict = {}
    if hints:
        kwargs["language_hints"] = hints

    collector = _StreamingTextCollector(event_queue)
    model = _stt_model()
    rc = Recognition(
        model=model,
        callback=collector,
        format="pcm",
        sample_rate=16000,
        **kwargs,
    )
    t0 = time.perf_counter()
    try:
        rc.start()
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            if chunk:
                rc.send_audio_frame(chunk)
    finally:
        try:
            rc.stop()
        except Exception:  # noqa: BLE001
            log.exception("STT streaming Recognition.stop failed")

    dashscope_ms = (time.perf_counter() - t0) * 1000.0

    if collector.error_message:
        raise RuntimeError(f"DashScope 流式 STT 错误: {collector.error_message}")

    text = collector.combined_text()
    if not text.strip():
        log.warning("STT 流式识别返回空文本")

    return text, dashscope_ms


async def transcribe_audio_file_path(upload_path: Path) -> str:
    """上传的任意 ffmpeg 可读音频 → 临时 WAV → DashScope 识别。"""
    t_total0 = time.perf_counter()
    upload_sz = upload_path.stat().st_size

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        t_ff0 = time.perf_counter()
        await ensure_wav_16k_mono(upload_path, wav_path)
        ffmpeg_ms = (time.perf_counter() - t_ff0) * 1000.0

        wav_sz = wav_path.stat().st_size
        # PCM16 单声道 16kHz：约 32000 字节/秒；WAV 头约 44 字节（粗略估算时长）
        pcm_bytes = max(0, wav_sz - 44)
        est_audio_s = pcm_bytes / 32000.0

        text, dashscope_ms = await asyncio.to_thread(_recognize_wav_sync, wav_path)

        total_ms = (time.perf_counter() - t_total0) * 1000.0
        preview = (text.strip()[:24] + "…") if len(text.strip()) > 24 else text.strip()
        log.info(
            "STT timing: model=%s upload_bytes=%d ffmpeg_ms=%.0f wav_bytes=%d ~audio=%.2fs "
            "dashscope_ms=%.0f total_ms=%.0f text_chars=%d preview=%r",
            _stt_model(),
            upload_sz,
            ffmpeg_ms,
            wav_sz,
            est_audio_s,
            dashscope_ms,
            total_ms,
            len(text.strip()),
            preview,
        )
        return text
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass
