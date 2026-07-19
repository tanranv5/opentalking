"""单会话外部动作 clip 任务控制。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from opentalking.runtime.bus import publish_event

log = logging.getLogger(__name__)


class ExternalClipTaskMixin:
    """为 runner 提供可取消、可被口播抢占的外部 clip 调度。"""

    session_id: str
    redis: Any
    speech_tasks: set[asyncio.Task[None]]
    _active_clip_task: asyncio.Task[None] | None
    _closed: bool
    _speaking: bool

    def start_clip(self, clip_id: str) -> bool:
        """启动 clip 后台任务；口播已排队或执行时拒绝。"""
        active = getattr(self, "_active_clip_task", None)
        active_running = active is not None and not active.done()
        pending_speech = any(not task.done() for task in getattr(self, "speech_tasks", set()))
        if self._closed or pending_speech or (self._speaking and not active_running):
            return False
        if active_running:
            setattr(active, "_clip_cancel_reason", "superseded")
            active.cancel()
        task = asyncio.create_task(self._run_external_clip(clip_id))
        self._active_clip_task = task
        task.add_done_callback(self._clear_active_clip_task)
        return True

    async def cancel_active_clip(self, reason: str) -> None:
        """取消当前外部 clip，并等待其释放媒体锁。"""
        task = getattr(self, "_active_clip_task", None)
        if task is None or task.done() or task is asyncio.current_task():
            return
        setattr(task, "_clip_cancel_reason", reason)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if getattr(self, "_active_clip_task", None) is task:
            self._active_clip_task = None

    async def play_clip(self, clip_id: str) -> bool:
        """由具体 runner 实现媒体写入，返回是否完成播放窗口。"""
        raise NotImplementedError

    async def _run_external_clip(self, clip_id: str) -> None:
        played = False
        reason: str | None = None
        cancelled = False
        task = asyncio.current_task()
        try:
            played = await self.play_clip(clip_id)
        except asyncio.CancelledError:
            cancelled = True
            reason = str(getattr(task, "_clip_cancel_reason", "cancelled"))
        except Exception as exc:  # noqa: BLE001
            reason = "error"
            log.exception("external clip failed: session=%s clip_id=%s", self.session_id, clip_id)
            await publish_event(
                self.redis,
                self.session_id,
                "error",
                {
                    "session_id": self.session_id,
                    "code": "CLIP_PLAYBACK_FAILED",
                    "message": str(exc),
                    "clip_id": clip_id,
                },
            )
        payload: dict[str, object] = {
            "session_id": self.session_id,
            "clip_id": clip_id,
            "played": played,
        }
        if reason:
            payload["reason"] = reason
        await publish_event(self.redis, self.session_id, "clip.ended", payload)
        if cancelled:
            raise asyncio.CancelledError

    def _clear_active_clip_task(self, task: asyncio.Task[None]) -> None:
        if getattr(self, "_active_clip_task", None) is task:
            self._active_clip_task = None
