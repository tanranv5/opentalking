from __future__ import annotations

import json
from typing import Any

from opentalking.core.redis_keys import FLASHTALK_QUEUE_STATUS


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _as_bool(value: Any) -> bool:
    return _as_text(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any) -> int:
    try:
        return max(0, int(_as_text(value).strip()))
    except (TypeError, ValueError):
        return 0


def _as_str_list(value: Any) -> list[str]:
    text = _as_text(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _hash_get(data: dict[Any, Any], key: str, default: str) -> Any:
    return data.get(key, data.get(key.encode("utf-8"), default))


async def set_flashtalk_queue_status(
    redis: Any,
    *,
    slot_occupied: bool,
    queue_size: int,
    active_session_id: str = "",
    queued_session_ids: list[str] | None = None,
) -> None:
    await redis.hset(
        FLASHTALK_QUEUE_STATUS,
        mapping={
            "slot_occupied": "1" if slot_occupied else "0",
            "queue_size": str(max(0, int(queue_size))),
            "active_session_id": active_session_id.strip(),
            "queued_session_ids": json.dumps(queued_session_ids or [], ensure_ascii=False),
        },
    )


async def get_flashtalk_queue_status(redis: Any) -> dict[str, bool | int | str | list[str]]:
    data = await redis.hgetall(FLASHTALK_QUEUE_STATUS)
    if not data:
        return {
            "slot_occupied": False,
            "queue_size": 0,
            "active_session_id": "",
            "queued_session_ids": [],
        }
    return {
        "slot_occupied": _as_bool(_hash_get(data, "slot_occupied", "0")),
        "queue_size": _as_int(_hash_get(data, "queue_size", "0")),
        "active_session_id": _as_text(_hash_get(data, "active_session_id", "")).strip(),
        "queued_session_ids": _as_str_list(_hash_get(data, "queued_session_ids", "[]")),
    }
