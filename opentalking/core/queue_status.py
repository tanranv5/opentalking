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
    active_session_ids: list[str] | None = None,
    active_count: int | None = None,
    slot_capacity: int = 1,
    queued_session_ids: list[str] | None = None,
) -> None:
    active_ids = active_session_ids or ([active_session_id.strip()] if active_session_id.strip() else [])
    active_count_value = len(active_ids) if active_count is None else max(0, int(active_count))
    capacity = max(1, int(slot_capacity))
    await redis.hset(
        FLASHTALK_QUEUE_STATUS,
        mapping={
            "slot_occupied": "1" if slot_occupied else "0",
            "queue_size": str(max(0, int(queue_size))),
            "active_session_id": (active_session_id or (active_ids[0] if active_ids else "")).strip(),
            "active_session_ids": json.dumps(active_ids, ensure_ascii=False),
            "active_count": str(active_count_value),
            "slot_capacity": str(capacity),
            "slots_available": str(max(0, capacity - active_count_value)),
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
            "active_session_ids": [],
            "active_count": 0,
            "slot_capacity": 1,
            "slots_available": 1,
            "queued_session_ids": [],
        }
    active_session_id = _as_text(_hash_get(data, "active_session_id", "")).strip()
    active_session_ids = _as_str_list(_hash_get(data, "active_session_ids", "[]"))
    if not active_session_ids and active_session_id:
        active_session_ids = [active_session_id]
    legacy_slot_occupied = _as_bool(_hash_get(data, "slot_occupied", "0"))
    active_count = _as_int(_hash_get(data, "active_count", str(len(active_session_ids))))
    if active_count == 0 and active_session_ids:
        active_count = len(active_session_ids)
    if active_count == 0 and legacy_slot_occupied:
        active_count = 1
    slot_capacity = max(1, _as_int(_hash_get(data, "slot_capacity", "1")) or 1)
    return {
        "slot_occupied": legacy_slot_occupied,
        "queue_size": _as_int(_hash_get(data, "queue_size", "0")),
        "active_session_id": active_session_id or (active_session_ids[0] if active_session_ids else ""),
        "active_session_ids": active_session_ids,
        "active_count": active_count,
        "slot_capacity": slot_capacity,
        "slots_available": max(0, slot_capacity - active_count),
        "queued_session_ids": _as_str_list(_hash_get(data, "queued_session_ids", "[]")),
    }
