from __future__ import annotations

import pytest

from opentalking.core.in_memory_redis import InMemoryRedis
from opentalking.core.queue_status import get_flashtalk_queue_status, set_flashtalk_queue_status


@pytest.mark.asyncio
async def test_empty_flashtalk_queue_status_returns_complete_shape() -> None:
    redis = InMemoryRedis()

    status = await get_flashtalk_queue_status(redis)

    assert status == {
        "slot_occupied": False,
        "queue_size": 0,
        "active_session_id": "",
        "active_session_ids": [],
        "active_count": 0,
        "slot_capacity": 1,
        "slots_available": 1,
        "queued_session_ids": [],
    }


@pytest.mark.asyncio
async def test_flashtalk_queue_status_round_trips_session_ids() -> None:
    redis = InMemoryRedis()

    await set_flashtalk_queue_status(
        redis,
        slot_occupied=True,
        queue_size=2,
        active_session_id="sess_active",
        active_session_ids=["sess_active", "sess_second"],
        active_count=2,
        slot_capacity=3,
        queued_session_ids=["sess_wait_1", "sess_wait_2"],
    )

    status = await get_flashtalk_queue_status(redis)

    assert status == {
        "slot_occupied": True,
        "queue_size": 2,
        "active_session_id": "sess_active",
        "active_session_ids": ["sess_active", "sess_second"],
        "active_count": 2,
        "slot_capacity": 3,
        "slots_available": 1,
        "queued_session_ids": ["sess_wait_1", "sess_wait_2"],
    }
