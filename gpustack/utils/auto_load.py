"""Shared cold-start / auto-load helpers (custom, fork-specific).

Both the gateway-auth path (routes/token.py) and the fallback-proxy path
(routes/openai.py) need to: resolve a route name to model ids, and long-poll
until a cold-started replica is ready (or fails / times out). The two call
sites differ only in their *return contract* (token.py returns an HTTP
Response or None to proceed; openai.py returns the loaded Model or raises),
so the duplicated target-resolution + poll loop + failure detection live here
and each caller keeps its own contract.

Keeping the poll loop in one place also means the cold-start concurrency gate
(cold_start_gate) protects both paths.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.model_routes import ModelRouteTarget
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

AUTO_LOAD_POLL_INTERVAL = 3
AUTO_LOAD_TIMEOUT = 300

# Poll outcomes.
READY = "ready"
FAILED = "failed"
TIMEOUT = "timeout"


@dataclass
class PollResult:
    outcome: str
    model: Optional[Model] = None  # set when READY
    message: Optional[str] = None  # set when FAILED


async def resolve_route_model_ids(session, model_name: str) -> List[int]:
    """Resolve a route/model name to its backing model ids."""
    targets = await ModelRouteTarget.all_by_fields(
        session,
        fields={"route_name": model_name, "deleted_at": None},
    )
    return [t.model_id for t in targets if t.model_id is not None]


def _detect_load_failure(instances, elapsed: int) -> Optional[str]:
    """Return a failure message if any instance has errored or, after a grace
    period, is stuck with "No suitable workers". Otherwise None."""
    for i in instances:
        if i.state == ModelInstanceStateEnum.ERROR:
            return i.state_message or "Instance failed"
    if elapsed > 15:
        for i in instances:
            if (
                i.state == ModelInstanceStateEnum.PENDING
                and i.state_message
                and "No suitable workers" in i.state_message
            ):
                return i.state_message
    return None


async def poll_until_ready(
    model_ids: List[int], *, mark_last_request: bool = False
) -> PollResult:
    """Long-poll until one of ``model_ids`` has a ready replica (READY), an
    instance errors / can't schedule (FAILED), or the timeout elapses
    (TIMEOUT). Opens a fresh session each tick so it observes committed state
    from the scheduler/controllers."""
    elapsed = 0
    while elapsed < AUTO_LOAD_TIMEOUT:
        await asyncio.sleep(AUTO_LOAD_POLL_INTERVAL)
        elapsed += AUTO_LOAD_POLL_INTERVAL

        async with async_session() as session:
            fresh_models = await Model.all_by_fields(
                session, fields={}, extra_conditions=[Model.id.in_(model_ids)]
            )
            for m in fresh_models:
                if m.ready_replicas > 0:
                    if mark_last_request:
                        m.last_request_time = datetime.now(timezone.utc)
                        await m.update(session)
                    logger.info(
                        f"Auto-load complete for model {m.name}, "
                        f"ready_replicas={m.ready_replicas}"
                    )
                    return PollResult(READY, model=m)

            instances = await ModelInstance.all_by_fields(
                session,
                fields={},
                extra_conditions=[ModelInstance.model_id.in_(model_ids)],
            )
            if instances:
                msg = _detect_load_failure(instances, elapsed)
                if msg:
                    logger.warning(f"Auto-load failed: {msg}")
                    return PollResult(FAILED, message=msg)

    return PollResult(TIMEOUT)
