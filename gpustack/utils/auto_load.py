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
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.model_routes import ModelRouteTarget
from gpustack.server.db import async_session
from gpustack.utils.cold_start_gate import cold_start_slot

logger = logging.getLogger(__name__)

AUTO_LOAD_POLL_INTERVAL = 3
AUTO_LOAD_TIMEOUT = 300

# Poll outcomes.
READY = "ready"
FAILED = "failed"
TIMEOUT = "timeout"
SKIPPED = "skipped"


@dataclass
class PollResult:
    outcome: str
    model: Optional[Model] = None  # set when READY
    message: Optional[str] = None  # set when FAILED


@dataclass
class AutoLoadResult:
    outcome: str  # READY / SKIPPED / TIMEOUT / FAILED
    model: Optional[Model] = None
    message: Optional[str] = None


_LLAMA_CPP_PARALLEL_FLAGS = {"--parallel", "-np", "--n-parallel"}


def llama_cpp_slot_capacity(model: Model) -> int:
    """Resolve concurrent slot capacity per replica from `backend_parameters`.

    llama-server's total slots aren't exposed via /metrics, so we parse the
    command-line config. Default 1 matches upstream llama-server's `-np 1`.
    """
    params = model.backend_parameters or []
    tokens: list[str] = []
    for raw in params:
        try:
            tokens.extend(shlex.split(raw))
        except ValueError:
            tokens.extend(raw.split())
    i = 0
    capacity = 1
    while i < len(tokens):
        tok = tokens[i]
        if "=" in tok:
            key, _, val = tok.partition("=")
            if key in _LLAMA_CPP_PARALLEL_FLAGS:
                try:
                    capacity = max(1, int(val))
                except ValueError:
                    pass
            i += 1
            continue
        if tok in _LLAMA_CPP_PARALLEL_FLAGS and i + 1 < len(tokens):
            try:
                capacity = max(1, int(tokens[i + 1]))
            except ValueError:
                pass
            i += 2
            continue
        i += 1
    return capacity


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
    deadline = time.monotonic() + AUTO_LOAD_TIMEOUT
    while time.monotonic() < deadline:
        await asyncio.sleep(AUTO_LOAD_POLL_INTERVAL)
        elapsed = AUTO_LOAD_TIMEOUT - (deadline - time.monotonic())

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


async def try_auto_load(session, model_name: str) -> AutoLoadResult:
    """Unified auto-load entry point used by both token.py and openai.py.

    Resolves a route/model name, triggers replica scaling if needed,
    and long-polls until the model is ready (or fails/times out).
    ColdStartCapacityExceeded is NOT caught here — callers handle it.
    """
    model_ids = await resolve_route_model_ids(session, model_name)
    if not model_ids:
        return AutoLoadResult(SKIPPED)

    models = await Model.all_by_fields(
        session, fields={}, extra_conditions=[Model.id.in_(model_ids)]
    )

    needs_loading = False
    now = datetime.now(timezone.utc)

    for model in models:
        if model.auto_load and model.replicas == 0:
            if model.auto_adjust_replicas:
                target = max(model.auto_load_replicas // 2, 1)
            else:
                target = max(model.auto_load_replicas, 1)
            model.replicas = target
            model.last_request_time = now
            await model.update(session)
            needs_loading = True
            logger.info(
                f"Auto-load triggered for model {model.name}, "
                f"scaling to {model.replicas} replicas"
            )
            continue

        if (
            model.auto_load
            and not model.auto_adjust_replicas
            and model.replicas > 0
            and model.replicas != max(model.auto_load_replicas, 1)
        ):
            target = max(model.auto_load_replicas, 1)
            logger.info(
                f"Reconciling replicas for model {model.name}, "
                f"{model.replicas} -> {target}"
            )
            model.replicas = target
            model.last_request_time = now
            await model.update(session)

        if model.replicas > 0 and model.ready_replicas == 0:
            needs_loading = True
            continue

        if (
            model.replicas > 0
            and model.ready_replicas > 0
            and model.auto_unload
            and (
                model.last_request_time is None
                or (now - model.last_request_time).total_seconds() > 5
            )
        ):
            model.last_request_time = now
            await model.update(session)

    if not needs_loading:
        return AutoLoadResult(SKIPPED)

    async with cold_start_slot(model_name):
        result = await poll_until_ready(model_ids)

    return AutoLoadResult(
        outcome=result.outcome,
        model=result.model,
        message=result.message,
    )
