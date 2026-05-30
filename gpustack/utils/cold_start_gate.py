"""
Per-model cold-start gate (custom v2.1.2 feature).

`_try_auto_load` in routes/openai.py long-polls up to AUTO_LOAD_TIMEOUT (5min)
waiting for a model that was at replicas=0 to come up. If thousands of
requests arrive against a cold model in the same instant, every handler
enters that long-poll — saturating the asyncio task pool and the upstream
connection pool with idle waiters that hold resources for minutes.

This gate caps the number of handlers that can simultaneously be in the
cold-start wait for a single model. Excess requests fail fast with
`ColdStartCapacityExceeded`, which the route layer converts to a 503
ServiceUnavailable with Retry-After so clients can back off.

Why a hand-rolled counter instead of asyncio.Semaphore:
  - We want **immediate fail**, not blocking acquire. `Semaphore.acquire()`
    would block; `wait_for(acquire(), timeout=0)` is awkward semantically.
  - The counter+lock pattern is tiny, correct, and self-documenting.

Disabled (no-op) when `envs.COLD_START_MAX_WAITERS <= 0`.
"""

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict

from gpustack import envs

logger = logging.getLogger(__name__)


_in_flight: Dict[str, int] = defaultdict(int)
_lock = asyncio.Lock()


class ColdStartCapacityExceeded(Exception):
    """Raised when too many handlers are simultaneously waiting for a model
    to finish cold-starting. Caller should translate to 503 Retry-After."""

    def __init__(self, model_name: str, current: int, cap: int):
        self.model_name = model_name
        self.current = current
        self.cap = cap
        super().__init__(
            f"Cold-start capacity exceeded for model {model_name}: "
            f"{current}/{cap} waiters in flight"
        )


@asynccontextmanager
async def cold_start_slot(model_name: str):
    """Acquire a slot in the per-model cold-start waiting queue.

    Use as ``async with cold_start_slot(name): ...`` around the section of
    code that is doing the long-poll wait. Raises ColdStartCapacityExceeded
    immediately if the cap is full — does not block.
    """
    cap = envs.COLD_START_MAX_WAITERS
    if cap <= 0:
        yield
        return

    async with _lock:
        current = _in_flight[model_name]
        if current >= cap:
            raise ColdStartCapacityExceeded(model_name, current, cap)
        _in_flight[model_name] = current + 1

    try:
        yield
    finally:
        async with _lock:
            new_val = _in_flight[model_name] - 1
            if new_val <= 0:
                _in_flight.pop(model_name, None)
            else:
                _in_flight[model_name] = new_val


def get_in_flight_count(model_name: str) -> int:
    """Observability helper: how many waiters this model currently has.

    Not thread-safe between read and use; intended for log / metric snapshots
    only, not for control flow."""
    return _in_flight.get(model_name, 0)
