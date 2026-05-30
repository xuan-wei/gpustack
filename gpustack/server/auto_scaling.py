"""
Auto-scaling driven by runtime KV-cache pressure and queue depth.

Scale-up:   `num_requests_waiting > 0` sustained ≥ scale_window_minutes
Scale-down: `waiting == 0` AND predicted-after-shrink load < threshold,
            sustained ≥ scale_window_minutes; floor is 1 (auto_unload owns 0)

            - vLLM (KV reported): `kv_sum / (replicas - 1) < scale_down_kv_threshold`
            - llama.cpp (no KV):  `running_total / ((replicas - 1) * slot_capacity)
                                   < scale_down_kv_threshold` where slot_capacity is
                                   parsed from `--parallel`/`-np` in backend_parameters
                                   (default 1, matching upstream llama-server default)

Cooldown:   scale_window_minutes / 2 after any scaling action

Scale-down victim selection:
- vLLM:      instance with lowest KV (most idle)
- llama.cpp: instance with lowest num_requests_running (fewest busy slots)
The victim is marked `is_draining=True`; AutoDrainTask deletes it after gateway
endpoint convergence.
"""

import asyncio
import logging
import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


# Flag names that set llama.cpp's concurrent slot count. Long form takes a
# following value (or `=value`); short form `-np` likewise.
_LLAMA_CPP_PARALLEL_FLAGS = {"--parallel", "-np", "--n-parallel"}


def _llama_cpp_slot_capacity(model: Model) -> int:
    """Resolve concurrent slot capacity per replica from `backend_parameters`.

    llama-server has no `/metrics` field for total slots, and `/props` works
    only on upstream llama-server (not llama-box) — parsing the config we
    already pass on the command line covers both. Default 1 matches upstream
    llama-server's default `-np 1` when the user omits the flag.
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
        # `--parallel=10`
        if "=" in tok:
            key, _, val = tok.partition("=")
            if key in _LLAMA_CPP_PARALLEL_FLAGS:
                try:
                    capacity = max(1, int(val))
                except ValueError:
                    pass
            i += 1
            continue
        # `--parallel 10`
        if tok in _LLAMA_CPP_PARALLEL_FLAGS and i + 1 < len(tokens):
            try:
                capacity = max(1, int(tokens[i + 1]))
            except ValueError:
                pass
            i += 2
            continue
        i += 1
    return capacity


@dataclass
class _ScaleState:
    up_first_met_at: Optional[datetime] = None
    down_first_met_at: Optional[datetime] = None
    last_action_at: Optional[datetime] = None


class AutoScalingTask:
    def __init__(self, interval: int = 30, aggregator=None):
        self.interval = interval
        self.task: Optional[asyncio.Task] = None
        self._aggregator = aggregator
        self._state: Dict[int, _ScaleState] = {}

    async def start(self):
        if self.task and not self.task.done():
            logger.warning("Auto scaling task is already running")
            return
        self.task = asyncio.create_task(self._run())
        logger.info("Auto scaling task started")

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            logger.info("Auto scaling task stopped")

    async def _run(self):
        while True:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Error in auto scaling task: {e}", exc_info=True)
            await asyncio.sleep(self.interval)

    async def _tick(self):
        if self._aggregator is None:
            return
        async with async_session() as session:
            models = await Model.all(session)
            for model in models:
                if not model.auto_adjust_replicas:
                    self._state.pop(model.id, None)
                    continue
                try:
                    await self._evaluate(session, model)
                except Exception as e:
                    logger.error(
                        f"Auto scaling evaluation failed for model {model.name}: {e}",
                        exc_info=True,
                    )

    async def _evaluate(self, session, model: Model):
        snap = self._aggregator.get_snapshot(model.id)
        state = self._state.setdefault(model.id, _ScaleState())
        now = datetime.now(timezone.utc)
        window = timedelta(minutes=max(1, model.scale_window_minutes))
        cooldown = timedelta(minutes=max(0.5, model.scale_window_minutes / 2))

        # Don't react while cold-starting: wait for ready_replicas to match replicas.
        if model.ready_replicas < model.replicas:
            state.up_first_met_at = None
            state.down_first_met_at = None
            return

        # No metrics yet → wait for next tick.
        if snap is None or snap.instance_count == 0:
            state.up_first_met_at = None
            state.down_first_met_at = None
            return

        if state.last_action_at and now - state.last_action_at < cooldown:
            return

        ceiling = max(1, model.auto_load_replicas)

        # ---- Scale up: queue building up ----
        if snap.waiting_total > 0 and model.replicas < ceiling:
            state.down_first_met_at = None
            if state.up_first_met_at is None:
                state.up_first_met_at = now
                return
            if now - state.up_first_met_at >= window:
                await self._scale_up(session, model, snap, now, state)
            return
        else:
            state.up_first_met_at = None

        # ---- Scale down: capacity headroom available ----
        if model.replicas > 1 and snap.waiting_total == 0:
            if snap.kv_supported:
                predicted = snap.kv_sum / (model.replicas - 1)
            else:
                # llama.cpp path: predict per-replica busy-slot utilization
                # after removing one replica. running_total is integer-ish
                # (one slot per active request); slot_capacity is the per-
                # replica `--parallel` value.
                slot_capacity = _llama_cpp_slot_capacity(model)
                predicted = snap.running_total / ((model.replicas - 1) * slot_capacity)
            if predicted < model.scale_down_kv_threshold:
                if state.down_first_met_at is None:
                    state.down_first_met_at = now
                    return
                if now - state.down_first_met_at >= window:
                    await self._scale_down(session, model, snap, now, state)
                return
        state.down_first_met_at = None

    async def _scale_up(
        self, session, model: Model, snap, now: datetime, state: _ScaleState
    ):
        previous = model.replicas
        model.replicas = min(previous + 1, max(1, model.auto_load_replicas))
        if model.replicas == previous:
            return
        model.last_scale_time = now
        model.last_scale_message = (
            f"up: waiting={snap.waiting_total}, kv_sum={snap.kv_sum:.2f}, "
            f"{previous}->{model.replicas}"
        )
        await model.update(session)
        state.last_action_at = now
        state.up_first_met_at = None
        logger.info(
            f"Model {model.name}: scale up {previous} -> {model.replicas} "
            f"(waiting={snap.waiting_total})"
        )

    async def _scale_down(
        self, session, model: Model, snap, now: datetime, state: _ScaleState
    ):
        # vLLM: pick lowest-KV instance. llama.cpp: pick lowest busy-slot
        # instance (instance_running). instance_running is populated for all
        # backends, so it's also a safe fallback if instance_kv is empty.
        if snap.kv_supported and snap.instance_kv:
            load_map = snap.instance_kv
            load_label = "kv"
        elif snap.instance_running:
            load_map = snap.instance_running
            load_label = "running"
        else:
            return

        candidate_instance_id = None
        candidate_load = None
        candidate_inst = None
        for inst_id, load in sorted(load_map.items(), key=lambda x: x[1]):
            inst = await ModelInstance.one_by_id(session, inst_id)
            if inst is None or inst.is_draining:
                continue
            if inst.state != ModelInstanceStateEnum.RUNNING:
                continue
            candidate_instance_id = inst_id
            candidate_load = load
            candidate_inst = inst
            break

        if candidate_instance_id is None:
            return

        previous = model.replicas
        new_replicas = max(1, previous - 1)
        if new_replicas == previous:
            return

        # Mark instance draining AND decrease model.replicas in the same
        # transaction. Otherwise sync_replicas can observe an intermediate
        # state (e.g. active < replicas after marking draining but before
        # decreasing replicas) and create a phantom replacement instance.
        candidate_inst.is_draining = True
        await candidate_inst.update(session, auto_commit=False)

        model.replicas = new_replicas
        model.last_scale_time = now
        if load_label == "kv":
            model.last_scale_message = (
                f"down: kv_sum={snap.kv_sum:.2f}, "
                f"pred_kv={snap.kv_sum / max(1, previous - 1):.2f}, "
                f"victim={candidate_instance_id}(kv={candidate_load:.2f}), "
                f"{previous}->{model.replicas}"
            )
        else:
            slot_capacity = _llama_cpp_slot_capacity(model)
            pred = snap.running_total / max(1, (previous - 1) * slot_capacity)
            model.last_scale_message = (
                f"down: running_total={snap.running_total:.0f}, "
                f"slot_capacity={slot_capacity}, pred_util={pred:.2f}, "
                f"victim={candidate_instance_id}(running={candidate_load:.0f}), "
                f"{previous}->{model.replicas}"
            )
        await model.update(session)
        state.last_action_at = now
        state.down_first_met_at = None
        logger.info(
            f"Model {model.name}: scale down {previous} -> {model.replicas} "
            f"(victim instance_id={candidate_instance_id} {load_label}={candidate_load:.2f})"
        )
