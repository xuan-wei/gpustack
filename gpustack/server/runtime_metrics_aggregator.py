"""
Server-side runtime metrics aggregator.

Polls each Worker's metrics exporter (`gpustack:*` unified metrics),
groups samples by `model_id`, and keeps the latest snapshot in memory.

Consumers:
- `AutoScalingTask` reads snapshots to decide scale up/down
- HTTP endpoint `/v1/models/runtime-snapshots` returns snapshots to UI
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from aiohttp import ClientSession as aiohttp_client, ClientTimeout
from prometheus_client.parser import text_string_to_metric_families

from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


# Histograms whose mean we expose (delta_sum / delta_count between ticks).
_HISTOGRAM_METRICS = {
    "gpustack:time_to_first_token_seconds": "ttft_avg_ms",
    "gpustack:time_per_output_token_seconds": "tpot_avg_ms",
}

_GAUGE_METRICS = {
    "gpustack:num_requests_running": "running",
    "gpustack:num_requests_waiting": "waiting",
    "gpustack:kv_cache_usage_ratio": "kv",
}

_COUNTER_METRICS = {
    "gpustack:prompt_tokens": "prompt_tokens",
    "gpustack:generation_tokens": "generation_tokens",
    # Cumulative seconds counters paired with the token counters above.
    # Used to derive TTFT/TPOT for backends (e.g. llama.cpp) that don't expose
    # latency histograms — see _tick() fallback below.
    "gpustack:prompt_seconds": "prompt_seconds",
    "gpustack:generation_seconds": "generation_seconds",
}


@dataclass
class InstanceSnapshot:
    instance_id: int
    worker_id: Optional[int] = None
    worker_name: str = ""
    running: float = 0.0
    waiting: float = 0.0
    # None when the backend doesn't expose kv_cache_usage_ratio (e.g. llama.cpp).
    kv: Optional[float] = None
    prompt_tokens: float = 0.0
    generation_tokens: float = 0.0
    # Cumulative prefill/decode seconds (only some backends expose these).
    prompt_seconds: float = 0.0
    generation_seconds: float = 0.0
    # raw histogram cumulative sum+count (seconds, requests)
    ttft_sum: float = 0.0
    ttft_count: float = 0.0
    tpot_sum: float = 0.0
    tpot_count: float = 0.0


@dataclass
class ModelRuntimeSnapshot:
    model_id: int
    running_total: float = 0.0
    waiting_total: float = 0.0
    kv_per_replica: List[float] = field(default_factory=list)
    # None when no replica reported KV (e.g. all replicas are llama.cpp).
    kv_avg: Optional[float] = None
    kv_sum: float = 0.0
    kv_supported: bool = False
    ttft_avg_ms: Optional[float] = None  # strict cross-instance weighted mean
    tpot_avg_ms: Optional[float] = None
    # Output token throughput (gpustack:generation_tokens delta / tick_seconds),
    # summed across all replicas. None on the first tick or if no counters moved.
    gen_throughput_tok_s: Optional[float] = None
    instance_kv: Dict[int, float] = field(default_factory=dict)
    # Per-instance busy-slot count (num_requests_running). Used by auto-scaling
    # to pick the least-loaded victim when KV is not reported (llama.cpp).
    instance_running: Dict[int, float] = field(default_factory=dict)
    instance_waiting: Dict[int, float] = field(default_factory=dict)
    instance_count: int = 0


class RuntimeMetricsAggregator:
    """Polls workers and produces per-model runtime snapshots."""

    def __init__(self, interval: int = 30):
        self._interval = interval
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[aiohttp_client] = None
        # Latest aggregated snapshot per model_id; consumers read this.
        self._snapshots: Dict[int, ModelRuntimeSnapshot] = {}
        # Previous-tick raw histogram totals per (model_id, instance_id), for delta means.
        self._prev_hist: Dict[tuple, Dict[str, float]] = {}
        # Wall-clock time of the last successful tick; None until first tick completes.
        self._last_tick_at: Optional[datetime] = None
        # Serializes force_tick to avoid concurrent worker scrapes.
        self._tick_lock = asyncio.Lock()

    async def start(self):
        if self._task and not self._task.done():
            return
        self._client = aiohttp_client(timeout=ClientTimeout(total=5))
        self._task = asyncio.create_task(self._run())
        logger.info("Runtime metrics aggregator started")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.close()
        logger.info("Runtime metrics aggregator stopped")

    def get_snapshot(self, model_id: int) -> Optional[ModelRuntimeSnapshot]:
        return self._snapshots.get(model_id)

    def all_snapshots(self) -> Dict[int, ModelRuntimeSnapshot]:
        return dict(self._snapshots)

    @property
    def last_tick_at(self) -> Optional[datetime]:
        return self._last_tick_at

    async def force_tick(self, min_interval_seconds: float = 5.0) -> bool:
        """
        Trigger an out-of-cycle scrape on user demand. Debounced: if the last
        tick (background OR forced) completed less than `min_interval_seconds`
        ago, this is a no-op and returns False. Otherwise runs `_tick` under
        a lock and returns True. Errors are swallowed and logged.
        """
        now = datetime.now(timezone.utc)
        if (
            self._last_tick_at is not None
            and (now - self._last_tick_at).total_seconds() < min_interval_seconds
        ):
            return False
        async with self._tick_lock:
            # Re-check under the lock in case another caller just ticked.
            now = datetime.now(timezone.utc)
            if (
                self._last_tick_at is not None
                and (now - self._last_tick_at).total_seconds() < min_interval_seconds
            ):
                return False
            try:
                await self._tick()
                return True
            except Exception as e:
                logger.error(f"force_tick failed: {e}", exc_info=True)
                return False

    async def _run(self):
        while True:
            try:
                async with self._tick_lock:
                    await self._tick()
            except Exception as e:
                logger.error(f"Error in runtime metrics aggregator: {e}", exc_info=True)
            await asyncio.sleep(self._interval)

    async def _tick(self):  # noqa: C901
        # Capture the previous tick timestamp BEFORE doing any work, so we can
        # compute throughput = (sum of generation_tokens deltas) / tick_seconds
        # at the end of this tick.
        prev_tick_at = self._last_tick_at
        async with async_session() as session:
            workers = await Worker.all(session)

        urls = []
        for w in workers:
            if w.state != WorkerStateEnum.READY:
                continue
            ip = w.advertise_address or w.ip
            port = getattr(w, "metrics_port", None)
            if not ip or not port or port == -1:
                continue
            urls.append(f"http://{ip}:{port}/metrics")

        per_instance: Dict[int, InstanceSnapshot] = {}
        instance_model: Dict[int, int] = {}

        results = await asyncio.gather(
            *(self._fetch(url) for url in urls), return_exceptions=True
        )
        for text in results:
            if isinstance(text, Exception) or not text:
                continue
            self._parse_into(text, per_instance, instance_model)

        # Group by model_id; compute deltas for histogram means.
        new_snaps: Dict[int, ModelRuntimeSnapshot] = {}
        new_prev: Dict[tuple, Dict[str, float]] = {}

        # Per-model accumulators for strict cross-instance weighted means.
        # Built fresh each tick; values are sums of deltas across all replicas
        # of one model, so the final mean is total_sum / total_count regardless
        # of replica count or iteration order.
        accum: Dict[int, Dict[str, float]] = {}

        def _acc(mid: int) -> Dict[str, float]:
            return accum.setdefault(
                mid,
                {
                    "hist_ttft_sum": 0.0,
                    "hist_ttft_count": 0.0,
                    "hist_tpot_sum": 0.0,
                    "hist_tpot_count": 0.0,
                    "pair_ttft_sec": 0.0,
                    "pair_ttft_tok": 0.0,
                    "pair_tpot_sec": 0.0,
                    "pair_tpot_tok": 0.0,
                    # Sum of generation_tokens delta across instances this tick.
                    "gen_token_delta": 0.0,
                },
            )

        for instance_id, inst in per_instance.items():
            model_id = instance_model.get(instance_id)
            if model_id is None:
                continue
            snap = new_snaps.setdefault(
                model_id, ModelRuntimeSnapshot(model_id=model_id)
            )
            snap.running_total += inst.running
            snap.waiting_total += inst.waiting
            snap.instance_running[instance_id] = inst.running
            snap.instance_waiting[instance_id] = inst.waiting
            # Only count KV from replicas that actually reported it. llama.cpp
            # never reports kv_cache_usage_ratio so it stays None — surfaced as
            # `kv_avg: null` / `kv_supported: false` to the API consumer.
            if inst.kv is not None:
                snap.kv_per_replica.append(inst.kv)
                snap.kv_sum += inst.kv
                snap.instance_kv[instance_id] = inst.kv
                snap.kv_supported = True
            snap.instance_count += 1

            # Accumulate histogram and counter-pair deltas for weighted means.
            key = (model_id, instance_id)
            new_prev[key] = {
                "ttft_sum": inst.ttft_sum,
                "ttft_count": inst.ttft_count,
                "tpot_sum": inst.tpot_sum,
                "tpot_count": inst.tpot_count,
                # Counter-pair cumulative values (llama.cpp et al)
                "prompt_seconds": inst.prompt_seconds,
                "prompt_tokens": inst.prompt_tokens,
                "generation_seconds": inst.generation_seconds,
                "generation_tokens": inst.generation_tokens,
            }
            prev = self._prev_hist.get(key)
            if prev is not None:
                a = _acc(model_id)
                d_ttft_count = inst.ttft_count - prev["ttft_count"]
                d_ttft_sum = inst.ttft_sum - prev["ttft_sum"]
                if d_ttft_count > 0 and d_ttft_sum >= 0:
                    a["hist_ttft_sum"] += d_ttft_sum
                    a["hist_ttft_count"] += d_ttft_count
                d_tpot_count = inst.tpot_count - prev["tpot_count"]
                d_tpot_sum = inst.tpot_sum - prev["tpot_sum"]
                if d_tpot_count > 0 and d_tpot_sum >= 0:
                    a["hist_tpot_sum"] += d_tpot_sum
                    a["hist_tpot_count"] += d_tpot_count

                # Counter-pair fallback for backends without latency histograms
                # (e.g. llama.cpp). For TTFT this is a proxy: ms-per-prompt-token,
                # not actual time-to-first-token, but trends the same direction.
                d_psec = inst.prompt_seconds - prev["prompt_seconds"]
                d_ptok = inst.prompt_tokens - prev["prompt_tokens"]
                if d_psec > 0 and d_ptok > 0:
                    a["pair_ttft_sec"] += d_psec
                    a["pair_ttft_tok"] += d_ptok
                d_gsec = inst.generation_seconds - prev["generation_seconds"]
                d_gtok = inst.generation_tokens - prev["generation_tokens"]
                if d_gsec > 0 and d_gtok > 0:
                    a["pair_tpot_sec"] += d_gsec
                    a["pair_tpot_tok"] += d_gtok
                # Output throughput: accumulate the generation_tokens delta
                # regardless of whether seconds was also reported. Skipped when
                # the counter went backwards (instance restarted).
                if d_gtok > 0:
                    a["gen_token_delta"] += d_gtok

        # Finalize per-model means.
        for model_id, snap in new_snaps.items():
            # KV avg: only over replicas that actually reported KV.
            kv_reporting = len(snap.kv_per_replica)
            if snap.kv_supported and kv_reporting > 0:
                snap.kv_avg = snap.kv_sum / kv_reporting
            else:
                snap.kv_avg = None

            a = accum.get(model_id)
            if a is None:
                continue
            # Prefer histogram-derived weighted mean; fall back to counter pair.
            if a["hist_ttft_count"] > 0:
                snap.ttft_avg_ms = (a["hist_ttft_sum"] / a["hist_ttft_count"]) * 1000.0
            elif a["pair_ttft_tok"] > 0:
                snap.ttft_avg_ms = (a["pair_ttft_sec"] / a["pair_ttft_tok"]) * 1000.0
            if a["hist_tpot_count"] > 0:
                snap.tpot_avg_ms = (a["hist_tpot_sum"] / a["hist_tpot_count"]) * 1000.0
            elif a["pair_tpot_tok"] > 0:
                snap.tpot_avg_ms = (a["pair_tpot_sec"] / a["pair_tpot_tok"]) * 1000.0

            # Output throughput tok/s = total generation tokens emitted across
            # all replicas this tick divided by tick interval. Needs at least
            # one prior tick to have a delta.
            if prev_tick_at is not None and a["gen_token_delta"] > 0:
                tick_seconds = (
                    datetime.now(timezone.utc) - prev_tick_at
                ).total_seconds()
                if tick_seconds > 0:
                    snap.gen_throughput_tok_s = a["gen_token_delta"] / tick_seconds

        self._snapshots = new_snaps
        self._prev_hist = new_prev
        self._last_tick_at = datetime.now(timezone.utc)

    async def _fetch(self, url: str) -> Optional[str]:
        try:
            async with self._client.get(url) as resp:
                if resp.status != 200:
                    logger.debug(f"Runtime metrics fetch {url}: HTTP {resp.status}")
                    return None
                return await resp.text()
        except Exception as e:
            logger.debug(f"Runtime metrics fetch {url} failed: {e}")
            return None

    def _parse_into(
        self,
        text: str,
        per_instance: Dict[int, InstanceSnapshot],
        instance_model: Dict[int, int],
    ):
        for family in text_string_to_metric_families(text):
            name = family.name
            is_gauge = name in _GAUGE_METRICS
            is_counter = name in _COUNTER_METRICS
            is_hist = name in _HISTOGRAM_METRICS
            if not (is_gauge or is_counter or is_hist):
                continue
            for sample in family.samples:
                labels = sample.labels
                mid_str = labels.get("model_id") or ""
                iid_str = labels.get("model_instance_id") or ""
                if not mid_str.isdigit() or not iid_str.isdigit():
                    continue
                instance_id = int(iid_str)
                inst = per_instance.setdefault(
                    instance_id,
                    InstanceSnapshot(
                        instance_id=instance_id,
                        worker_id=(
                            int(labels["worker_id"])
                            if labels.get("worker_id", "").isdigit()
                            else None
                        ),
                        worker_name=labels.get("worker_name", "") or "",
                    ),
                )
                instance_model[instance_id] = int(mid_str)

                # Gauges: name == sample.name (no _bucket / _sum / _count suffix)
                if is_gauge and sample.name == name:
                    field_name = _GAUGE_METRICS[name]
                    setattr(inst, field_name, sample.value)
                elif is_counter and sample.name == name + "_total":
                    field_name = _COUNTER_METRICS[name]
                    setattr(inst, field_name, sample.value)
                elif is_counter and sample.name == name:
                    # some exporters emit counter without _total
                    field_name = _COUNTER_METRICS[name]
                    setattr(inst, field_name, sample.value)
                elif is_hist:
                    short = _HISTOGRAM_METRICS[name].replace("_avg_ms", "")
                    if sample.name == name + "_sum":
                        setattr(inst, f"{short}_sum", sample.value)
                    elif sample.name == name + "_count":
                        setattr(inst, f"{short}_count", sample.value)
