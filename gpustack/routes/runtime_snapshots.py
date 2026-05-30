"""Runtime snapshot endpoint (custom, fork-specific).

Extracted from routes/models.py so this fork-only endpoint does not grow that
official file's diff and does not depend on in-file route ordering relative to
``/{id}`` (this router is registered before models.router in routes.py, so
``/models/runtime-snapshots`` is always matched before ``/models/{id}``).
"""

from fastapi import APIRouter, Request

from gpustack.schemas.models import Model
from gpustack.server.db import async_session

router = APIRouter()


@router.get("/runtime-snapshots")
async def get_runtime_snapshots(request: Request, force_refresh: bool = False):
    """
    Return latest in-memory runtime snapshot for each model (running, waiting,
    load %, TTFT/TPOT means, output throughput, req/min). Driven by
    RuntimeMetricsAggregator + each model's avg_request_rate from auto_metrics.

    `load_ratio` unifies vLLM (KV cache usage) and llama.cpp (busy slot ratio)
    onto one 0–1 scale, so the UI shows a single percentage per model.

    With `?force_refresh=1`, trigger an out-of-cycle scrape before returning.
    Debounced server-side (5s) — multiple concurrent callers do not multiply
    the worker load.
    """
    from gpustack.server.auto_scaling import _llama_cpp_slot_capacity

    aggregator = getattr(request.app.state, "runtime_metrics_aggregator", None)
    if aggregator is None:
        return {"snapshots": {}, "updated_at": None, "refreshed": False}
    refreshed = False
    if force_refresh:
        refreshed = await aggregator.force_tick(min_interval_seconds=5.0)

    snapshots = aggregator.all_snapshots()

    # Fetch the models we have snapshots for so we can compute llama.cpp slot
    # capacity (from backend_parameters) and surface avg_request_rate (computed
    # separately by auto_metrics.py).
    model_lookup = {}
    if snapshots:
        async with async_session() as session:
            for mid in snapshots.keys():
                model = await Model.one_by_id(session, mid)
                if model is not None:
                    model_lookup[mid] = model

    out = {}
    for model_id, snap in snapshots.items():
        model = model_lookup.get(model_id)
        replicas = getattr(model, "replicas", 0) if model else 0

        # Unified load ratio across backends. vLLM: KV usage. llama.cpp:
        # busy-slot ratio (running_total / total slot capacity).
        #
        # slot_capacity is parsed from the model's current backend_parameters,
        # which can briefly disagree with a still-running instance whose
        # --parallel was changed but not yet rebuilt. In that window the raw
        # ratio can exceed 1 (e.g. 8 running against a stale capacity of 1 ->
        # 800%). load_ratio is a 0–1 utilization gauge, so clamp to [0, 1];
        # a >100% display is never meaningful and self-corrects on rebuild.
        if snap.kv_supported and snap.kv_avg is not None:
            load_ratio = snap.kv_avg
        elif model is not None and replicas > 0:
            slot_capacity = _llama_cpp_slot_capacity(model)
            total_slots = replicas * slot_capacity
            load_ratio = snap.running_total / total_slots if total_slots > 0 else None
        else:
            load_ratio = None
        if load_ratio is not None:
            load_ratio = max(0.0, min(1.0, load_ratio))

        out[str(model_id)] = {
            "running": int(snap.running_total),
            "waiting": int(snap.waiting_total),
            "kv_avg": (round(snap.kv_avg, 4) if snap.kv_avg is not None else None),
            "kv_per_replica": [round(v, 4) for v in snap.kv_per_replica],
            "kv_supported": snap.kv_supported,
            "load_ratio": (round(load_ratio, 4) if load_ratio is not None else None),
            "ttft_avg_ms": (
                round(snap.ttft_avg_ms, 2) if snap.ttft_avg_ms is not None else None
            ),
            "tpot_avg_ms": (
                round(snap.tpot_avg_ms, 2) if snap.tpot_avg_ms is not None else None
            ),
            "gen_throughput_tok_s": (
                round(snap.gen_throughput_tok_s, 2)
                if snap.gen_throughput_tok_s is not None
                else None
            ),
            "req_per_min": (
                round(model.avg_request_rate, 2)
                if model is not None and model.avg_request_rate is not None
                else None
            ),
            "instance_count": snap.instance_count,
        }
    last_tick = aggregator.last_tick_at
    return {
        "snapshots": out,
        "updated_at": last_tick.isoformat() if last_tick is not None else None,
        "refreshed": refreshed,
    }
