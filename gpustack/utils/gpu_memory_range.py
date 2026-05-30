"""Helpers for the custom GPU memory range feature.

Lets users declare an absolute VRAM bound (in GiB) per model that overrides
or constrains the ratio-based ``--gpu-memory-utilization``. Used by the
vLLM scheduler (per-candidate clamping) and the vLLM launcher (final
``--gpu-memory-utilization`` injection).
"""

from typing import Iterable, Optional, Sequence

GIB = 1024**3


def effective_gpu_memory_utilization(
    base_gmu: float,
    gpu_total_bytes: int,
    min_gib: Optional[int],
    max_gib: Optional[int],
) -> float:
    """
    Apply the user's absolute VRAM range to a base GPU memory utilization.

    Semantics (matches the user spec):
      - Both bounds None: pass-through ``base_gmu``.
      - Only ``min_gib`` set: floor — force at least that many GiB.
      - Only ``max_gib`` set: ceiling — clamp at most that many GiB.
      - Both set and equal: fixed absolute value.
      - Both set and different: clamp into [min, max] GiB.

    Falls back to ``base_gmu`` if any input is invalid, and stays within
    (0, 1] to keep vLLM happy. ``base_gmu == 0`` is the "non-LLM" sentinel
    used by GPUStack — propagated as 0 unchanged.
    """
    if base_gmu <= 0:
        return 0.0
    if gpu_total_bytes <= 0:
        return base_gmu
    if min_gib is None and max_gib is None:
        return base_gmu

    target_bytes = gpu_total_bytes * base_gmu

    if max_gib is not None and max_gib > 0:
        target_bytes = min(target_bytes, max_gib * GIB)
    if min_gib is not None and min_gib > 0:
        target_bytes = max(target_bytes, min_gib * GIB)

    # When min_gib exceeds the GPU's total VRAM, return eff > 1.0 so the
    # scheduler skips this GPU — the card simply cannot satisfy the floor.
    if target_bytes <= 0:
        return base_gmu

    eff = target_bytes / gpu_total_bytes
    if eff <= 0:
        return base_gmu
    return eff


def anchored_gpu_memory_utilization(
    base_gmu: float,
    gpu_total_bytes_list: Sequence[int],
    min_gib: Optional[int],
    max_gib: Optional[int],
) -> float:
    """
    Resolve a single ``--gpu-memory-utilization`` value for a set of GPUs
    (e.g. one TP group). Anchors on the GPU with the smallest effective
    GMU so the strictest clamp wins, which matches the user's intent of
    aligning to the most-constrained card and releasing surplus VRAM on
    larger cards.
    """
    if not gpu_total_bytes_list:
        return base_gmu
    candidates = [
        effective_gpu_memory_utilization(base_gmu, total, min_gib, max_gib)
        for total in gpu_total_bytes_list
        if total > 0
    ]
    if not candidates:
        return base_gmu
    return min(candidates)


def has_range(min_gib: Optional[int], max_gib: Optional[int]) -> bool:
    return bool((min_gib and min_gib > 0) or (max_gib and max_gib > 0))


def resolve_vllm_gmu_override(
    model,
    selected_gpu_devices: Iterable,
    base_gmu_default: float = 0.9,
) -> Optional[float]:
    """
    Return the ``--gpu-memory-utilization`` value to inject for a vLLM
    launcher when the user has set a GPU memory range on the model;
    otherwise ``None`` (pass-through, let vLLM use whatever the user or
    upstream default decided).

    Anchored on the worker's locally selected GPUs only. For single-worker
    TP this matches the scheduler's anchor exactly (see
    ``vllm_resource_fit_selector._effective_gmu_for_gpu``). For
    multi-worker distributed inference each rank computes its own anchor;
    vLLM tolerates the result because each rank's allocation stays within
    its local card.
    """
    # Local import avoids pulling gpustack.utils.command (and its torch
    # transitive imports) into modules that just need the math helpers.
    from gpustack.utils.command import find_parameter

    min_gib = getattr(model, "gpu_memory_min_gib", None)
    max_gib = getattr(model, "gpu_memory_max_gib", None)
    if not has_range(min_gib, max_gib):
        return None

    base_str = find_parameter(
        getattr(model, "backend_parameters", None), ["gpu-memory-utilization"]
    )
    try:
        base_gmu = float(base_str) if base_str else base_gmu_default
    except (TypeError, ValueError):
        base_gmu = base_gmu_default
    if base_gmu <= 0:
        return None

    totals = [
        gpu.memory.total
        for gpu in (selected_gpu_devices or [])
        if gpu.memory and gpu.memory.total
    ]
    if not totals:
        return None

    return anchored_gpu_memory_utilization(base_gmu, totals, min_gib, max_gib)
