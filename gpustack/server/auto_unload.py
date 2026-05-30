import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)

LOADING_STATES = {
    ModelInstanceStateEnum.PENDING,
    ModelInstanceStateEnum.ANALYZING,
    ModelInstanceStateEnum.SCHEDULED,
    ModelInstanceStateEnum.INITIALIZING,
    ModelInstanceStateEnum.DOWNLOADING,
    ModelInstanceStateEnum.STARTING,
}


class AutoUnloadTask:
    def __init__(self, interval: int = 60):
        self.interval = interval
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        if self.task and not self.task.done():
            logger.warning("Auto unload task is already running")
            return

        self.task = asyncio.create_task(self._run())
        logger.info("Auto unload task started")

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            logger.info("Auto unload task stopped")

    async def _run(self):
        while True:
            try:
                await self._check_and_unload_idle_models()
            except Exception as e:
                logger.error(f"Error in auto unload task: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    async def _is_model_loading(self, session, model_id: int) -> bool:
        instances = await ModelInstance.all_by_field(session, "model_id", model_id)
        return any(i.state in LOADING_STATES for i in instances)

    async def _has_active_requests(self, model_id: int) -> bool:
        """Check runtime snapshot for in-flight requests."""
        try:
            from gpustack.server.app import app

            aggregator = getattr(app.state, "runtime_metrics_aggregator", None)
            if aggregator is None:
                return False
            snap = aggregator.get_snapshot(model_id)
            if snap is None:
                return False
            return (snap.running_total + snap.waiting_total) > 0
        except Exception:
            return False

    async def _check_and_unload_idle_models(self):
        async with async_session() as session:
            models = await Model.all(session)
            now = datetime.now(timezone.utc)
            for model in models:
                if model.replicas <= 0 or not model.auto_unload:
                    continue

                if model.ready_replicas < 1:
                    if await self._is_model_loading(session, model.id):
                        continue

                candidates = [
                    t
                    for t in (
                        model.last_request_time,
                        model.last_scale_time,
                        model.created_at,
                    )
                    if t
                ]
                ref_time = (
                    max(candidates)
                    if candidates
                    else datetime(1970, 1, 1, tzinfo=timezone.utc)
                )
                idle_seconds = (now - ref_time).total_seconds()
                if idle_seconds >= model.auto_unload_timeout * 60:
                    if await self._has_active_requests(model.id):
                        logger.debug(
                            "Skipping auto unload of %s: still has active requests",
                            model.name,
                        )
                        continue
                    model.replicas = 0
                    await model.update(session)
                    logger.info(
                        "Auto unloaded model %s after %.0f seconds of inactivity",
                        model.name,
                        idle_seconds,
                    )
