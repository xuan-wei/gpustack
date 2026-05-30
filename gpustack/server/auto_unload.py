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

    async def _check_and_unload_idle_models(self):
        async with async_session() as session:
            models = await Model.all(session)
            now = datetime.now(timezone.utc)
            for model in models:
                if (
                    model.replicas <= 0
                    or not model.auto_unload
                    or not model.last_request_time
                ):
                    continue

                if model.ready_replicas < 1:
                    if await self._is_model_loading(session, model.id):
                        continue

                idle_seconds = (now - model.last_request_time).total_seconds()
                if idle_seconds >= model.auto_unload_timeout * 60:
                    model.replicas = 0
                    await model.update(session)
                    logger.info(
                        "Auto unloaded model %s after %.0f seconds of inactivity",
                        model.name,
                        idle_seconds,
                    )
