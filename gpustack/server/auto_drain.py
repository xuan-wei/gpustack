import asyncio
import logging
from typing import Optional, Set

from gpustack.schemas.models import Model, ModelInstance
from gpustack.server.bus import Event, EventType, event_bus
from gpustack.server.db import async_session
from gpustack.server.services import ModelInstanceService

logger = logging.getLogger(__name__)


class AutoDrainTask:
    """
    Finalizes scale-down by deleting model instances marked `is_draining=True`
    after giving the gateway routing layer (Higress) time to converge so that
    in-flight requests aren't sent to terminating workers.
    """

    def __init__(self, interval: int = 10, drain_wait_seconds: int = 5):
        self.interval = interval
        self.drain_wait_seconds = drain_wait_seconds
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        if self.task and not self.task.done():
            logger.warning("Auto drain task is already running")
            return

        self.task = asyncio.create_task(self._run())
        logger.info("Auto drain task started")

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            logger.info("Auto drain task stopped")

    async def _run(self):
        while True:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Error in auto drain task: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    async def _tick(self):
        instance_ids: list[int] = []
        model_ids: Set[int] = set()

        async with async_session() as session:
            draining = await ModelInstance.all_by_field(session, "is_draining", True)
            for inst in draining:
                instance_ids.append(inst.id)
                if inst.model_id is not None:
                    model_ids.add(inst.model_id)

        if not instance_ids:
            return

        # Defensive: re-publish synthetic Model UPDATED events so that the
        # gateway sync path runs again. Normally sync_replicas (which set
        # is_draining=True) already triggered this chain via Model UPDATED's
        # notify_model_route_target. This handles edge cases like a server
        # restart leaving stale draining rows.
        await self._republish_model_events(model_ids)

        # Wait for gateway routing layer to drop the draining endpoints.
        await asyncio.sleep(self.drain_wait_seconds)

        async with async_session() as session:
            for inst_id in instance_ids:
                inst = await ModelInstance.one_by_id(session, inst_id)
                if inst is None or not inst.is_draining:
                    continue
                try:
                    await ModelInstanceService(session).delete(inst)
                    logger.info(f"Drained and deleted instance {inst.name}")
                except Exception as e:
                    logger.error(f"Failed to delete drained instance {inst.name}: {e}")

    async def _republish_model_events(self, model_ids: Set[int]):
        if not model_ids:
            return
        async with async_session() as session:
            for model_id in model_ids:
                model = await Model.one_by_id(session, model_id)
                if model is None:
                    continue
                copied = Model.model_validate(model.model_dump())
                await event_bus.publish(
                    Model.__name__.lower(),
                    Event(
                        type=EventType.UPDATED,
                        data=copied,
                        changed_fields={
                            "replicas": (model.replicas, model.replicas),
                        },
                    ),
                )
