import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy import func, select

from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


class AutoMetricsTask:
    def __init__(self, interval: int = 30, collector=None):
        self.interval = interval
        self.task: Optional[asyncio.Task] = None
        self._prev_request_counts: Dict[int, int] = {}
        self._collector = collector
        self._prev_cumulative: Dict[int, Dict] = {}
        self._prev_timestamp: float = 0

    async def start(self):
        if self.task and not self.task.done():
            logger.warning("Auto metrics task is already running")
            return

        self.task = asyncio.create_task(self._run())
        logger.info("Auto metrics task started")

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            logger.info("Auto metrics task stopped")

    async def _run(self):
        while True:
            try:
                await self._update_all_model_metrics()
            except Exception as e:
                logger.error(f"Error in auto metrics task: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    async def _query_request_counts(self, session) -> Dict[int, int]:
        stmt = (
            select(
                ModelUsage.model_id,
                func.sum(ModelUsage.request_count).label("total"),
            )
            .where(ModelUsage.model_id.isnot(None))
            .group_by(ModelUsage.model_id)
        )
        result = await session.execute(stmt)
        return {row.model_id: int(row.total) for row in result.all()}

    def _aggregate_gateway_metrics(self) -> Dict[int, Dict]:
        if self._collector is None:
            return {}
        totals: Dict[int, Dict] = {}
        for metrics in self._collector.cached_dict.values():
            mid = metrics.model_id
            if mid is None:
                continue
            if mid not in totals:
                totals[mid] = {"request_count": 0, "llm_service_duration": 0}
            totals[mid]["request_count"] += metrics.request_count
            totals[mid]["llm_service_duration"] += metrics.llm_service_duration
        return totals

    async def _update_all_model_metrics(self):
        now = time.monotonic()
        gateway_totals = self._aggregate_gateway_metrics()

        async with async_session() as session:
            request_counts = await self._query_request_counts(session)
            models = await Model.all(session)

            for model in models:
                needs_update = False

                if model.ready_replicas > 0:
                    current_count = request_counts.get(model.id, 0)
                    prev_count = self._prev_request_counts.get(model.id, 0)

                    if current_count > prev_count:
                        model.last_request_time = datetime.now(timezone.utc)
                        needs_update = True
                        logger.debug(
                            f"Model {model.name}: new requests detected "
                            f"({prev_count} -> {current_count}), "
                            f"updated last_request_time"
                        )

                    self._prev_request_counts[model.id] = current_count

                if (
                    gateway_totals
                    and self._prev_timestamp > 0
                    and model.id in gateway_totals
                ):
                    cur = gateway_totals[model.id]
                    prev = self._prev_cumulative.get(model.id)
                    if prev is not None:
                        delta_time = now - self._prev_timestamp
                        if delta_time > 0:
                            delta_requests = (
                                cur["request_count"] - prev["request_count"]
                            )
                            delta_duration = (
                                cur["llm_service_duration"]
                                - prev["llm_service_duration"]
                            )
                            if delta_requests >= 0 and delta_duration >= 0:
                                delta_minutes = delta_time / 60.0
                                new_rate = (
                                    delta_requests / delta_minutes
                                    if delta_minutes > 0
                                    else 0
                                )
                                if new_rate != model.avg_request_rate:
                                    model.avg_request_rate = round(new_rate, 2)
                                    needs_update = True

                                if delta_requests > 0 and model.ready_replicas > 0:
                                    avg_duration_ms = delta_duration / delta_requests
                                    if avg_duration_ms > 0:
                                        process_rate = model.ready_replicas * (
                                            60000.0 / avg_duration_ms
                                        )
                                        if process_rate != model.avg_process_rate:
                                            model.avg_process_rate = round(
                                                process_rate, 2
                                            )
                                            needs_update = True

                if needs_update:
                    await model.update(session)

        self._prev_cumulative = gateway_totals
        self._prev_timestamp = now
