import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.models import Model
from gpustack.server.db import get_engine
from gpustack.server.metrics_manager import metrics_manager
from gpustack.server.services import ModelInstanceService

logger = logging.getLogger(__name__)


class AutoScalingTask:
    """Task to automatically adjust model replicas based on demand."""

    def __init__(self, interval: int = 60):
        """Initialize the auto scaling task.

        Args:
            interval: Interval in seconds to check for scaling needs.
        """
        self.interval = interval
        self.task: Optional[asyncio.Task] = None
        self._engine = get_engine()

    async def start(self):
        """Start the auto scaling task."""
        if self.task and not self.task.done():
            logger.warning("Auto scaling task is already running")
            return

        self.task = asyncio.create_task(self._run())
        logger.info("Auto scaling task started")

    async def stop(self):
        """Stop the auto scaling task."""
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            logger.info("Auto scaling task stopped")

    async def _run(self):
        """Run the auto scaling task periodically."""
        while True:
            try:
                await self._check_and_scale_models()
            except Exception as e:
                logger.error(f"Error in auto scaling task: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    async def _check_and_scale_models(self):
        """Check for models that need scaling and adjust their replicas."""
        async with AsyncSession(self._engine) as session:
            try:
                # Get all models with ready replicas > 0 and auto_adjust_replicas enabled
                models = await Model.all(session)
                auto_scale_models = [
                    model
                    for model in models
                    if (
                        model.ready_replicas > 0
                        and hasattr(model, 'auto_adjust_replicas')
                        and model.auto_adjust_replicas
                    )
                ]

                for model in auto_scale_models:
                    try:
                        await self._scale_model_if_needed(session, model)
                    except Exception as e:
                        logger.error(
                            f"Failed to scale model {model.name}: {e}",
                            exc_info=True,
                        )

            except Exception as e:
                logger.error(f"Error checking models for scaling: {e}", exc_info=True)

    async def _scale_model_if_needed(self, session: AsyncSession, model: Model):
        """Scale a model if needed based on current metrics."""
        # Get running instances
        running_instances = await ModelInstanceService(
            session
        ).get_running_instances_no_cache(model.id)
        current_replicas = len(running_instances)

        # Calculate desired replicas
        desired_replicas = await self._calculate_desired_replicas(
            model, current_replicas
        )

        # Create scaling message
        scale_message = f"{model.avg_request_rate:.1f},{model.avg_process_rate:.1f},{model.replicas},{desired_replicas}"

        # Apply scaling if needed
        if desired_replicas != model.replicas:
            logger.info(
                f"Auto-scaling model {model.name} from {model.replicas} to {desired_replicas} "
                f"replicas (demand={model.avg_request_rate:.1f} req/min, "
                f"supply={model.avg_process_rate:.1f} req/min, "
                f"utilization={f'{(model.avg_request_rate / model.avg_process_rate * 100):.2f}' if model.avg_process_rate > 0 else 'N/A'}"
            )
        model.replicas = desired_replicas
        model.last_scale_time = datetime.now(timezone.utc)
        model.last_scale_message = scale_message
        await model.update(session)

    async def _calculate_desired_replicas(
        self, model: Model, current_replicas: int
    ) -> int:
        """Calculate desired replicas based on demand-supply relationship."""
        # For first request or no replicas, set desired replicas to auto_load_replicas // 2
        if metrics_manager.get_queue_length(model.id) == 0 or model.replicas == 0:
            desired_replicas = getattr(model, 'auto_load_replicas', 1) // 2
            if desired_replicas == 0:
                desired_replicas = 1
            logger.debug(
                f"First request for model {model.name}, setting desired_replicas to {desired_replicas}"
            )
            return desired_replicas

        # Use current metrics for scaling decision
        if current_replicas == 0 or model.avg_process_rate == 0:
            return getattr(model, 'auto_load_replicas', 1) // 2 or 1

        # Supply: total processing capacity (requests per minute) - directly from avg_process_rate
        supply = model.avg_process_rate

        # Demand: current request rate (requests per minute)
        demand = model.avg_request_rate

        # Calculate desired replicas based on demand-supply relationship
        desired_replicas = current_replicas

        # Scale up if demand > 120% of supply
        if demand > supply * 1.2:
            desired_replicas = min(
                current_replicas + 1,
                getattr(model, 'auto_load_replicas', current_replicas + 1),
            )
        # Scale down if demand < 80% of what supply would be after scaling down by 1 replica
        elif (
            current_replicas > 1
            and demand < supply * (current_replicas - 1) / current_replicas * 0.8
        ):
            desired_replicas = max(current_replicas - 1, 1)

        return desired_replicas
