import asyncio
import logging
from typing import Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.models import Model
from gpustack.server.db import get_engine
from gpustack.server.metrics_manager import metrics_manager
from gpustack.server.services import ModelInstanceService

logger = logging.getLogger(__name__)


class AutoMetricsTask:
    """Task to automatically update model metrics."""

    def __init__(self, interval: int = 30):
        """Initialize the auto metrics task.

        Args:
            interval: Interval in seconds to update metrics.
        """
        self.interval = interval
        self.task: Optional[asyncio.Task] = None
        self._engine = get_engine()

    async def start(self):
        """Start the auto metrics task."""
        if self.task and not self.task.done():
            logger.warning("Auto metrics task is already running")
            return

        self.task = asyncio.create_task(self._run())
        logger.info("Auto metrics task started")

    async def stop(self):
        """Stop the auto metrics task."""
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            logger.info("Auto metrics task stopped")

    async def _run(self):
        """Run the auto metrics task periodically."""
        while True:
            try:
                await self._update_all_model_metrics()
            except Exception as e:
                logger.error(f"Error in auto metrics task: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    async def _update_all_model_metrics(self):  # noqa: C901
        """Update metrics for models that have ready replicas and recent requests."""
        async with AsyncSession(self._engine) as session:
            try:
                # Get all models and immediately cache their attributes to avoid session detachment issues
                models = await Model.all(session)

                # Cache all necessary model attributes while still in session context
                model_data = []
                for model in models:
                    try:
                        # Access and cache all attributes we need while model is attached to session
                        model_info = {
                            'model': model,
                            'id': model.id,
                            'name': model.name,
                            'ready_replicas': model.ready_replicas,
                            'avg_request_rate': getattr(model, 'avg_request_rate', 0.0)
                            or 0.0,
                            'avg_process_rate': getattr(model, 'avg_process_rate', 0.0)
                            or 0.0,
                        }
                        model_data.append(model_info)
                    except Exception as e:
                        # If we can't access attributes, create minimal info with fallbacks
                        logger.warning(f"Failed to access model attributes: {e}")
                        model_info = {
                            'model': model,
                            'id': getattr(model, 'id', None),
                            'name': f"model_{getattr(model, 'id', 'unknown')}",
                            'ready_replicas': getattr(model, 'ready_replicas', 0),
                            'avg_request_rate': 0.0,
                            'avg_process_rate': 0.0,
                        }
                        model_data.append(model_info)

                # Split models into active and inactive based on cached data
                active_models = [
                    info for info in model_data if info['ready_replicas'] > 0
                ]
                inactive_models = [
                    info for info in model_data if info['ready_replicas'] == 0
                ]

                # Process active models (ready_replicas > 0)
                for model_info in active_models:
                    model = model_info['model']
                    model_name = model_info['name']
                    try:
                        # Get running instances to determine current replicas
                        running_instances = await ModelInstanceService(
                            session
                        ).get_running_instances_no_cache(model_info['id'])
                        current_replicas = len(running_instances)

                        # Calculate new metrics
                        avg_request_rate, avg_process_rate = (
                            metrics_manager.calculate_metrics(model, current_replicas)
                        )

                        # Only update if there are changes (use cached values for comparison)
                        if (
                            model_info['avg_request_rate'] != avg_request_rate
                            or model_info['avg_process_rate'] != avg_process_rate
                        ):
                            # Update model metrics in database
                            model.avg_request_rate = avg_request_rate
                            model.avg_process_rate = avg_process_rate
                            await model.update(session)

                            logger.debug(
                                f"Updated metrics for model {model_name}: "
                                f"avg_request_rate={avg_request_rate:.1f} req/min, "
                                f"avg_process_rate={avg_process_rate:.1f} req/min"
                            )

                    except Exception as e:
                        logger.error(
                            f"Failed to update metrics for model {model_name}: {e}",
                            exc_info=True,
                        )

                # Process inactive models (ready_replicas == 0)
                for model_info in inactive_models:
                    model = model_info['model']
                    model_name = model_info['name']
                    try:
                        # Check if metrics are not zero, reset them to zero (use cached values)
                        current_request_rate = model_info['avg_request_rate']
                        current_process_rate = model_info['avg_process_rate']

                        if current_request_rate != 0.0 or current_process_rate != 0.0:
                            model.avg_request_rate = 0.0
                            model.avg_process_rate = 0.0
                            await model.update(session)

                            logger.debug(
                                f"Reset metrics to zero for inactive model {model_name}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Failed to reset metrics for model {model_name}: {e}",
                            exc_info=True,
                        )

            except Exception as e:
                logger.error(f"Error updating model metrics: {e}", exc_info=True)
