import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.models import Model
from gpustack.server.db import get_engine

logger = logging.getLogger(__name__)


class AutoUnloadTask:
    """Task to automatically unload idle model instances."""

    def __init__(self, interval: int = 60):
        """Initialize the auto unload task.

        Args:
            interval: Interval in seconds to check for idle models.
        """
        self.interval = interval
        self.task: Optional[asyncio.Task] = None
        self._engine = get_engine()

    async def start(self):
        """Start the auto unload task."""
        if self.task and not self.task.done():
            logger.warning("Auto unload task is already running")
            return

        self.task = asyncio.create_task(self._run())
        logger.info("Auto unload task started")

    async def _run(self):
        """Run the auto unload task periodically."""
        while True:
            try:
                await self._check_and_unload_idle_models()
            except Exception as e:
                logger.error(f"Error in auto unload task: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    async def _check_and_unload_idle_models(self):
        """Check for idle model instances and unload them if needed."""
        async with AsyncSession(self._engine) as session:
            try:
                # Get all models directly using Model.all method
                models = await Model.all(session)

                # Filter models with replicas > 0
                active_models = [model for model in models if model.replicas > 0]

                now = datetime.now(timezone.utc)
                for model in active_models:
                    # Ensure auto_unload attribute exists and is enabled
                    if not hasattr(model, 'auto_unload') or not model.auto_unload:
                        continue

                    # Skip if last_request_time is not set
                    if (
                        not hasattr(model, 'last_request_time')
                        or not model.last_request_time
                    ):
                        continue

                    # Calculate idle time in seconds
                    idle_seconds = (now - model.last_request_time).total_seconds()

                    # Check if model should be unloaded
                    if idle_seconds >= model.auto_unload_timeout * 60:
                        logger.info(
                            f"Auto unloading model {model.name} "
                            f"after {idle_seconds:.0f} seconds of inactivity"
                        )

                        try:
                            # Set replicas to 0 to unload all instances
                            model.replicas = 0
                            await model.update(session)

                            logger.info(
                                f"Set replicas to 0 for model {model.name} to trigger unload"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to auto unload model {model.name}: {e}",
                                exc_info=True,
                            )
            except Exception as e:
                logger.error(f"Error checking for idle models: {e}", exc_info=True)
