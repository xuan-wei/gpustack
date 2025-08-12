import time
import logging
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, Tuple
from gpustack.schemas.models import Model, BackendEnum

logger = logging.getLogger(__name__)


class ThreadSafeRequestQueue:
    """Thread-safe request queue for storing recent requests."""

    def __init__(self):
        self.queue = deque(maxlen=1000)
        self.lock = Lock()

    def append(self, item):
        with self.lock:
            self.queue.append(item)

    def get_recent_requests(self, window_start):
        with self.lock:
            recent_requests_completed = []
            recent_requests_all = []
            for r in self.queue:
                if r[1] and r[1] >= window_start:  # completed requests
                    recent_requests_completed.append(r)
                if r[0] >= window_start and r[1] is None:  # all requests
                    recent_requests_all.append(r)
            return recent_requests_completed, recent_requests_all

    def get_length(self):
        with self.lock:
            return len(self.queue)


class MetricsManager:
    """Global metrics manager for managing request queues and metrics calculation."""

    def __init__(self):
        self.request_queues: Dict[int, ThreadSafeRequestQueue] = defaultdict(
            ThreadSafeRequestQueue
        )

    def record_request_start(self, model_id: int, start_time: float):
        """Record request start time for metrics calculation."""
        self.request_queues[model_id].append((start_time, None))

    def record_request_completion(
        self, model_id: int, start_time: float, end_time: float
    ):
        """Record request completion time for metrics calculation."""
        self.request_queues[model_id].append((start_time, end_time))

    def calculate_metrics(
        self, model: Model, current_replicas: int, window_seconds: int = 300
    ) -> Tuple[float, float]:
        """
        Calculate avg_request_rate and avg_process_rate for a model.

        Args:
            model_id: Model ID
            current_replicas: Current number of replicas
            window_seconds: Time window for calculating metrics (default: 2 minutes)

        Returns:
            Tuple of (avg_request_rate, avg_process_rate)
        """
        model_id = model.id
        queue = self.request_queues.get(model_id, ThreadSafeRequestQueue())

        # Calculate metrics from requests in the past window_seconds
        now = time.time()
        window_start = now - window_seconds
        recent_requests_completed, recent_requests_all = queue.get_recent_requests(
            window_start
        )

        if not recent_requests_all:
            return 0.0, 0.0

        # Get the actual time span from first request to last request (all requests)
        first_request_time = recent_requests_all[0][0]
        last_request_time = recent_requests_all[-1][0]
        actual_time_span_seconds = last_request_time - first_request_time

        # Ensure minimum time span to avoid division by very small numbers
        if actual_time_span_seconds < 1.0:
            actual_time_span_seconds = 1.0

        actual_time_span_minutes = actual_time_span_seconds / 60.0

        # Calculate average request rate from ALL requests (requests per minute)
        avg_request_rate = len(recent_requests_all) / actual_time_span_minutes

        # Calculate average process rate from COMPLETED requests only
        if not recent_requests_completed:
            return avg_request_rate, 0.0

        # Calculate average request duration from recent_requests_completed
        avg_finish_time = sum(r[1] - r[0] for r in recent_requests_completed) / len(
            recent_requests_completed
        )

        # Determine concurrent processing capacity based on backend type
        if model.backend == BackendEnum.LLAMA_BOX:
            # For llama-box, get parallel count from backend parameters
            parallels = self._extract_parallel_from_backend_parameters(
                model.backend_parameters
            )
            max_concurrent = parallels * current_replicas
        else:
            # For other backends, estimate from queue data
            last_scale_time = model.last_scale_time.timestamp()
            recent_requests_completed_after_last_scale_time = [
                r for r in recent_requests_completed if r[1] > last_scale_time
            ]
            max_concurrent = self._estimate_max_concurrent_requests(
                recent_requests_completed_after_last_scale_time
            )

        # Calculate processing capacity based on concurrency and duration
        if avg_finish_time > 0 and max_concurrent > 0:
            # Total capacity = max_concurrent_requests * (requests_per_minute_per_slot)
            requests_per_minute_per_slot = 60.0 / avg_finish_time
            avg_process_rate = max_concurrent * requests_per_minute_per_slot
        else:
            # Fallback to simple calculation if we can't estimate concurrency
            avg_process_rate = (
                current_replicas * (60.0 / avg_finish_time)
                if avg_finish_time > 0
                else 0.0
            )

        # Determine concurrency source for logging
        concurrency_source = (
            "backend_params"
            if model.backend == BackendEnum.LLAMA_BOX
            else "queue_analysis"
        )

        logger.debug(
            f"Model {model_id} ({model.backend}) metrics: {len(recent_requests_all)} total requests "
            f"({len(recent_requests_completed)} completed) over {actual_time_span_minutes:.1f} minutes, "
            f"avg_duration={avg_finish_time:.1f}s, max_concurrent={max_concurrent} (from {concurrency_source}), "
            f"demand={avg_request_rate:.1f} req/min, capacity={avg_process_rate:.1f} req/min"
        )

        return avg_request_rate, avg_process_rate

    def _estimate_max_concurrent_requests(
        self, recent_requests_completed_after_last_scale_time
    ) -> int:
        """
        Estimate maximum concurrent requests by analyzing overlapping time periods.

        Args:
            recent_requests_completed_after_last_scale_time: List of (start_time, end_time) tuples

        Returns:
            Estimated maximum concurrent requests across all replicas
        """
        if not recent_requests_completed_after_last_scale_time:
            return 0

        # Create timeline events: +1 for start, -1 for end
        events = []
        for start_time, end_time in recent_requests_completed_after_last_scale_time:
            events.append((start_time, 1))  # Request starts
            events.append((end_time, -1))  # Request ends

        # Sort events by time
        events.sort()

        # Track concurrent requests and find maximum
        current_concurrent = 0
        max_concurrent = 0

        for timestamp, delta in events:
            current_concurrent += delta
            max_concurrent = max(max_concurrent, current_concurrent)

        return max_concurrent

    def _extract_parallel_from_backend_parameters(self, backend_parameters) -> int:
        """
        Extract parallel count from llama-box backend parameters.

        Args:
            backend_parameters: List of backend parameter strings

        Returns:
            Parallel count (default: 1 if not found)
        """
        if not backend_parameters:
            return 1

        for param in backend_parameters:
            if param.startswith('--parallel='):
                try:
                    parallel_value = param.split('=', 1)[1]
                    return int(parallel_value)
                except (ValueError, IndexError):
                    logger.warning(f"Invalid --parallel parameter: {param}")
                    continue

        return 1  # Default if --parallel not found

    def get_queue_length(self, model_id: int) -> int:
        """Get the length of request queue for a model (containinig both completed and all requests)."""
        queue = self.request_queues.get(model_id, ThreadSafeRequestQueue())
        return queue.get_length()


# Global metrics manager instance
metrics_manager = MetricsManager()
