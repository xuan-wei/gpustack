"""Configurable environment variables for GPUStack."""

import os

# Database configuration
DB_ECHO = os.getenv("GPUSTACK_DB_ECHO", "false").lower() == "true"
DB_POOL_SIZE = int(os.getenv("GPUSTACK_DB_POOL_SIZE", 30))
DB_MAX_OVERFLOW = int(os.getenv("GPUSTACK_DB_MAX_OVERFLOW", 20))
DB_POOL_TIMEOUT = int(os.getenv("GPUSTACK_DB_POOL_TIMEOUT", 30))

# Proxy configuration
PROXY_TIMEOUT = int(os.getenv("GPUSTACK_PROXY_TIMEOUT_SECONDS", 1800))
# Connect timeout: how long aiohttp waits to *establish* the upstream TCP
# connection. Default 5s — when the upstream container is killed/wedged, this
# fails fast and lets the handler release its slot, instead of sitting on
# PROXY_TIMEOUT for half an hour.
PROXY_CONNECT_TIMEOUT = int(os.getenv("GPUSTACK_PROXY_CONNECT_TIMEOUT_SECONDS", 5))
# Socket read timeout: max gap between bytes during the response stream.
# Default 60s — normal streaming chat emits at least one token per second, so a
# 60s silence reliably means upstream died mid-stream. Long enough not to kill
# legitimate slow first-token responses on big models.
PROXY_SOCK_READ_TIMEOUT = int(os.getenv("GPUSTACK_PROXY_SOCK_READ_TIMEOUT_SECONDS", 60))
PROXY_UPSTREAM_IDLE_TIMEOUT = int(
    os.getenv("GPUSTACK_PROXY_UPSTREAM_IDLE_TIMEOUT_SECONDS", 3)
)

# Auto-load cold-start gate (per-model in-flight waiter cap).
# When a model is at replicas=0 and a request arrives, openai.py long-polls up
# to AUTO_LOAD_TIMEOUT (5 min) waiting for it to come up. Without this cap, a
# burst of requests against a cold model puts hundreds of handlers in that
# long-poll. Default 300 waiters per model — chosen to comfortably absorb
# normal-scale concurrent demand against a single model that's coming up,
# while still capping a truly pathological burst (1k+). Excess requests return
# 503 Retry-After. Set to 0 to disable the gate entirely.
COLD_START_MAX_WAITERS = int(os.getenv("GPUSTACK_COLD_START_MAX_WAITERS", 300))

# HTTP client TCP connector configuration
TCP_CONNECTOR_LIMIT = int(os.getenv("GPUSTACK_TCP_CONNECTOR_LIMIT", 1000))

# JWT Expiration
JWT_TOKEN_EXPIRE_MINUTES = int(os.getenv("GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES", 120))

# Higress plugin configuration
HIGRESS_EXT_AUTH_TIMEOUT_MS = int(
    os.getenv("GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS", 30000)
)

# Server Cache
SERVER_CACHE_TTL_SECONDS = int(os.getenv("GPUSTACK_SERVER_CACHE_TTL_SECONDS", 600))
SERVER_CACHE_LOCKS_MAX_SIZE = int(
    os.getenv("GPUSTACK_SERVER_CACHE_LOCKS_MAX_SIZE", 10000)
)

# Worker configuration
WORKER_HEARTBEAT_INTERVAL = int(
    os.getenv("GPUSTACK_WORKER_HEARTBEAT_INTERVAL", 30)
)  # in seconds
WORKER_STATUS_SYNC_INTERVAL = int(
    os.getenv("GPUSTACK_WORKER_STATUS_SYNC_INTERVAL", 30)
)  # in seconds
WORKER_HEARTBEAT_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD", 150)
)  # 2.5 minutes in seconds
WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD", 300)
)  # 5 minutes in seconds
WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD", 300)
)  # 5 minutes in seconds
# Worker unreachable check mode: auto, enabled, disabled
# - auto: automatically disable check when worker count > 50 (default)
# - enabled: always perform unreachable check
# - disabled: never perform unreachable check
WORKER_UNREACHABLE_CHECK_MODE = os.getenv(
    "GPUSTACK_WORKER_UNREACHABLE_CHECK_MODE", "auto"
).lower()

# Model instance configuration
MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD = int(
    os.getenv("GPUSTACK_MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD", 300)
)  # 5 minutes in seconds
MODEL_INSTANCE_HEALTH_CHECK_INTERVAL = int(
    os.getenv("GPUSTACK_MODEL_INSTANCE_HEALTH_CHECK_INTERVAL", 3)
)
DISABLE_OS_FILELOCK = os.getenv("GPUSTACK_DISABLE_OS_FILELOCK", "false").lower() in [
    "true",
    "1",
]
# Add debug logs for slow worker status collection, default to 3 minutes
WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS = float(
    os.getenv("GPUSTACK_WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS", 180)
)

# Model evaluation cache configuration
MODEL_EVALUATION_CACHE_MAX_SIZE = int(
    os.getenv("GPUSTACK_MODEL_EVALUATION_CACHE_MAX_SIZE", 1000)
)
MODEL_EVALUATION_CACHE_TTL = int(os.getenv("GPUSTACK_MODEL_EVALUATION_CACHE_TTL", 3600))

# Scheduler configuration (server-side)
SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE", 100)
)
SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE", 5)
)
# Scale-down scoring weights (relative, normalized in score chain)
SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE", 100)
)
SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE", 10)
)
SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE = float(
    os.getenv("GPUSTACK_SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE", 1)
)

MIGRATION_DATA_DIR = os.getenv("GPUSTACK_MIGRATION_DATA_DIR", None)

DATA_MIGRATION = os.getenv("GPUSTACK_DATA_MIGRATION", "false").lower() == "true"

GATEWAY_PORT_CHECK_INTERVAL = int(
    os.getenv("GPUSTACK_GATEWAY_PORT_CHECK_INTERVAL", 2)
)  # in seconds

GATEWAY_PORT_CHECK_RETRY_COUNT = int(
    os.getenv("GPUSTACK_GATEWAY_PORT_CHECK_RETRY_COUNT", 300)
)  # number of retries

GATEWAY_MIRROR_INGRESS_NAME = os.getenv(
    "GPUSTACK_GATEWAY_MIRROR_INGRESS_NAME", "gpustack"
)

GATEWAY_EXTERNAL_METRICS_URL = os.getenv("GPUSTACK_GATEWAY_EXTERNAL_METRICS_URL", None)

GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES = [
    ct.strip()
    for ct in os.getenv(
        "GPUSTACK_GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES",
        "application/json,text/event-stream",
    ).split(",")
    if ct.strip()
]

DEFAULT_CLUSTER_KUBERNETES = (
    os.getenv("GPUSTACK_DEFAULT_CLUSTER_KUBERNETES", "false").lower() == "true"
)

# Benchmark configuration
BENCHMARK_DATASET_SHAREGPT_PATH = os.getenv(
    "GPUSTACK_BENCHMARK_DATASET_SHAREGPT_PATH",
    "/workspace/benchmark-runner/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json",
)
BENCHMARK_REQUEST_TIMEOUT = int(
    os.getenv("GPUSTACK_BENCHMARK_REQUEST_TIMEOUT", 3600)  # 1 hour
)  # in seconds
