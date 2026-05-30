import re
import random
import asyncio
import time
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional, Tuple
import aiohttp
import logging

from fastapi import APIRouter, Query, Request, Response, status
from openai.types import Model as OAIModel
from openai.pagination import SyncPage
from sqlmodel import or_, select
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.datastructures import UploadFile

from gpustack.api.exceptions import (
    BadRequestException,
    NotFoundException,
    InternalServerErrorException,
    OpenAIAPIError,
    OpenAIAPIErrorResponse,
    ServiceUnavailableException,
    GatewayTimeoutException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack import envs
from gpustack.http_proxy.load_balancer import LoadBalancer
from gpustack.routes.model_common import build_category_conditions
from gpustack.schemas.models import (
    BackendEnum,
    Model,
)
from gpustack.schemas.model_routes import (
    ModelRoute,
    MyModel,
)
from gpustack.server.deps import SessionDep, CurrentUserDep
from gpustack.server.services import (
    ModelInstanceService,
    ModelRouteService,
    ModelService,
    WorkerService,
    UserService,
)
from gpustack.utils.network import use_proxy_env_for_url
from gpustack.utils.auto_load import (
    AUTO_LOAD_TIMEOUT,
    FAILED,
    READY,
    poll_until_ready,
    resolve_route_model_ids,
)
from gpustack.utils.cold_start_gate import (
    ColdStartCapacityExceeded,
    cold_start_slot,
)


logger = logging.getLogger(__name__)


async def _try_auto_load(session, model_name: str) -> Optional[Model]:
    model_ids = await resolve_route_model_ids(session, model_name)
    if not model_ids:
        return None

    models = await Model.all_by_fields(
        session, fields={}, extra_conditions=[Model.id.in_(model_ids)]
    )

    triggered = False
    waiting = False
    now = datetime.now(timezone.utc)
    for model in models:
        if model.replicas == 0 and model.auto_load:
            model.replicas = max(model.auto_load_replicas, 1)
            model.last_request_time = now
            await model.update(session)
            triggered = True
            logger.info(
                f"Auto-load triggered for model {model.name}, "
                f"scaling to {model.replicas} replicas"
            )
        elif model.replicas > 0 and model.ready_replicas == 0:
            # Another in-flight request already triggered auto-load and the
            # model is still warming up. Long-poll instead of returning 404.
            waiting = True

    if not triggered and not waiting:
        return None

    # Gate the long-poll wait per model. If too many handlers are already
    # waiting for this model to cold-start, fail fast with 503 so the asyncio
    # task pool / upstream connection pool isn't saturated by idle waiters.
    try:
        async with cold_start_slot(model_name):
            result = await poll_until_ready(model_ids, mark_last_request=True)
    except ColdStartCapacityExceeded as exc:
        logger.warning(str(exc))
        raise ServiceUnavailableException(
            message=(
                f"Model '{model_name}' is cold-starting and has reached its "
                f"in-flight waiter cap ({exc.cap}). Retry shortly."
            ),
            is_openai_exception=True,
        )

    if result.outcome == READY:
        return result.model
    if result.outcome == FAILED:
        raise ServiceUnavailableException(
            message=f"Model failed to load: {result.message}",
            is_openai_exception=True,
        )

    logger.warning(
        f"Auto-load timeout for route {model_name} after {AUTO_LOAD_TIMEOUT}s"
    )
    return None


load_balancer = LoadBalancer()

last_update_time = {}
last_update_lock = asyncio.Lock()
UPDATE_INTERVAL = 5


router = APIRouter()


async def update_model_last_request_time(
    session: AsyncSession,
    model: Model,
) -> None:
    should_update = False
    now = time.time()

    async with last_update_lock:
        last = last_update_time.get(model.id, 0)
        if now - last > UPDATE_INTERVAL:
            last_update_time[model.id] = now
            should_update = True

    if should_update:
        await ModelService(session).update_last_request_time(model.id)


@router.get("/models")
async def list_models(
    user: CurrentUserDep,
    session: SessionDep,
    categories: List[str] = Query(
        [],
        description="Model categories to filter by.",
    ),
    with_meta: Optional[bool] = Query(
        None,
        description="Include model meta information.",
    ),
):
    target_class = ModelRoute if user.is_admin else MyModel
    statement = select(target_class).where(target_class.ready_targets > 0)
    if target_class == MyModel:
        # Non-admin users should only see their own private models when filtering by categories.
        statement = statement.where(target_class.user_id == user.id)

    if categories:
        conditions = build_category_conditions(session, target_class, categories)
        statement = statement.where(or_(*conditions))

    models = (await session.exec(statement)).all()
    result = SyncPage[OAIModel](data=[], object="list")
    for model in models:
        result.data.append(
            OAIModel(
                id=model.name,
                object="model",
                created=int(model.created_at.timestamp()),
                owned_by="gpustack",
                meta=model.meta if with_meta else None,
            )
        )
    return result


@router.post("/completions")
@router.post("/chat/completions")
@router.post("/responses")
@router.post("/embeddings")
@router.post("/images/generations")
@router.post("/images/edits")
@router.post("/audio/speech")
@router.post("/audio/transcriptions")
async def proxy_request_by_model(
    request: Request,
    user: CurrentUserDep,
    session: SessionDep,
):
    endpoint = re.sub(r"^/(v1|v1-openai)/", "", request.url.path)
    """
    Proxy the request to the model instance that is running the model specified in the
    request body.
    """
    model_name, stream, body_json, form_data = await parse_request_body(request)
    if not await UserService(session).model_allowed_for_user(
        model_name=model_name,
        user_id=user.id,
        api_key=getattr(request.state, "api_key", None),
    ):
        raise NotFoundException(
            message="Model not found",
            is_openai_exception=True,
        )
    models: List[Model] = await ModelRouteService(
        session
    ).get_model_ids_by_model_route_name(model_name)
    if len(models) == 0:
        loaded_model = await _try_auto_load(session, model_name)
        if loaded_model is not None:
            models = [loaded_model]
            session.expire_all()
    if len(models) == 0:
        raise NotFoundException(
            message="Model not found or no running instances available",
            is_openai_exception=True,
        )
    request.state.stream = stream
    model = random.choice(models)
    request.state.model = model

    mutate_request(request, model_name, body_json, form_data)

    await update_model_last_request_time(session, model)

    instance = await get_running_instance(session, model.id)
    worker = await WorkerService(session).get_by_id(instance.worker_id)
    if not worker:
        raise InternalServerErrorException(
            message=f"Worker with ID {instance.worker_id} not found",
            is_openai_exception=True,
        )

    url = f"http://{instance.worker_ip}:{worker.port}/proxy/v1/{endpoint}"
    token = worker.token
    extra_headers = {
        "X-Target-Port": str(instance.port),
        "Authorization": f"Bearer {token}",
    }

    if model.backend == BackendEnum.ASCEND_MINDIE:
        # Connectivity to the loopback address via worker proxy does not work for Ascend MindIE.
        # Bypassing the worker proxy and directly connecting to the instance as a workaround.
        url = f"http://{instance.worker_ip}:{instance.port}/v1/{endpoint}"
        extra_headers = {}

    logger.debug(f"proxying to {url}, instance port: {instance.port}")

    try:
        if stream:
            return await handle_streaming_request(
                request,
                url,
                body_json,
                form_data,
                extra_headers,
            )
        else:
            return await handle_standard_request(
                request,
                url,
                body_json,
                form_data,
                extra_headers,
            )
    except asyncio.TimeoutError as e:
        error_message = f"Request to {url} timed out"
        if str(e):
            error_message += f": {e}"
        raise GatewayTimeoutException(
            message=error_message,
            is_openai_exception=True,
        )
    except Exception as e:
        error_message = "An unexpected error occurred"
        if str(e):
            error_message += f": {e}"
        raise ServiceUnavailableException(
            message=error_message,
            is_openai_exception=True,
        )


async def parse_request_body(request: Request):
    model_name = None
    stream = False
    body_json = None
    form_data = None
    content_type = request.headers.get("content-type", "application/json").lower()

    if request.method == "GET":
        model_name = request.query_params.get("model")
    elif content_type.startswith("multipart/form-data"):
        form_data, model_name, stream = await parse_form_data(request)
    else:
        body_json, model_name, stream = await parse_json_body(request)

    if not model_name:
        raise BadRequestException(
            message="Missing 'model' field",
            is_openai_exception=True,
        )

    return model_name, stream, body_json, form_data


async def parse_form_data(request: Request) -> Tuple[aiohttp.FormData, str, bool]:
    try:
        form = await request.form()
        model_name = form.get("model")
        stream = form.get("stream", False)

        form_data = aiohttp.FormData()
        for key, value in form.items():
            if isinstance(value, UploadFile):
                form_data.add_field(
                    key,
                    await value.read(),
                    filename=value.filename,
                    content_type=value.content_type,
                )
            else:
                form_data.add_field(key, value)

        return form_data, model_name, stream
    except Exception as e:
        raise BadRequestException(
            message=f"We could not parse the form body of your request: {e}",
            is_openai_exception=True,
        )


async def parse_json_body(request: Request):
    try:
        body_json = await request.json()
        model_name = body_json.get("model")
        stream = body_json.get("stream", False)
        return body_json, model_name, stream
    except Exception as e:
        raise BadRequestException(
            message=f"We could not parse the JSON body of your request: {e}",
            is_openai_exception=True,
        )


async def _stream_response_chunks(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[str, None]:
    """Stream the response content in chunks, processing each line."""

    chunk_size = 4096  # 4KB
    chunk_buffer = b""
    async for data in resp.content.iter_chunked(chunk_size):
        lines = (chunk_buffer + data).split(b'\n')
        # Keep the last line in the buffer if it's incomplete
        chunk_buffer = lines.pop(-1)

        for line_bytes in lines:
            if line_bytes:
                yield _process_line(line_bytes)

    if chunk_buffer:
        yield _process_line(chunk_buffer)


def _process_line(line_bytes: bytes) -> str:
    """Process a line of bytes to ensure it is properly formatted for streaming."""
    line = line_bytes.decode("utf-8").strip()
    return line + "\n\n" if line else ""


async def handle_streaming_request(
    request: Request,
    url: str,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
    extra_headers: Optional[dict] = None,
):
    timeout = aiohttp.ClientTimeout(
        total=envs.PROXY_TIMEOUT,
        connect=envs.PROXY_CONNECT_TIMEOUT,
        sock_read=envs.PROXY_SOCK_READ_TIMEOUT,
    )
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    if body_json and "stream_options" not in body_json:
        # Defaults to include usage.
        # TODO Record usage without client awareness.
        body_json["stream_options"] = {"include_usage": True}

    async def stream_generator():
        try:
            use_proxy_env = use_proxy_env_for_url(url)
            http_client: aiohttp.ClientSession = (
                request.app.state.http_client
                if use_proxy_env
                else request.app.state.http_client_no_proxy
            )
            async with http_client.request(
                method=request.method,
                url=url,
                headers=headers,
                json=body_json if body_json else None,
                data=form_data,
                timeout=timeout,
            ) as resp:
                if resp.status >= 400:
                    yield await resp.read(), resp.headers, resp.status
                    return

                async for chunk in _stream_response_chunks(resp):
                    yield chunk, resp.headers, resp.status
        except aiohttp.ClientError as e:
            error_response = OpenAIAPIErrorResponse(
                error=OpenAIAPIError(
                    message=f"Service unavailable. Please retry your requests after a brief wait. Original error: {e}",
                    code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    type="ServiceUnavailable",
                ),
            )
            yield error_response.model_dump_json(), {}, status.HTTP_503_SERVICE_UNAVAILABLE
        except Exception as e:
            error_response = OpenAIAPIErrorResponse(
                error=OpenAIAPIError(
                    message=f"Internal server error: {e}",
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    type="InternalServerError",
                ),
            )
            yield error_response.model_dump_json(), {}, status.HTTP_500_INTERNAL_SERVER_ERROR

    return StreamingResponseWithStatusCode(
        stream_generator(), media_type="text/event-stream"
    )


async def handle_standard_request(
    request: Request,
    url: str,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
    extra_headers: Optional[dict] = None,
):
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    use_proxy_env = use_proxy_env_for_url(url)
    http_client: aiohttp.ClientSession = (
        request.app.state.http_client
        if use_proxy_env
        else request.app.state.http_client_no_proxy
    )
    timeout = aiohttp.ClientTimeout(
        total=envs.PROXY_TIMEOUT,
        connect=envs.PROXY_CONNECT_TIMEOUT,
        sock_read=envs.PROXY_SOCK_READ_TIMEOUT,
    )
    async with http_client.request(
        method=request.method,
        url=url,
        headers=headers,
        json=body_json if body_json else None,
        data=form_data,
        timeout=timeout,
    ) as response:
        content = await response.read()
        return Response(
            status_code=response.status,
            headers=dict(response.headers),
            content=content,
        )


def filter_headers(headers):
    return {
        key: value
        for key, value in headers.items()
        if key.lower() != "content-length"
        and key.lower() != "host"
        and key.lower() != "content-type"
        and key.lower() != "transfer-encoding"
        and key.lower() != "authorization"
        and key.lower() != "x-gpustack-model"
    }


async def get_running_instance(session: AsyncSession, model_id: int):
    running_instances = await ModelInstanceService(session).get_running_instances(
        model_id
    )
    model = await Model.one_by_id(session, model_id)
    if model is None:
        raise ServiceUnavailableException(
            message="Model not found",
            is_openai_exception=True,
        )

    if running_instances:
        return await load_balancer.get_instance(running_instances)

    if not model.auto_load:
        raise ServiceUnavailableException(
            message=f"Auto-load is disabled for model {model.name}. Please start the model manually.",
            is_openai_exception=True,
        )

    desired_replicas = model.auto_load_replicas
    if model.auto_adjust_replicas:
        desired_replicas = (
            model.replicas if model.replicas > 0 else model.auto_load_replicas // 2
        )
        if desired_replicas == 0:
            desired_replicas = 1

    return await _handle_auto_load(session, model, desired_replicas)


async def _handle_auto_load(
    session: AsyncSession,
    model: Model,
    desired_replicas: int,
):
    if model.replicas < desired_replicas:
        model.replicas = desired_replicas
        await model.update(session)

    refreshed_instances = await ModelInstanceService(session).get_running_instances(
        model.id
    )
    if refreshed_instances:
        return await load_balancer.get_instance(refreshed_instances)

    raise ServiceUnavailableException(
        message=f"No running instances available for model {model.name}",
        is_openai_exception=True,
    )


def mutate_request(
    request: Request,
    model_name: str,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
):
    path = request.url.path
    model: Model = request.state.model
    if (
        path == "/v1/rerank"
        and body_json
        and model.env
        and model.env.get("GPUSTACK_APPLY_QWEN3_RERANKER_TEMPLATES", False)
    ):
        apply_qwen3_reranker_templates(body_json)
    if model_name != model.name:
        if body_json is not None:
            body_json["model"] = model.name
        elif form_data is not None:
            form_data.add_field("model", model.name)


def apply_qwen3_reranker_templates(body_json: dict):
    """
    Apply Qwen3 reranker templates to the request body.
    See instructions in https://huggingface.co/Qwen/Qwen3-Reranker-0.6B.
    Note: Once vLLM supports built-in template rendering for this model, this can be removed.
    """
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    query_template = "{prefix}<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n"
    document_template = "<Document>: {doc}{suffix}"
    if "query" in body_json and "documents" in body_json:
        query = body_json["query"]
        documents = body_json["documents"]
        body_json["query"] = query_template.format(prefix=prefix, query=query)
        body_json["documents"] = [
            document_template.format(doc=doc, suffix=suffix) for doc in documents
        ]
