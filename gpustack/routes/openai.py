import asyncio
from typing import AsyncGenerator, List, Optional, Tuple
import aiohttp
import logging
import random
import time
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
from gpustack.config.envs import PROXY_TIMEOUT
from gpustack.http_proxy.load_balancer import LoadBalancer
from gpustack.routes.models import build_category_conditions
from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    Model,
    ModelInstanceStateEnum,
    ModelInstance,
)
from gpustack.server.db import get_engine
from gpustack.server.deps import SessionDep
from gpustack.server.metrics_manager import metrics_manager

from gpustack.server.services import ModelInstanceService, ModelService, WorkerService

logger = logging.getLogger(__name__)

load_balancer = LoadBalancer()

aliasable_router = APIRouter()

# 本地缓存每个 model_id 的上次更新时间
last_update_time = {}
last_update_lock = asyncio.Lock()
UPDATE_INTERVAL = 5  # 秒

CHECK_SCALING_INTERVAL = 60  # 秒
UPDATE_REQUEST_RATE_INTERVAL = 10  # 秒


# 更新模型的最后请求时间
async def update_model_last_request_time(model):
    """更新模型的最后请求时间，使用本地缓存减少数据库写操作"""
    should_update = False
    now = time.time()

    async with last_update_lock:
        last = last_update_time.get(model.id, 0)
        if now - last > UPDATE_INTERVAL:
            last_update_time[model.id] = now
            should_update = True

    if should_update:
        async with AsyncSession(get_engine()) as session:
            try:
                model_service = ModelService(session)
                await model_service.update_last_request_time(model.id)
            except Exception as e:
                logger.warning(f"Failed to update last_request_time: {e}")


@aliasable_router.post("/chat/completions")
async def chat_completions(request: Request):
    return await proxy_request_by_model(request, "chat/completions")


@aliasable_router.post("/completions")
async def completions(request: Request):
    return await proxy_request_by_model(request, "completions")


@aliasable_router.post("/embeddings")
async def embeddings(request: Request):
    return await proxy_request_by_model(request, "embeddings")


@aliasable_router.post("/images/generations")
async def images_generations(request: Request):
    return await proxy_request_by_model(request, "images/generations")


@aliasable_router.post("/images/edits")
async def images_edits(request: Request):
    return await proxy_request_by_model(request, "images/edits")


@aliasable_router.post("/audio/speech")
async def audio_speech(request: Request):
    return await proxy_request_by_model(request, "audio/speech")


@aliasable_router.post("/audio/transcriptions")
async def audio_transcriptions(request: Request):
    return await proxy_request_by_model(request, "audio/transcriptions")


router = APIRouter()
router.include_router(aliasable_router)


@router.get("/models")
async def list_models(
    session: SessionDep,
    embedding_only: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    image_only: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    reranker: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    speech_to_text: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    text_to_speech: Optional[bool] = Query(
        None,
        deprecated=True,
        description="This parameter is deprecated and will be removed in a future version.",
    ),
    categories: List[str] = Query(
        [],
        description="Model categories to filter by.",
    ),
    with_meta: Optional[bool] = Query(
        None,
        description="Include model meta information.",
    ),
):
    all_categories = set(categories)
    if embedding_only:
        all_categories.add(CategoryEnum.EMBEDDING.value)
    if image_only:
        all_categories.add(CategoryEnum.IMAGE.value)
    if reranker:
        all_categories.add(CategoryEnum.RERANKER.value)
    if speech_to_text:
        all_categories.add(CategoryEnum.SPEECH_TO_TEXT.value)
    if text_to_speech:
        all_categories.add(CategoryEnum.TEXT_TO_SPEECH.value)
    all_categories = list(all_categories)

    statement = select(Model).where(Model.ready_replicas > 0)

    if all_categories:
        conditions = build_category_conditions(session, all_categories)
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


async def proxy_request_by_model(request: Request, endpoint: str):  # noqa: C901
    """
    Proxy the request to the model instance that is running the model specified in the
    request body.
    """
    # Record request start time
    request_start_time = time.time()

    # Use short session to keep transactions short
    # First session - parse request body and get model
    async with AsyncSession(get_engine()) as session:
        model, stream, body_json, form_data = await parse_request_body(request, session)

        if not model:
            raise NotFoundException(
                message="Model not found",
                is_openai_exception=True,
            )

        request.state.model = model
        request.state.stream = stream

        mutate_request(request, body_json, form_data)

    # Record request start for metrics calculation
    metrics_manager.record_request_start(model.id, request_start_time)

    # Second session - update last request time（加本地缓存，减少写频率）
    await update_model_last_request_time(model)

    # Third session - get instance and worker
    instance = None
    worker = None

    async with AsyncSession(get_engine()) as session:
        try:
            instance = await get_running_instance(session, model.id)
            worker = await WorkerService(session).get_by_id(instance.worker_id)

            # Validate instance and worker properties
            if not instance or not worker:
                raise InternalServerErrorException(
                    message=f"Failed to get valid worker or instance for model {model.name}",
                    is_openai_exception=True,
                )
        except Exception as e:
            logger.error(f"Failed to get running instance: {e}")
            raise

    url = f"http://{instance.worker_ip}:{worker.port}/proxy/v1/{endpoint}"
    token = request.app.state.server_config.token
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
                model,
                request_start_time,
            )
        else:
            return await handle_standard_request(
                request,
                url,
                body_json,
                form_data,
                extra_headers,
                model,
                request_start_time,
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


async def parse_request_body(request: Request, session: SessionDep):
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

    model = await ModelService(session).get_by_name(model_name)
    return model, stream, body_json, form_data


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
    model: Optional[Model] = None,
    request_start_time: float = None,
):
    timeout = aiohttp.ClientTimeout(total=300)
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    if body_json and "stream_options" not in body_json:
        # Defaults to include usage.
        # TODO Record usage without client awareness.
        body_json["stream_options"] = {"include_usage": True}

    async def stream_generator():
        try:
            http_client: aiohttp.ClientSession = request.app.state.http_client
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

                # Record request completion
                if model:
                    metrics_manager.record_request_completion(
                        model.id, request_start_time, time.time()
                    )
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
        finally:
            # Record request completion even on error
            if model:
                metrics_manager.record_request_completion(
                    model.id, request_start_time, time.time()
                )

    # 更新模型最后请求时间
    await update_model_last_request_time(model)

    return StreamingResponseWithStatusCode(
        stream_generator(), media_type="text/event-stream"
    )


async def handle_standard_request(
    request: Request,
    url: str,
    body_json: Optional[dict],
    form_data: Optional[aiohttp.FormData],
    extra_headers: Optional[dict] = None,
    model: Optional[Model] = None,
    request_start_time: float = None,
):
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    try:
        http_client: aiohttp.ClientSession = request.app.state.http_client
        timeout = aiohttp.ClientTimeout(total=PROXY_TIMEOUT)
        async with http_client.request(
            method=request.method,
            url=url,
            headers=headers,
            json=body_json if body_json else None,
            data=form_data,
            timeout=timeout,
        ) as response:
            content = await response.read()
            # Record request completion
            if model:
                metrics_manager.record_request_completion(
                    model.id, request_start_time, time.time()
                )

            # 更新模型最后请求时间
            await update_model_last_request_time(model)

            return Response(
                status_code=response.status,
                headers=dict(response.headers),
                content=content,
            )
    except Exception as e:
        # Record request completion even on error
        if model:
            metrics_manager.record_request_completion(
                model.id, request_start_time, time.time()
            )
        raise e


def filter_headers(headers):
    return {
        key: value
        for key, value in headers.items()
        if key.lower() != "content-length"
        and key.lower() != "host"
        and key.lower() != "content-type"
        and key.lower() != "transfer-encoding"
        and key.lower() != "authorization"
    }


async def get_running_instance(session: AsyncSession, model_id: int):  # noqa: C901
    """
    Get a running instance for the model, or start instances if needed.
    Returns a running instance for the model.
    """
    # First check for already running instances
    running_instances = await ModelInstanceService(
        session
    ).get_running_instances_no_cache(model_id)

    model = await Model.one_by_id(session, model_id)
    logger.debug(
        f"Found {len(running_instances)} running instances for model {model.name}"
    )

    if not model.auto_load:
        if len(running_instances) > 0:
            return await load_balancer.get_instance(running_instances)
        else:
            raise ServiceUnavailableException(
                message=f"Auto-load is disabled for model {model.name}. Please start the model manually.",
                is_openai_exception=True,
            )

    # For auto_load enabled models
    if model.auto_adjust_replicas:  # just wait for the periodic auto_scaling
        desired_replicas = (
            model.replicas if model.replicas > 0 else model.auto_load_replicas // 2
        )
        if desired_replicas == 0:
            desired_replicas = 1
    else:
        desired_replicas = model.auto_load_replicas

    # Return existing running instance or handle auto-load
    return await _handle_auto_load(session, model, running_instances, desired_replicas)


async def _handle_auto_load(
    session: AsyncSession,
    model: Model,
    running_instances: List[ModelInstance],
    desired_replicas: int,
) -> ModelInstance:
    """Handle auto-loading logic for model instances using desired_replicas."""
    # auto_load is enabled
    if len(running_instances) > desired_replicas:
        # If we have more running instances than desired_replicas, we need to stop some instances
        # if there are error instances, will stop it BTW

        if model.replicas > desired_replicas:
            logger.info(
                f"Stopping {model.replicas - desired_replicas} instances for model {model.name}"
            )
            model.replicas = desired_replicas
            await model.update(session)

        # Sort running instances by creation time
        running_instances.sort(key=lambda x: x.created_at, reverse=True)
        return await load_balancer.get_instance(running_instances[:desired_replicas])

    elif len(running_instances) == desired_replicas:
        logger.debug(
            f"Number of running instance equals to desired_replicas for model {model.name}"
        )

        if model.replicas > desired_replicas:
            logger.info(
                f"Stopping {model.replicas - desired_replicas} instances for model {model.name}"
            )
            model.replicas = desired_replicas
            await model.update(session)

        return await load_balancer.get_instance(running_instances)

    else:
        # Set replicas to desired_replicas + len(error_instances) to trigger instance creation
        model_instances = await ModelInstance.all_by_field(
            session=session, field="model_id", value=model.id
        )

        error_instances = [
            inst
            for inst in model_instances
            if inst.state == ModelInstanceStateEnum.ERROR
        ]
        target = len(error_instances) + desired_replicas
        if model.replicas < target:
            logger.info(
                f"Setting replicas from {model.replicas} to {target} for model {model.name} to trigger instance creation"
            )
            model.replicas = target
            await model.update(session)

        # Wait for instances to be ready with timeout
        wait_start_time = asyncio.get_event_loop().time()
        timeout = 120

        while True:
            # Re-check running instances
            running_instances = await ModelInstanceService(
                session
            ).get_running_instances_no_cache(model.id)

            if running_instances:
                logger.debug(
                    f"Found {len(running_instances)} running instances for model {model.name}"
                )
                # Validate instances
                valid_instances = [
                    instance
                    for instance in running_instances
                    if instance
                    and instance.worker_ip
                    and instance.port
                    and instance.worker_id
                ]

                if valid_instances:
                    logger.debug(
                        f"Found {len(valid_instances)} valid running instances for model {model.name}"
                    )
                    return await load_balancer.get_instance(valid_instances)
                else:
                    logger.debug(
                        f"No valid instances found for model {model.name}, will retry"
                    )

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - wait_start_time
            if elapsed > timeout:
                raise ServiceUnavailableException(
                    message=f"Timeout after waiting for 2 minutes. No instances ready for model {model.name}.",
                    is_openai_exception=True,
                )

            # Wait before checking again
            logger.debug(
                f"Waiting for instances to start for model {model.name}, elapsed time: {elapsed:.1f}s"
            )
            await asyncio.sleep(random.uniform(2, 5))


def mutate_request(
    request: Request, body_json: Optional[dict], form_data: Optional[aiohttp.FormData]
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
