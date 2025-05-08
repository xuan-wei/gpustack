from typing import List, Optional
import httpx
import logging
import asyncio
import random
import time
from fastapi import APIRouter, Query, Request, Response, status
from openai.types import Model as OAIModel
from openai.pagination import SyncPage
from sqlmodel import col, or_, select
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
from gpustack.http_proxy.load_balancer import LoadBalancer
from gpustack.routes.models import build_pg_category_condition
from gpustack.schemas.models import (
    CategoryEnum,
    Model,
    ModelInstanceStateEnum,
    ModelInstance,
)
from gpustack.server.db import get_session_context
from gpustack.server.deps import SessionDep

from gpustack.server.services import ModelInstanceService, ModelService, WorkerService

logger = logging.getLogger(__name__)

load_balancer = LoadBalancer()

aliasable_router = APIRouter()

# 本地缓存每个 model_id 的上次更新时间
last_update_time = {}
last_update_lock = asyncio.Lock()
UPDATE_INTERVAL = 5  # 秒


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
        async with get_session_context() as session:
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
    statement = select(Model).where(Model.ready_replicas > 0)

    if embedding_only is not None:
        categories.append(CategoryEnum.EMBEDDING)

    if image_only is not None:
        categories.append(CategoryEnum.IMAGE)

    if reranker is not None:
        categories.append(CategoryEnum.RERANKER)

    if speech_to_text is not None:
        categories.append(CategoryEnum.SPEECH_TO_TEXT)

    if text_to_speech is not None:
        categories.append(CategoryEnum.TEXT_TO_SPEECH)

    if categories:
        if session.bind.dialect.name == "sqlite":
            statement = statement.where(
                or_(
                    *[
                        (
                            col(Model.categories) == []
                            if category == ""
                            else col(Model.categories).contains(category)
                        )
                        for category in categories
                    ]
                )
            )
        else:  # For PostgreSQL
            category_conditions = [
                build_pg_category_condition(category) for category in categories
            ]
            statement = statement.where(or_(*category_conditions))

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
    # Use short session to keep transactions short
    # First session - parse request body and get model
    async with get_session_context() as session:
        model, stream, body_json, form_data, form_files = await parse_request_body(
            request, session
        )

        if not model:
            raise NotFoundException(
                message="Model not found",
                is_openai_exception=True,
            )

        request.state.model = model
        request.state.stream = stream

    # Second session - update last request time（加本地缓存，减少写频率）
    await update_model_last_request_time(model)

    # Third session - get instance and worker
    instance = None
    worker = None

    async with get_session_context() as session:
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

    logger.debug(f"proxying to {url}, instance port: {instance.port}")

    try:
        if stream:
            return await handle_streaming_request(
                request, url, body_json, form_data, form_files, extra_headers, model
            )
        else:
            return await handle_standard_request(
                request, url, body_json, form_data, form_files, extra_headers, model
            )
    except httpx.TimeoutException as e:
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
    form_files = None
    content_type = request.headers.get("content-type", "application/json").lower()

    if request.method == "GET":
        model_name = request.query_params.get("model")
    elif content_type.startswith("multipart/form-data"):
        form_data, form_files, model_name = await parse_form_data(request)
    else:
        body_json, model_name, stream = await parse_json_body(request)

    if not model_name:
        raise BadRequestException(
            message="Missing 'model' field",
            is_openai_exception=True,
        )

    if form_data and form_data.get("stream", False):
        # stream may be set in form data, e.g., image edits.
        stream = True

    model = await ModelService(session).get_by_name(model_name)
    return model, stream, body_json, form_data, form_files


async def parse_form_data(request: Request):
    try:
        form = await request.form()
        model_name = form.get("model")

        form_files = []
        form_data = {}
        for key, value in form.items():
            if isinstance(value, UploadFile):
                form_files.append(
                    (key, (value.filename, await value.read(), value.content_type))
                )
            else:
                form_data[key] = value

        return form_data, form_files, model_name
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


async def handle_streaming_request(  # noqa: C901
    request: Request,
    url: str,
    body_json: Optional[dict],
    form_data: Optional[dict],
    form_files: Optional[list],
    extra_headers: Optional[dict] = None,
    model: Optional[Model] = None,
):
    timeout = 300
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    if body_json and "stream_options" not in body_json:
        # Defaults to include usage.
        # TODO Record usage without client awareness.
        body_json["stream_options"] = {"include_usage": True}

    async def stream_generator():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=body_json if body_json else None,
                    data=form_data if form_data else None,
                    files=form_files if form_files else None,
                    timeout=timeout,
                ) as resp:
                    if resp.status_code >= 400:
                        yield await resp.aread(), resp.headers, resp.status_code

                    chunk = ""
                    async for line in resp.aiter_lines():
                        if line != "":
                            chunk = line + "\n"
                        else:
                            chunk += "\n"
                            yield chunk, resp.headers, resp.status_code
        except httpx.ConnectError as e:
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

    # 更新模型最后请求时间
    await update_model_last_request_time(model)

    return StreamingResponseWithStatusCode(
        stream_generator(), media_type="text/event-stream"
    )


async def handle_standard_request(
    request: Request,
    url: str,
    body_json: Optional[dict],
    form_data: Optional[dict],
    form_files: Optional[list],
    extra_headers: Optional[dict] = None,
    model: Optional[Model] = None,
):
    timeout = 600
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            json=body_json if body_json else None,
            data=form_data if form_data else None,
            files=form_files if form_files else None,
            timeout=timeout,
        )

        # 更新模型最后请求时间
        await update_model_last_request_time(model)

        return Response(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            content=resp.content,
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
    }


async def get_running_instance(session: AsyncSession, model_id: int):  # noqa: C901
    """
    Get a running instance for the model, or start instances if needed.
    Returns a running instance for the model.

    Simplified logic: If running instances < auto_load_replicas, set replicas = auto_load_replicas
    and wait for instances to start, checking every 2 seconds with a 2-minute timeout.
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
    # auto_load is enabled
    if len(running_instances) > model.auto_load_replicas:
        # If we have more running instances than the model.auto_load_replicas, we need to stop some instances
        # if there are error instances, will stop it BTW

        if model.replicas > model.auto_load_replicas:
            logger.info(
                f"Stopping {model.replicas - model.auto_load_replicas} instances for model {model.name}"
            )
            model.replicas = model.auto_load_replicas
            await model.update(session)

        # Sort running instances by creation time
        running_instances.sort(key=lambda x: x.created_at, reverse=True)
        return await load_balancer.get_instance(
            running_instances[: model.auto_load_replicas]
        )

    elif len(running_instances) == model.auto_load_replicas:
        logger.debug(
            f"Number of running instance equals to auto_load_replicas for model {model.name}"
        )

        if model.replicas > model.auto_load_replicas:
            logger.info(
                f"Stopping {model.replicas - model.auto_load_replicas} instances for model {model.name}"
            )
            model.replicas = model.auto_load_replicas
            await model.update(session)

        return await load_balancer.get_instance(running_instances)

    else:
        # Set replicas to auto_load_replicas + len(error_instances) to trigger instance creation
        model_instances = await ModelInstance.all_by_field(
            session=session, field="model_id", value=model_id
        )

        error_instances = [
            inst
            for inst in model_instances
            if inst.state == ModelInstanceStateEnum.ERROR
        ]
        target = len(error_instances) + model.auto_load_replicas
        if model.replicas < target:
            logger.info(
                f"Setting replicas from {model.replicas} to {target} for model {model.name} to trigger instance creation"
            )
            model.replicas = target
            await model.update(session)

        # Wait for instances to be ready with timeout
        wait_start_time = asyncio.get_event_loop().time()
        timeout = 120  # 2 minutes timeout

        while True:
            # Re-check running instances
            running_instances = await ModelInstanceService(
                session
            ).get_running_instances_no_cache(model_id)

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
