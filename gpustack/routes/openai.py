from typing import List, Optional
import httpx
import logging
import asyncio

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

# Add shared locks and tasks for concurrent instance requests
model_startup_locks = {}  # model_id -> asyncio.Lock()
model_startup_tasks = {}  # model_id -> asyncio.Task


aliasable_router = APIRouter()


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

        # Update the last_request_time for auto-unload feature
        model_service = ModelService(session)
        await model_service.update_last_request_time(model.id)

        # Get an instance - this may wait for an instance to become available

        try:
            instance, instance_just_started = await get_running_instance(
                session, model.id
            )
        except Exception as e:
            logger.error(f"Failed to get running instance: {e}")
            raise

        worker = await WorkerService(session).get_by_id(instance.worker_id)
        if not worker:
            raise InternalServerErrorException(
                message=f"Worker with ID {instance.worker_id} not found",
                is_openai_exception=True,
            )

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
                request, url, body_json, form_data, form_files, extra_headers
            )
        else:
            return await handle_standard_request(
                request, url, body_json, form_data, form_files, extra_headers
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


async def handle_streaming_request(
    request: Request,
    url: str,
    body_json: Optional[dict],
    form_data: Optional[dict],
    form_files: Optional[list],
    extra_headers: Optional[dict] = None,
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
    Get a running instance for the model, or start one if none are running.
    Returns the instance and a boolean indicating if it was just started.

    Handles concurrent requests with a lock and shared task mechanism to prevent
    creating multiple instances when multiple requests arrive simultaneously.
    """

    # First check for already running instances
    running_instances = await ModelInstanceService(
        session
    ).get_running_instances_no_cache(model_id)

    model = await Model.one_by_id(session, model_id)
    if len(running_instances) != model.auto_load_replicas:
        error_instances = await ModelInstance.all_by_field(
            session=session, field="model_id", value=model_id
        )
        error_instances = [
            inst
            for inst in error_instances
            if inst.state == ModelInstanceStateEnum.ERROR
        ]

        model.replicas = model.auto_load_replicas + len(error_instances)
        await model.update(session)

    if running_instances:
        # Normal case - instance already running
        return await load_balancer.get_instance(running_instances), False

    # No running instances - need to either wait for an existing startup task
    # or create a new one, using a lock to synchronize between concurrent requests

    # Initialize lock for this model if needed
    if model_id not in model_startup_locks:
        model_startup_locks[model_id] = asyncio.Lock()

    # Check if there's an ongoing startup task we can wait for
    if model_id in model_startup_tasks and not model_startup_tasks[model_id].done():
        logger.info(f"Waiting for existing startup task for model ID {model_id}")
        try:
            return await model_startup_tasks[model_id]
        except Exception as e:
            # If the task failed, we'll start a new one below
            logger.error(f"Existing startup task for model ID {model_id} failed: {e}")

    # Acquire the lock to ensure only one request initializes a new startup task
    async with model_startup_locks[model_id]:
        # Double-check for running instances after acquiring the lock
        fresh_running = await ModelInstanceService(
            session
        ).get_running_instances_no_cache(model_id)
        if fresh_running:
            return await load_balancer.get_instance(fresh_running), False

        # Check again for an ongoing startup task (might have been created while waiting for lock)
        if model_id in model_startup_tasks and not model_startup_tasks[model_id].done():
            # Release lock and wait for the task
            pass  # Lock is released when exiting this block
        else:
            # Create a new startup task
            logger.info(f"Creating new startup task for model ID {model_id}")
            model_startup_tasks[model_id] = asyncio.create_task(
                _start_model_instance(session, model_id)
            )

    # Wait for the task result outside the lock
    try:
        return await model_startup_tasks[model_id]
    except Exception as e:
        # Clean up the task on failure so future requests can try again
        if model_id in model_startup_tasks:
            del model_startup_tasks[model_id]
        logger.error(f"Failed to start instance for model ID {model_id}: {e}")
        raise


async def _start_model_instance(session: AsyncSession, model_id: int):  # noqa: C901
    """
    Internal method to start a model instance.
    This runs as a task and is shared by concurrent requests.
    """
    # Get model info
    model = await Model.one_by_id(session, model_id)
    if not model:
        raise NotFoundException(
            message="Model not found",
            is_openai_exception=True,
        )

    # Get all instances for this model
    model_instances = await ModelInstance.all_by_field(
        session=session, field="model_id", value=model_id
    )

    # Check for instances in non-ERROR states (might be starting up)
    starting_instances = [
        inst
        for inst in model_instances
        if inst.state != ModelInstanceStateEnum.ERROR
        and inst.state != ModelInstanceStateEnum.RUNNING
    ]

    # If no starting instances, check error instances and potentially create a new one
    if not starting_instances:
        error_instances = [
            inst
            for inst in model_instances
            if inst.state == ModelInstanceStateEnum.ERROR
        ]

        if error_instances and not model.restart_on_error:
            error_messages = [
                f"{inst.name}: {inst.state_message}" for inst in error_instances
            ]
            raise ServiceUnavailableException(
                message=f"All instances for model {model.name} are in error state: {'; '.join(error_messages)}",
                is_openai_exception=True,
            )
        # Check if auto_load is enabled
        if not model.auto_load:
            raise ServiceUnavailableException(
                message=f"Auto-load is disabled for model {model.name}. Please start the model manually.",
                is_openai_exception=True,
            )

        # Create a new instance by increasing replicas
        logger.info(
            f"No running instances for model {model.name}, increasing replicas to start one"
        )
        model.replicas = model.auto_load_replicas + len(error_instances)
        await model.update(session)

        # Wait for ModelController to create the instance
        logger.info(
            f"Waiting for ModelController to create an instance for model {model.name}"
        )
        creation_timeout = 60  # seconds to wait for creation
        creation_interval = 1  # seconds between checks
        creation_time = 0

        while creation_time < creation_timeout:
            await asyncio.sleep(creation_interval)
            creation_time += creation_interval

            # Check for new instances
            fresh_instances = await ModelInstance.all_by_field(
                session=session, field="model_id", value=model_id
            )

            if len(fresh_instances) > len(model_instances):
                starting_instances = [
                    inst
                    for inst in fresh_instances
                    if inst.state != ModelInstanceStateEnum.ERROR
                    and inst.state != ModelInstanceStateEnum.RUNNING
                ]

                if starting_instances:
                    logger.info(
                        f"Found new instance {starting_instances[0].name} after {creation_time}s"
                    )
                    break

            if creation_time % 10 == 0:
                logger.info(f"Still waiting for instance creation ({creation_time}s)")

        if not starting_instances:
            raise ServiceUnavailableException(
                message=f"Timeout waiting for instance creation for model {model.name}",
                is_openai_exception=True,
            )

    # Set initial message
    instance = starting_instances[0]
    instance_id = instance.id
    instance.state_message = "Instance is starting on demand. Please wait..."
    await instance.update(session)

    # Wait for the instance to become RUNNING with unified timeout
    logger.info(
        f"Waiting for instance {instance.name} to become available, current state: {instance.state}"
    )

    max_wait_time = 300  # 5 minutes total wait time
    check_interval = 2  # Check every 2 seconds
    wait_time = 0

    # Main wait loop
    while wait_time < max_wait_time:
        # Check current state - get fresh instance each time to avoid stale state
        instance = await ModelInstance.one_by_id(session, instance_id)

        # Return immediately if running
        if instance.state == ModelInstanceStateEnum.RUNNING:
            logger.info(f"Instance {instance.name} is now running after {wait_time}s")
            return instance, True

        # Error state check
        if instance.state == ModelInstanceStateEnum.ERROR:
            raise ServiceUnavailableException(
                message=f"Instance failed to start: {instance.state_message}",
                is_openai_exception=True,
            )

        # Update progress message periodically
        if wait_time % 30 == 0:
            progress_msg = f"Instance is starting on demand. Waited {wait_time}s..."

            if (
                instance.state == ModelInstanceStateEnum.DOWNLOADING
                and instance.download_progress
            ):
                max_wait_time += (
                    300  # if downloading, extend max_wait_time by 5 minutes
                )
                progress_msg += f" Downloading: {instance.download_progress:.1f}%"

            if instance.state_message != progress_msg:
                instance.state_message = progress_msg
                await instance.update(session)

            logger.info(
                f"Still waiting for instance {instance.name}, state: {instance.state}"
            )

        # Check for other instances that might have become RUNNING
        if wait_time % 20 == 0 and wait_time > 0:  # Every 20s after first check
            running_instances = [
                inst
                for inst in await ModelInstance.all_by_field(
                    session=session, field="model_id", value=model_id
                )
                if inst.state == ModelInstanceStateEnum.RUNNING
            ]

            if running_instances:
                logger.info(f"Found another running instance for model {model.name}")
                return await load_balancer.get_instance(running_instances), True

        # Wait before next check
        await asyncio.sleep(check_interval)
        wait_time += check_interval
        session.expunge_all()

    # Timeout reached
    raise GatewayTimeoutException(
        message=f"Timeout waiting for model {model.name} instance to become available",
        is_openai_exception=True,
    )
