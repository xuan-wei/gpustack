"""
Model-specific lock file cleaner for GPUStack reset functionality.

This module provides functions to clean up lock files when user clicks
the reset/retry download button.
"""

import os
import logging
import psutil

from gpustack.schemas.models import ModelSource, SourceEnum

logger = logging.getLogger(__name__)


def get_expected_lock_paths(model_source: ModelSource, cache_dir: str) -> list[str]:
    """
    Get the expected lock file paths for a given model source.

    Args:
        model_source: The model source information
        cache_dir: The cache directory

    Returns:
        List of potential lock file paths for this model
    """
    lock_paths = []

    try:
        if (
            model_source.source == SourceEnum.HUGGING_FACE
            and model_source.huggingface_repo_id
        ):
            from modelscope.hub.utils.utils import model_id_to_group_owner_name

            group_or_owner, name = model_id_to_group_owner_name(
                model_source.huggingface_repo_id
            )
            lock_path = os.path.join(
                cache_dir, "huggingface", group_or_owner, f"{name}.lock"
            )
            lock_paths.append(lock_path)

        elif (
            model_source.source == SourceEnum.MODEL_SCOPE
            and model_source.model_scope_model_id
        ):
            from modelscope.hub.utils.utils import model_id_to_group_owner_name

            group_or_owner, name = model_id_to_group_owner_name(
                model_source.model_scope_model_id
            )
            lock_path = os.path.join(
                cache_dir, "model_scope", group_or_owner, f"{name}.lock"
            )
            lock_paths.append(lock_path)

        elif (
            model_source.source == SourceEnum.OLLAMA_LIBRARY
            and model_source.ollama_library_model_name
        ):
            import re

            sanitized_filename = re.sub(
                r"[^a-zA-Z0-9]", "_", model_source.ollama_library_model_name
            )
            lock_path = os.path.join(cache_dir, "ollama", f"{sanitized_filename}.lock")
            lock_paths.append(lock_path)

    except Exception as e:
        logger.warning(f"Error determining lock paths for model: {e}")

    return lock_paths


def clean_model_lock_files_for_reset(model_source: ModelSource, cache_dir: str) -> bool:
    """
    Clean up any stale lock files when user clicks reset/retry download.

    Args:
        model_source: The model source to clean locks for
        cache_dir: The cache directory

    Returns:
        True if any locks were cleaned, False otherwise
    """
    lock_paths = get_expected_lock_paths(model_source, cache_dir)
    cleaned_any = False

    for lock_path in lock_paths:
        if clean_specific_lock_file_for_reset(lock_path):
            cleaned_any = True

    return cleaned_any


def clean_specific_lock_file_for_reset(lock_path: str) -> bool:  # noqa: C901
    """
    Clean a specific lock file when user clicks reset/retry download.

    This is more aggressive than normal cleanup since user explicitly
    requested to retry the download.

    Args:
        lock_path: Path to the lock file to check and possibly clean

    Returns:
        True if the lock file was cleaned, False otherwise
    """
    if not os.path.exists(lock_path):
        return False

    logger.info(f"Checking lock file for reset: {lock_path}")

    try:
        # Check if the process that created the lock is still running
        try:
            with open(lock_path, 'r') as f:
                content = f.read().strip()
                if content and content.isdigit():
                    pid = int(content)
                    if psutil.pid_exists(pid):
                        try:
                            process = psutil.Process(pid)
                            cmdline = ' '.join(process.cmdline())
                            if (
                                'gpustack' in cmdline.lower()
                                or 'python' in cmdline.lower()
                            ):
                                logger.info(
                                    f"GPUStack process {pid} still running, preserving lock {lock_path}"
                                )
                                return False
                            else:
                                logger.info(
                                    f"Process {pid} is not GPUStack related, cleaning lock {lock_path}"
                                )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            logger.info(
                                f"Process {pid} no longer accessible, cleaning lock {lock_path}"
                            )
                    else:
                        logger.info(
                            f"Process {pid} no longer exists, cleaning lock {lock_path}"
                        )
                else:
                    logger.info(f"Lock file {lock_path} has no valid PID, cleaning")
        except (IOError, ValueError):
            logger.info(f"Cannot read lock file {lock_path}, cleaning")

        # Remove the lock file
        try:
            os.remove(lock_path)
            logger.info(f"Successfully removed lock file for reset: {lock_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to remove lock file {lock_path}: {e}")
            return False

    except Exception as e:
        logger.warning(f"Error checking lock file {lock_path}: {e}")
        return False


# This function is removed as we only need simple reset functionality
