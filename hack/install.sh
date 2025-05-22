#!/usr/bin/env bash

# Set error handling
set -o errexit
set -o nounset
set -o pipefail

# Get the root directory and third_party directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Include the common functions
source "${ROOT_DIR}/hack/lib/init.sh"

function download_deps() {
  pip install poetry==1.8.3 pre-commit==3.7.1
  poetry install
  pre-commit install
}

function download_ui() {
  local default_tag="latest"
  local ui_path="${ROOT_DIR}/gpustack/ui"
  local tmp_ui_path="${ui_path}/tmp"
  local tag="latest"
  # local tag="${1}"

  rm -rf "${ui_path}"
  mkdir -p "${tmp_ui_path}/ui"

  gpustack::log::info "downloading ui assets"

  local download_url
  if [[ "${tag}" == "latest" ]]; then
    download_url="https://github.com/xuan-wei/gpustack-ui/releases/latest/download/latest.tar.gz"
  else
    download_url="https://github.com/xuan-wei/gpustack-ui/releases/download/${tag}/${tag}.tar.gz"
  fi

  if ! curl --retry 3 --retry-connrefused --retry-delay 3 -sSfL "${download_url}" 2>/dev/null |
    tar -xzf - --directory "${tmp_ui_path}/ui" 2>/dev/null; then

    if [[ "${tag:-}" =~ ^v([0-9]+)\.([0-9]+)(\.[0-9]+)?(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]]; then
      gpustack::log::fatal "failed to download '${tag}' ui archive"
    fi

    gpustack::log::warn "failed to download '${tag}' ui archive, fallback to '${default_tag}' ui archive"
    
    local fallback_url
    if [[ "${default_tag}" == "latest" ]]; then
      fallback_url="https://github.com/xuan-wei/gpustack-ui/releases/latest/download/latest.tar.gz"
    else
      fallback_url="https://github.com/xuan-wei/gpustack-ui/releases/download/${default_tag}/${default_tag}.tar.gz"
    fi

    if ! curl --retry 3 --retry-connrefused --retry-delay 3 -sSfL "${fallback_url}" |
      tar -xzf - --directory "${tmp_ui_path}/ui" 2>/dev/null; then
      gpustack::log::fatal "failed to download '${default_tag}' ui archive"
    fi
  fi
  cp -a "${tmp_ui_path}/ui/dist/." "${ui_path}"

  rm -rf "${tmp_ui_path}"

}

# Copy extra static files to ui including catalog icons
function copy_extra_static() {
  local extra_static_path="${ROOT_DIR}/static"
  local ui_static_path="${ROOT_DIR}/gpustack/ui/static"
  if [ -d "${extra_static_path}" ]; then
    cp -a "${extra_static_path}/." "${ui_static_path}"
  fi
}

#
# main
#

gpustack::log::info "+++ DEPENDENCIES +++"
download_deps
download_ui
copy_extra_static
gpustack::log::info "--- DEPENDENCIES ---"
