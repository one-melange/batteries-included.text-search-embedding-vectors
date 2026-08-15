#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
frontend_dir="${repo_dir}/frontend"
backend_pid=""
frontend_pid=""

cleanup() {
  set +e
  for pid in "${frontend_pid}" "${backend_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null
    fi
  done
  [[ -n "${frontend_pid}" ]] && wait "${frontend_pid}" 2>/dev/null
  [[ -n "${backend_pid}" ]] && wait "${backend_pid}" 2>/dev/null
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for command in uv bun; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "${command} is required to launch the app." >&2
    exit 1
  fi
done

cd "${repo_dir}"
uv sync --frozen

cd "${frontend_dir}"
bun install --frozen-lockfile

cd "${repo_dir}"
uv run uvicorn packages.vector_search.src.document_preparation.api:app --reload &
backend_pid=$!

cd "${frontend_dir}"
bun run dev -- "$@" &
frontend_pid=$!

while kill -0 "${backend_pid}" 2>/dev/null && kill -0 "${frontend_pid}" 2>/dev/null; do
  sleep 1
done

if ! kill -0 "${backend_pid}" 2>/dev/null; then
  wait "${backend_pid}"
else
  wait "${frontend_pid}"
fi
