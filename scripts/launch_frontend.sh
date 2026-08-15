#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
frontend_dir="$(cd -- "${script_dir}/../frontend" && pwd)"

if ! command -v bun >/dev/null 2>&1; then
  echo "Bun is required. Install it from https://bun.sh and try again." >&2
  exit 1
fi

cd "${frontend_dir}"
bun install --frozen-lockfile
exec bun run dev -- "$@"
