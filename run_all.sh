#!/usr/bin/env bash
set -euo pipefail

SUITE_OUT_DIR="${1:-artifacts/weight_space_detection_suite}"
E2E_OUT_DIR="${2:-artifacts/e2e_covert_channel_demo}"

if command -v uv >/dev/null 2>&1; then
  uv run ruff check .
  uv run pytest -q
  uv run python examples/e2e_covert_channel_demo.py --out-dir "${E2E_OUT_DIR}"
  uv run wso-audit suite --out-dir "${SUITE_OUT_DIR}"
else
  PYTHON_BIN="${PYTHON:-python3}"
  "${PYTHON_BIN}" -m pip install -e ".[dev,dashboard]"
  "${PYTHON_BIN}" -m ruff check .
  "${PYTHON_BIN}" -m pytest -q
  "${PYTHON_BIN}" examples/e2e_covert_channel_demo.py --out-dir "${E2E_OUT_DIR}"
  wso-audit suite --out-dir "${SUITE_OUT_DIR}"
fi

printf 'E2E covert-channel demo artifacts written to %s\n' "${E2E_OUT_DIR}"
printf 'Detection suite artifacts written to %s\n' "${SUITE_OUT_DIR}"
