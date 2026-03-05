#!/usr/bin/env bash

set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  exec ".venv/bin/python" -m src.lab.cli "$@"
fi

if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  exec "${PYTHON_BIN}" -m src.lab.cli "$@"
fi

for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    exec "${candidate}" -m src.lab.cli "$@"
  fi
done

echo "No supported Python interpreter found." >&2
exit 1
