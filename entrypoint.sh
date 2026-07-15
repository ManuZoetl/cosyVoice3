#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${COSYVOICE_ROOT:-/opt/CosyVoice}:${COSYVOICE_ROOT:-/opt/CosyVoice}/third_party/Matcha-TTS:${PYTHONPATH:-}"

exec uvicorn server:app \
  --app-dir /app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --log-level "${LOG_LEVEL:-info}"
