#!/usr/bin/env bash
set -euo pipefail

# Telegram bot runner
# - Activates local venv if present
# - Loads .env for TELEGRAM_BOT_TOKEN and API_BASE
# - Runs the Telegram bot script

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate || true
fi

if [ -f .env ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs) || true
fi

if ! python -c 'import telegram' >/dev/null 2>&1; then
  echo "[run_tg.sh] python-telegram-bot not found in current interpreter." >&2
  echo "Install deps first: pip install -r requirements.txt" >&2
  exit 1
fi

exec python scripts/run_telegram_bot.py "$@"

