#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python"
fi
LOG_DIR="$ROOT_DIR/logs"

SEEDS=(0 1 2)
ENVS=(
  humanoidmaze-medium-navigate-singletask-v0
  humanoidmaze-large-navigate-singletask-v0
  antsoccer-arena-navigate-singletask-v0
)

mkdir -p "$LOG_DIR"

for ENV_NAME in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "===== ${ENV_NAME} | seed=${SEED} | fql =====" | tee -a "$LOG_DIR/scsfql_vs_fql_3env_3seed.log"
    (
      cd "$ROOT_DIR"
      WANDB_MODE="${WANDB_MODE:-online}" "$PYTHON_BIN" main.py seed="$SEED" agent=fql env_name="$ENV_NAME"
    ) 2>&1 | tee -a "$LOG_DIR/scsfql_vs_fql_3env_3seed.log"

    echo "===== ${ENV_NAME} | seed=${SEED} | scsfql =====" | tee -a "$LOG_DIR/scsfql_vs_fql_3env_3seed.log"
    (
      cd "$ROOT_DIR"
      WANDB_MODE="${WANDB_MODE:-online}" "$PYTHON_BIN" main.py seed="$SEED" agent=scsfql env_name="$ENV_NAME"
    ) 2>&1 | tee -a "$LOG_DIR/scsfql_vs_fql_3env_3seed.log"
  done
done
