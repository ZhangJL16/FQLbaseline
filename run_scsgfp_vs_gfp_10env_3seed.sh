#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python"
fi
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/gfp_vs_scsgfp_10env_3seed.log"

ENVS=(
  humanoidmaze-medium-navigate-singletask-v0
  humanoidmaze-large-navigate-singletask-v0
  antsoccer-arena-navigate-singletask-v0
  antmaze-large-navigate-singletask-v0
  antmaze-giant-navigate-singletask-v0
  cube-single-play-singletask-v0
  cube-double-play-singletask-v0
  scene-play-singletask-v0
  puzzle-3x3-play-singletask-v0
  puzzle-4x4-play-singletask-v0
)

mkdir -p "$LOG_DIR"

for ENV_NAME in "${ENVS[@]}"; do
  echo "===== ${ENV_NAME} | seeds=0,1,2 | agents=gfp,scsgfp =====" | tee -a "$LOG_FILE"
  (
    cd "$ROOT_DIR"
    WANDB_MODE="${WANDB_MODE:-online}" "$PYTHON_BIN" main.py -m \
      seed=0,1,2 \
      agent=gfp,scsgfp \
      env_name="$ENV_NAME" \
      wandb.group=scsgfp_vs_gfp_10env
  ) 2>&1 | tee -a "$LOG_FILE"
done
