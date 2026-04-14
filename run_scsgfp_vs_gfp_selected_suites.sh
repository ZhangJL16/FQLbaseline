#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/gfp_vs_scsgfp_selected_batch_${STAMP}.log"

# W&B: a new project will be created automatically on first upload if it does not exist.
export WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-gfp-scsgfp-selected-batch-${STAMP}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-gfp-vs-scsgfp}"

# Edit here if you want more seeds.
SEEDS=(0)

OGBENCH_BASE_ENVS=(
  cube-double-play-singletask
  antmaze-large-navigate-singletask
  humanoidmaze-medium-navigate-singletask
  antsoccer-arena-navigate-singletask
)
OGBENCH_TASK_IDS=(1 2 3 4 5)

D4RL_ENVS=(
  antmaze-umaze-v2
  door-expert-v1
)

MINARI_ENVS=(
  "mujoco/walker2d/medium-v0"
  "mujoco/halfcheetah/expert-v0"
)

sanitize_name() {
  local raw="$1"
  echo "$raw" | tr '/:' '__' | tr -c 'A-Za-z0-9_.-=' '_'
}

run_multirun() {
  local env_name="$1"
  local group_name="$2"
  local seed_csv
  seed_csv="$(IFS=,; echo "${SEEDS[*]}")"

  local cmd=(
    "$PYTHON_BIN" main.py -m
    "seed=${seed_csv}"
    "agent=gfp,scsgfp"
    "env_name=${env_name}"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.group=${group_name}"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    cmd+=("wandb.entity=${WANDB_ENTITY}")
  fi

  echo ">>> RUN ${cmd[*]}" | tee -a "$LOG_FILE"
  "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
}

prefetch_ogbench() {
  local env_name="$1"
  echo ">>> PREFETCH OGBench ${env_name}" | tee -a "$LOG_FILE"
  "$PYTHON_BIN" - "$env_name" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import sys
import ogbench

env_name = sys.argv[1]
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
train_size = len(train_dataset["actions"]) if isinstance(train_dataset, dict) else "unknown"
val_size = len(val_dataset["actions"]) if isinstance(val_dataset, dict) else "unknown"
print(f"prefetched OGBench env={env_name} train_size={train_size} val_size={val_size}")
if hasattr(env, "close"):
    env.close()
PY
}

prefetch_d4rl() {
  local env_name="$1"
  echo ">>> PREFETCH D4RL ${env_name}" | tee -a "$LOG_FILE"
  "$PYTHON_BIN" - "$env_name" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import sys
from envs.env_utils import make_env_and_datasets
from utils.log_utils import BENCHMARK_D4RL

env_name = sys.argv[1]
env, _, train_dataset, _ = make_env_and_datasets(BENCHMARK_D4RL, env_name)
print(f"prefetched D4RL env={env_name} train_size={train_dataset.size}")
if hasattr(env, "close"):
    env.close()
PY
}

prefetch_minari() {
  echo ">>> PREFETCH Minari datasets: ${MINARI_ENVS[*]}" | tee -a "$LOG_FILE"
  "$PYTHON_BIN" download_minari_datasets.py --datasets "${MINARI_ENVS[@]}" 2>&1 | tee -a "$LOG_FILE"
}

echo "===== Batch start $(date -Iseconds) =====" | tee -a "$LOG_FILE"
echo "ROOT_DIR=$ROOT_DIR" | tee -a "$LOG_FILE"
echo "PYTHON_BIN=$PYTHON_BIN" | tee -a "$LOG_FILE"
echo "WANDB_MODE=$WANDB_MODE" | tee -a "$LOG_FILE"
echo "WANDB_PROJECT=$WANDB_PROJECT" | tee -a "$LOG_FILE"
if [[ -n "$WANDB_ENTITY" ]]; then
  echo "WANDB_ENTITY=$WANDB_ENTITY" | tee -a "$LOG_FILE"
fi
echo "SEEDS=${SEEDS[*]}" | tee -a "$LOG_FILE"

echo "===== Prefetch datasets =====" | tee -a "$LOG_FILE"
for base_env in "${OGBENCH_BASE_ENVS[@]}"; do
  for task_id in "${OGBENCH_TASK_IDS[@]}"; do
    prefetch_ogbench "${base_env}-task${task_id}-v0"
  done
done
for env_name in "${D4RL_ENVS[@]}"; do
  prefetch_d4rl "$env_name"
done
prefetch_minari

echo "===== Run OGBench 5-task suites =====" | tee -a "$LOG_FILE"
for base_env in "${OGBENCH_BASE_ENVS[@]}"; do
  suite_name="$(sanitize_name "$base_env")"
  for task_id in "${OGBENCH_TASK_IDS[@]}"; do
    env_name="${base_env}-task${task_id}-v0"
    run_multirun "$env_name" "${WANDB_GROUP_PREFIX}-ogbench-${suite_name}"
  done
done

echo "===== Run D4RL =====" | tee -a "$LOG_FILE"
for env_name in "${D4RL_ENVS[@]}"; do
  run_multirun "$env_name" "${WANDB_GROUP_PREFIX}-d4rl"
done

echo "===== Run Minari =====" | tee -a "$LOG_FILE"
for env_name in "${MINARI_ENVS[@]}"; do
  run_multirun "$env_name" "${WANDB_GROUP_PREFIX}-minari"
done

echo "===== Batch done $(date -Iseconds) =====" | tee -a "$LOG_FILE"
