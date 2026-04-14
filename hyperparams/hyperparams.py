from pathlib import Path
import csv

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


_DEFAULT_TASK_ALIASES = {
    "OGBench_Env_antmaze_large_navigate_singletask_v0": "OGBench_Env_antmaze_large_navigate_singletask_task1_v0",
    "OGBench_Env_antmaze_giant_navigate_singletask_v0": "OGBench_Env_antmaze_giant_navigate_singletask_task1_v0",
    "OGBench_Env_humanoidmaze_medium_navigate_singletask_v0": "OGBench_Env_humanoidmaze_medium_navigate_singletask_task1_v0",
    "OGBench_Env_humanoidmaze_large_navigate_singletask_v0": "OGBench_Env_humanoidmaze_large_navigate_singletask_task1_v0",
    "OGBench_Env_antsoccer_arena_navigate_singletask_v0": "OGBench_Env_antsoccer_arena_navigate_singletask_task4_v0",
    "OGBench_Env_cube_single_play_singletask_v0": "OGBench_Env_cube_single_play_singletask_task2_v0",
    "OGBench_Env_cube_double_play_singletask_v0": "OGBench_Env_cube_double_play_singletask_task2_v0",
    "OGBench_Env_scene_play_singletask_v0": "OGBench_Env_scene_play_singletask_task2_v0",
    "OGBench_Env_puzzle_3x3_play_singletask_v0": "OGBench_Env_puzzle_3x3_play_singletask_task4_v0",
    "OGBench_Env_puzzle_4x4_play_singletask_v0": "OGBench_Env_puzzle_4x4_play_singletask_task4_v0",
}


def get_cli_overrides():
    try:
        overrides: list[str] = HydraConfig.get().overrides.task
    except ValueError:
        overrides = []
    agent_overrides = []
    pixel_overrides = []
    overrode_offline_steps = False
    overrode_online_steps = False
    for ovr in overrides:
        if ovr.startswith("agent."):
            agent_overrides.append(ovr[6:].split("=")[0])
        elif ovr.startswith("pixel_based."):
            pixel_overrides.append(ovr[12:].split("=")[0])
        elif ovr.startswith("offline_steps="):
            overrode_offline_steps = True
        elif ovr.startswith("online_steps="):
            overrode_online_steps = True
    return agent_overrides, pixel_overrides, overrode_offline_steps, overrode_online_steps


def _candidate_processed_env_names(processed_env_name: str, *, is_online: bool) -> tuple[str, ...]:
    candidates = [processed_env_name]
    if not is_online and processed_env_name in _DEFAULT_TASK_ALIASES:
        candidates.append(_DEFAULT_TASK_ALIASES[processed_env_name])
    return tuple(candidates)


def _convert_like(current_value, raw_value: str):
    if current_value is None:
        return raw_value
    if isinstance(current_value, bool):
        return raw_value.lower() in ("1", "true", "t", "yes", "y")
    return type(current_value)(raw_value)


def _apply_row_value(cfg: DictConfig, key: str, val: str, agent_overrides, pixel_overrides, overrode_offline_steps, overrode_online_steps):
    if val == "":
        return

    if key == "offline_steps":
        if not overrode_offline_steps:
            cfg.offline_steps = int(val)
        return
    if key == "online_steps":
        if not overrode_online_steps:
            cfg.online_steps = int(val)
        return
    if hasattr(cfg.agent, key) and key not in agent_overrides:
        setattr(cfg.agent, key, _convert_like(getattr(cfg.agent, key), val))
        return
    if hasattr(cfg.pixel_based, key) and key not in pixel_overrides:
        setattr(cfg.pixel_based, key, _convert_like(getattr(cfg.pixel_based, key), val))


def override_non_cli_args_with_default(
        cfg: DictConfig,
        processed_env_name: str
    ):
    agent_name = cfg.agent.agent_name
    candidate_names = [agent_name]
    if agent_name == "gfp" and cfg.agent.guidance_fn == "advantage":
        candidate_names = ["gfp-advantage", "gfp"]
    elif agent_name == "scsgfp":
        if cfg.agent.guidance_fn == "advantage":
            candidate_names = ["scsgfp-advantage", "scsgfp", "gfp-advantage", "gfp"]
        else:
            candidate_names = ["scsgfp", "gfp"]
    elif agent_name == "scsfql":
        candidate_names.append("fql")

    params_file = None
    for candidate_name in candidate_names:
        candidate_path = Path(__file__).parent / f"{candidate_name}_params.csv"
        if candidate_path.exists():
            params_file = candidate_path
            break
    if params_file is None:
        return

    target_suite = "online" if cfg.online_steps > 0 else "offline"
    env_candidates = _candidate_processed_env_names(processed_env_name, is_online=cfg.online_steps > 0)
    agent_overrides, pixel_overrides, overrode_offline_steps, overrode_online_steps = get_cli_overrides()

    with open(params_file, mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        corresponding_rows = []
        for row in reader:
            row_suite = row.get("suite", "")
            if row_suite and row_suite != target_suite:
                continue
            if not any(row["benchmark"] in env_name and row["env_name"] in env_name for env_name in env_candidates):
                continue
            corresponding_rows.append(row)

        n_rows = len(corresponding_rows)
        if n_rows == 0:
            return

        print(f"Found {n_rows} matching hyper-param sets")
        corresponding_rows.sort(key=lambda row: len(row["env_name"]))
        row = corresponding_rows[-1]
        for key, val in row.items():
            _apply_row_value(
                cfg,
                key,
                val,
                agent_overrides,
                pixel_overrides,
                overrode_offline_steps,
                overrode_online_steps,
            )
        print(f"Recovered hyper parameters from {params_file}->{row['env_name']} [{target_suite}]")
