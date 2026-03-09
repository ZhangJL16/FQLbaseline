from pathlib import Path
import csv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

def get_agent_cli_args():
    overrides: list[str] = HydraConfig.get().overrides.task
    agent_overrides = []
    overrode_offline_steps = False
    for ovr in overrides: 
        if ovr.startswith("agent."):
            agent_overrides.append(ovr[6:].split('=')[0])
        elif ovr.startswith("offline_steps="):
            overrode_offline_steps = True
    return agent_overrides, overrode_offline_steps


def override_non_cli_args_with_default(
        cfg: DictConfig, 
        processed_env_name: str
    ):
    agent_name = cfg.agent.agent_name
    if agent_name == "gfp" and cfg.agent.guidance_fn == "advantage":
        agent_name = "gfp-advantage"
    params_file = Path(__file__).parent / f"{agent_name}_params.csv"
    if not params_file.exists():
        return
    agent_overrides,overrode_offline_steps = get_agent_cli_args()
    with open(params_file,mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        corresponding_rows = [
            row for row in reader
            if (row["benchmark"] in processed_env_name 
                and row["env_name"] in processed_env_name)
        ]
        n_rows = len(corresponding_rows)
        if n_rows > 0:
            print(f"Found {n_rows} matching hyper-param sets")
            corresponding_rows.sort(key=lambda row: len(row["env_name"]))
            row = corresponding_rows[-1]
            for key,val in row.items():
                if key == "offline_steps":
                    if not overrode_offline_steps:
                        cfg.offline_steps = int(val)
                if hasattr(cfg.agent,key) and key not in agent_overrides:
                    setattr(cfg.agent,key,type(getattr(cfg.agent,key))(val))
            print(f"Recovered hyper parameters from {params_file}->{row['env_name']}")
            return
