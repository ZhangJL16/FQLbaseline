import os
import json
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import wandb

BENCHMARK_OGBENCH = "OGBench"
BENCHMARK_MINARI = "Minari"
BENCHMARK_D4RL = "D4RL"


def process_env_name(env_name: str) -> tuple[str,str]:
    # OGBench
    if "singletask" in env_name:
        benchmark = BENCHMARK_OGBENCH
    # Minari
    elif "mujoco" in env_name or "D4RL" in env_name:
        assert "/" in env_name, "Did not recognize the benchmark name"
        parts = env_name.split("/")
        benchmark = BENCHMARK_MINARI
        env_name = "_".join(parts[1:])
    # D4RL
    else:
        benchmark = BENCHMARK_D4RL
    processed_name = benchmark+"_Env_"+env_name.replace("-","_")
    return benchmark,processed_name


def get_exp_name(
    cfg: DictConfig,
    processed_env_name: str,
):
    parts = [
        cfg.agent.agent_name,
        processed_env_name,
        f"date_{datetime.now().strftime("%Y%m%d_%H%M%S")}",
        f"seed_{int(cfg.seed)}",
    ]
    agent_and_env = "_".join(parts[:2])
    date_and_seed = "_".join(parts[-2:])
    full_name = "_".join([agent_and_env,date_and_seed])
    return agent_and_env, date_and_seed, full_name



def create_save_dir(cfg,*paths):
    cfg.save_dir = save_dir = os.path.join(cfg.save_dir, *paths)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)



def wandb_init(
    cfg: DictConfig,
    exp_name: str,
    processed_env_name: str,
):
    if wandb.run is not None:
        wandb.finish()
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    to_sci = lambda x : f"f{float(x):.1e}"
    cfg_agent = config["agent"]
    for key in list(cfg_agent.keys()):
        if "alpha" in key or "temperature" in key:
            if cfg_agent[key] is not None:
                cfg_agent[key] = to_sci(cfg_agent[key]) 
                # to fix an issue with floats in wandb convert them to str
    config["env"] = processed_env_name
    cfg_wandb:dict = config["wandb"] 
    # -> Likely contains: project, entity, group, mode
    wandb.init(
        name=exp_name,
        tags=[cfg_wandb.get("group"),processed_env_name],
        config=config,
        **cfg_wandb
    )
    # TO REMOVE
    # wandb.define_metric("global_step")
    # wandb.define_metric("eval/*", step_metric="global_step")



def wandb_log(prefix:str, data:dict, step:int):
    if prefix is not None:
        data = dict(
            (f"{prefix}/{key}",value)
            for key,value in data.items()
            if value is not None
        )
    if data:
        wandb.log(data,step=step)


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path, header=None):
        self.path = path
        self.header = header
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
            self.file.write(','.join(self.header) + '\n')
        filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
        self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
