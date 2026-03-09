import os

os.environ["MUJOCO_GL"] = "egl"
import random
import time
import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import gymnasium
import jax
import jax.numpy as jnp
from flax.core import FrozenDict


from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import ReplayBuffer
from utils.evaluation import (
    evaluate,
    evaluate_parallel,
    flatten,
    extract_success_from_eval_info,
)
from utils.flax_utils import restore_agent, save_agent
from utils import log_utils
from utils.log_utils import BENCHMARK_OGBENCH, BENCHMARK_D4RL, wandb_log
from utils.video_utils import get_wandb_video

from hyperparams.hyperparams import override_non_cli_args_with_default


# To get access to the "eval" function in the yamls
OmegaConf.register_new_resolver("eval", lambda s: eval(s, globals()), replace=True)


@hydra.main(config_path="config", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main training and evaluation loop.
    Decorated by Hydra, which parses the config and command-line arguments.
    """

    if cfg.seed is None:
        cfg.seed = np.random.randint(2**15)
    benchmark, processed_env_name = log_utils.process_env_name(cfg.env_name)
    agent_and_env, date_and_seed, exp_name = log_utils.get_exp_name(
        cfg, processed_env_name
    )
    override_non_cli_args_with_default(cfg, processed_env_name)
    if cfg.log_on_wandb:
        log_utils.wandb_init(cfg, exp_name, processed_env_name)
    if cfg.log_locally or cfg.save_interval != 0 or cfg.save_last_checkpoint:
        log_utils.create_save_dir(cfg, agent_and_env, date_and_seed)

    # global config params
    is_ogbench = benchmark == BENCHMARK_OGBENCH
    is_d4rl_ant = benchmark == BENCHMARK_D4RL and "antmaze" in processed_env_name
    ogbench_3_evals = is_ogbench and cfg.is_final
    is_visual = "visual" in cfg.env_name
    if is_visual:
        cfg.dataset_on_gpu = False
        cfg.n_eval_envs = 1
        frame_stack = cfg.pixel_based.frame_stack
    else:
        frame_stack = None
    off_steps = cfg.offline_steps
    on_steps = cfg.online_steps
    batch_size = cfg.agent.batch_size
    if not cfg.log_on_wandb and not cfg.log_locally:
        cfg.log_metrics = False

    # agent config
    cfg_agent = OmegaConf.to_container(cfg.agent, resolve=True, throw_on_missing=True)
    cfg_agent["log_metrics"] = cfg.log_metrics
    is_rebrac = cfg_agent["agent_name"] == "rebrac"
    # if is_rebrac:
    # actor_freq = cfg_agent["actor_freq"]

    # Set up datasets.
    (env, eval_env_creator, train_dataset, val_dataset) = make_env_and_datasets(
        benchmark, cfg.env_name, frame_stack=frame_stack
    )
    if cfg.video_episodes > 0:
        assert benchmark == BENCHMARK_OGBENCH, (
            "Rendering is currently only supported for OGBench environments."
        )
    if cfg.online_steps > 0:
        assert not is_visual, (
            "Online fine-tuning is currently not supported for visual environments."
        )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=cfg.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(cfg.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset

    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            if is_rebrac:
                dataset.return_next_actions = True
            # For OGBench's pixel-based envs: set p_aug and frame_stack
            if is_visual:
                dataset.p_aug = cfg.pixel_based.p_aug
                dataset.frame_stack = cfg.pixel_based.frame_stack
                if cfg_agent.get("encoder") is None:
                    cfg_agent["encoder"] = cfg.pixel_based.encoder

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[cfg_agent["agent_name"]]
    agent = agent_class.create(
        cfg.seed,
        example_batch["observations"],
        example_batch["actions"],
        cfg_agent,
    )
    has_a_switch_to_online = hasattr(agent, "switch_config_to_online")

    # == Configure functions used to evaluate ==
    components_to_eval = [("Actor", "sample_actions", [])]
    if not hasattr(agent, "sample_flow_actions"):
        cfg.eval_flow_policy = False
    if cfg.eval_flow_policy:
        components_to_eval.append(("Flow", "sample_flow_actions", []))
    components_to_eval = tuple(components_to_eval)  # Not mutable

    eval_kwargs = {
        "num_eval_episodes": cfg.eval_episodes,
        "num_video_episodes": cfg.video_episodes,
        "video_frame_skip": cfg.video_frame_skip,
    }
    if cfg.n_eval_envs > 1 and cfg.eval_episodes % cfg.n_eval_envs != 0:
        print(
            "Warning: cfg.eval_episodes is not a multiple of cfg.n_eval_envs, "
            "hence we can't use evaluate_parallel."
        )
        cfg.n_eval_envs = 1
    if cfg.n_eval_envs > 1:
        assert cfg.eval_episodes % cfg.n_eval_envs == 0, (
            "ERROR: cfg.eval_episodes must be a multiple of cfg.n_eval_envs"
        )
        eval_function = evaluate_parallel
        if benchmark != BENCHMARK_D4RL:
            gym_kwargs = {"autoreset_mode": gymnasium.vector.AutoresetMode.DISABLED}
        else:
            gym_kwargs = {}
        eval_kwargs["envs"] = gymnasium.vector.AsyncVectorEnv(
            [eval_env_creator] * cfg.n_eval_envs, **gym_kwargs
        )
        eval_kwargs["n_eval_envs"] = cfg.n_eval_envs
        if cfg.video_episodes > 0:
            eval_kwargs["video_env"] = eval_env_creator()
    else:
        eval_function = evaluate
        eval_kwargs["env"] = eval_env_creator()

    # Restore agent.
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Train agent.
    if cfg.log_locally:
        success_header = [
            "success_" + comp_name for (comp_name, _, _) in components_to_eval
        ]
        if ogbench_3_evals:
            success_header.extend(
                ["final_" + comp_name for (comp_name, _, _) in components_to_eval]
            )
        success_logger = log_utils.CsvLogger(
            os.path.join(cfg.save_dir, "success.csv"), header=success_header
        )
        if cfg.log_metrics:
            train_logger = log_utils.CsvLogger(os.path.join(cfg.save_dir, "train.csv"))
            eval_logger = log_utils.CsvLogger(os.path.join(cfg.save_dir, "eval.csv"))
    first_time = time.time()

    done = True
    expl_metrics = dict()
    success_info = dict()
    online_rng = jax.random.PRNGKey(cfg.seed)

    # Put the buffer on GPU
    if cfg.dataset_on_gpu:
        if is_rebrac:
            assert cfg["n_steps_fused"] % cfg_agent["actor_freq"] == 0, (
                "For ReBRAC, to fuse the operations, for simplicity we impose "
                "'n_steps_fused' to be a multiple of 'actor_freq'."
            )

        dataset_size = train_dataset.size
        train_dataset_gpu = dict()
        for key, cpu_array in train_dataset.items():
            train_dataset_gpu[key] = jnp.asarray(cpu_array)
        if train_dataset.return_next_actions:
            act = train_dataset_gpu["actions"]
            train_dataset_gpu["next_actions"] = jnp.concatenate(
                [act[1:], act[-1:]], axis=0
            )
        if on_steps == 0:
            del train_dataset

    print(f"==> Agent: {cfg_agent['agent_name']}")
    print(f"==> Env: {processed_env_name}")
    print(f"==> dataset on {'gpu' if cfg.dataset_on_gpu else 'cpu'}")
    print(cfg_agent)

    cfg = FrozenDict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    for i in tqdm.tqdm(
        range(1, off_steps + on_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        # ============= Training step =============
        if i <= off_steps and cfg["dataset_on_gpu"]:
            # Special case offline RL with dataset on GPU
            if i % cfg["n_steps_fused"]:
                continue
            agent, update_info = agent.multi_sample_and_update(
                cfg["n_steps_fused"], train_dataset_gpu, dataset_size, batch_size
            )

        else:  # Dataset on CPU
            if i <= off_steps:  # Offline
                batch = train_dataset.sample(batch_size)
            else:
                if i == off_steps + 1 and has_a_switch_to_online:
                    agent = agent.switch_config_to_online()
                # ==== Online play to collect ====
                online_rng, key = jax.random.split(online_rng)
                # Play with the env
                if done:
                    ob, _ = env.reset()
                action = agent.sample_actions(observations=ob, temperature=1, seed=key)
                action = np.array(action)
                next_ob, reward, terminated, truncated, info = env.step(action.copy())
                done = terminated or truncated

                # Save the transition
                if is_d4rl_ant:
                    reward = reward - 1.0
                replay_buffer.add_transition(
                    dict(
                        observations=ob,
                        actions=action,
                        rewards=reward,
                        terminals=float(done),
                        masks=1.0 - terminated,
                        next_observations=next_ob,
                    )
                )
                ob = next_ob
                if done:
                    expl_metrics = {
                        f"exploration/{k}": np.mean(v) for k, v in flatten(info).items()
                    }

                # Sample the batch
                if cfg["balanced_sampling"]:
                    dataset_batch = train_dataset.sample(batch_size // 2)
                    replay_batch = replay_buffer.sample(batch_size // 2)
                    batch = {
                        k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0)
                        for k in dataset_batch
                    }
                else:
                    batch = replay_buffer.sample(batch_size)
                # ==== End of online play ====

            if is_rebrac:
                agent, update_info = agent.update(
                    batch, full_update=(i % cfg_agent["actor_freq"] == 0)
                )
            else:
                agent, update_info = agent.update(batch)
        # ============= End of training step =============

        is_last_iters = (
            i == off_steps
            or i == off_steps + on_steps
            or (
                ogbench_3_evals and (i == off_steps - 100000 or i == off_steps - 200000)
            )
        )

        # === log training and validation metrics ===
        if cfg["log_metrics"] and (
            is_last_iters or (cfg["log_interval"] != 0 and i % cfg["log_interval"] == 0)
        ):
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(cfg_agent["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update(
                    {f"validation/{k}": v for k, v in val_info.items()}
                )
            train_metrics["time/total_time"] = time.time() - first_time
            train_metrics.update(expl_metrics)
            if cfg["log_on_wandb"]:
                wandb_log("train", train_metrics, i)
            if cfg["log_locally"]:
                train_logger.log(train_metrics, i)

        # === Evaluate the agent ===
        if is_last_iters or (
            cfg["eval_interval"] != 0 and i % cfg["eval_interval"] == 0
        ):
            all_eval_info = dict()
            success_info = dict()
            for comp_name, comp_func_name, comp_final_results in components_to_eval:
                eval_info, trajs, renders = eval_function(
                    agent_fn=getattr(agent, comp_func_name), **eval_kwargs
                )
                success = extract_success_from_eval_info(benchmark, eval_info)
                success_info["success_" + comp_name] = success
                print(f"Success rate of {comp_name}: ", success)
                if cfg["log_on_wandb"]:
                    if cfg["video_episodes"] > 0:
                        video = get_wandb_video(renders)
                        wandb_log(None, {f"video_{comp_name}": video}, i)
                if cfg["log_metrics"]:
                    all_eval_info.update(
                        {f"{comp_name}/{k}": v for k, v in eval_info.items()}
                    )

                if ogbench_3_evals:
                    comp_final_results.append(success)
                    if i == off_steps:
                        mean_v = np.mean(comp_final_results[-3:])
                        print(
                            f"Final success rate (averaged over last 3 evals) of {comp_name}: ",
                            mean_v,
                        )
                        success_info["final_" + comp_name] = mean_v

            if cfg["log_metrics"]:
                if cfg["log_on_wandb"]:
                    wandb_log("eval", all_eval_info, i)
                if cfg["log_locally"]:
                    eval_logger.log(all_eval_info, i)
            if cfg["log_on_wandb"]:
                wandb_log("success", success_info, i)
            if cfg["log_locally"]:
                success_logger.log(success_info, i)

        # === Save the agent ===
        if (is_last_iters and cfg["save_last_checkpoint"]) or (
            cfg["save_interval"] != 0 and i % cfg["save_interval"] == 0
        ):
            save_agent(agent, cfg["save_dir"], i)

    if cfg["log_locally"]:
        success_logger.close()
        if cfg["log_metrics"]:
            train_logger.close()
            eval_logger.close()

    if "final_Actor" in success_info:
        return success_info["final_Actor"]
    elif "success_Actor" in success_info:
        return success_info["success_Actor"]
    else:
        return None


if __name__ == "__main__":
    main()
