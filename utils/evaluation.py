from collections import defaultdict
import jax
import numpy as np
from tqdm import trange

from .log_utils import BENCHMARK_OGBENCH, BENCHMARK_D4RL, BENCHMARK_MINARI


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def extract_success_from_eval_info(benchmark,eval_info:dict):
    if benchmark == BENCHMARK_OGBENCH:
        return 100*eval_info["success"]
    elif benchmark == BENCHMARK_D4RL:
        return eval_info["final_info.episode.normalized_return"]
    elif benchmark == BENCHMARK_MINARI:
        from envs.minari_utils import minari_normalized_score
        return minari_normalized_score(eval_info["acc_reward"])
    else:
        raise Exception("Unknown benchmark")



def evaluate(
    agent_fn,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    # may either be agent.sample_actions or agent.sample_flow_actions
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        acc_reward = 0.
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            acc_reward = acc_reward + reward
            info["acc_reward"] = acc_reward
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def get_kth_element(x, k):
    if isinstance(x, dict):
        return {
            key: get_kth_element(x[key], k) for key in filter_keys(x.keys())
            }
    else:
        return x[k]


def filter_keys(key_list):
    # Remove the <_name> keys added by gymnasium
    keys = []
    for key in key_list:
        if key[0] == '_' and key[1:] in key_list:
            pass
        else:
            keys.append(key)
    return keys
        

def evaluate_parallel(
    agent_fn,
    envs,
    num_eval_episodes=50,
    num_video_episodes=0,
    eval_temperature=0,
    n_eval_envs=10,
    video_env=None,
    **kwargs # past to the evaluate function for videos
):
    """Parallel version of FQL's evaluate function"""
    actor_fn_rng = supply_rng(agent_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    # may either be agent.sample_actions or agent.sample_flow_actions
    stats = defaultdict(list)
    
    iterations = num_eval_episodes // n_eval_envs
    finished = False
    counts = np.zeros(n_eval_envs, dtype=int)
    inactive = np.zeros(n_eval_envs, dtype=int)
    
    initial_seed = np.random.randint(1e6)
    rng = jax.random.PRNGKey(initial_seed)
    rng, subkey = jax.random.split(rng)
    observation, _ = envs.reset(seed=subkey[0].item())

    trajs = [defaultdict(list) for _ in range(num_eval_episodes)]

    while not finished:
        action = actor_fn_rng(observations=observation, temperature=eval_temperature)
        action = np.array(action)
        action = np.clip(action, -1, 1)

        next_observation, reward, terminated, truncated, info = envs.step(action)
        done = np.logical_or(terminated, truncated)

        for k in range(n_eval_envs):
            if not inactive[k]:
                transition = dict(
                    observation=get_kth_element(observation, k),
                    next_observation=get_kth_element(next_observation, k),
                    action=get_kth_element(action, k),
                    reward=get_kth_element(reward, k),
                    done=get_kth_element(done, k),
                    info=get_kth_element(info, k),
                )
                index = counts[k] + k * iterations
                add_to(trajs[index], transition)
        
        observation = next_observation

        if done.max():
            for k in range(n_eval_envs):
                if done[k]:
                    add_to(stats, flatten(get_kth_element(info, k)))

            rng, subkey = jax.random.split(rng)
            observation, _ = envs.reset(
                options={"reset_mask": done.flatten()}, 
                seed=subkey[0].item()
            )
            
            counts += done
            inactive = counts >= iterations
            finished = inactive.min()
    for k, v in stats.items():
        stats[k] = np.mean(v)

    if num_video_episodes > 0:
        _, _, renders = evaluate(
            agent_fn=agent_fn, 
            env=video_env,
            num_eval_episodes=0,
            num_video_episodes=num_video_episodes,
            **kwargs)
    else:
        renders = []

    # For Minari
    acc_rewards = []
    for traj in trajs:
        rewards = traj["reward"]
        acc_rewards.append(np.sum(rewards))
    stats["acc_reward"] = np.mean(acc_rewards)

    return stats, trajs, renders
