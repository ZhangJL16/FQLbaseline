import warnings
import numpy as np
from minari import MinariDataset

MINARI_BENCHMARK_DATASETS = (
    "D4RL/door/cloned-v2",
    "D4RL/door/expert-v2",
    "D4RL/door/human-v2",
    "D4RL/hammer/cloned-v2",
    "D4RL/hammer/expert-v2",
    "D4RL/hammer/human-v2",
    "D4RL/pen/cloned-v2",
    "D4RL/pen/expert-v2",
    "D4RL/pen/human-v2",
    "D4RL/relocate/cloned-v2",
    "D4RL/relocate/expert-v2",
    "D4RL/relocate/human-v2",
    "mujoco/halfcheetah/expert-v0",
    "mujoco/halfcheetah/medium-v0",
    "mujoco/halfcheetah/simple-v0",
    "mujoco/hopper/expert-v0",
    "mujoco/hopper/medium-v0",
    "mujoco/hopper/simple-v0",
    "mujoco/walker2d/expert-v0",
    "mujoco/walker2d/medium-v0",
    "mujoco/walker2d/simple-v0",
)

MINARI_BENCHMARK_GROUPS = {
    "all": MINARI_BENCHMARK_DATASETS,
    "adroit": tuple(ds for ds in MINARI_BENCHMARK_DATASETS if ds.startswith("D4RL/")),
    "mujoco": tuple(ds for ds in MINARI_BENCHMARK_DATASETS if ds.startswith("mujoco/")),
}


def get_minari_benchmark_datasets(group: str = "all") -> tuple[str, ...]:
    try:
        return MINARI_BENCHMARK_GROUPS[group]
    except KeyError as exc:
        valid = ", ".join(sorted(MINARI_BENCHMARK_GROUPS))
        raise ValueError(f"Unknown Minari dataset group '{group}'. Valid groups: {valid}.") from exc

def convert_episodes_to_transitions(episodes):
    # Recover OGBench-like dataset
    transitions = dict(
        (key,[]) for key in [
            "actions",
            "observations",
            "next_observations",
            "rewards",
            "masks",
            "terminals",
        ])

    for episode in episodes:
        for t in range(len(episode.rewards)):
            transitions["actions"].append(episode.actions[t])
            transitions["observations"].append(episode.observations[t])
            transitions["next_observations"].append(episode.observations[t+1])
            transitions["rewards"].append(episode.rewards[t])
            transitions["masks"].append(1.0 - episode.terminations[t])
            transitions["terminals"].append(0.0)

    return dict((k,np.array(v)) for (k,v) in transitions.items())


MINARI_DICT_REF_SCORES = dict(
    hopper = (-20.272305,3234.3),
    halfcheetah = (-280.178953,12135.0),
    walker2d = (1.629008,4592.3),
    pen = (137.92955108500217,8820.50845967583),
    door = (-45.80706024169922,2940.578369140625),
    hammer = (-267.074462890625,12635.712890625),
    relocate = (9.189092636108398,4287.70458984375),
)
"""
Minari's datasets are suppose to provide 
[reference scores](https://github.com/Farama-Foundation/Minari/issues/306) in
`dataset.storage.metadata["ref_max_score"]`, but some were 
[missing](https://github.com/Farama-Foundation/Minari/issues/323) (e.g. hopper) 
or [incorrect](https://github.com/Farama-Foundation/Minari/issues/303) (e.g. pen)

So we manually specified the reference scores in this dict.
If the metadata does contain reference scores, we compare them to these numbers and raise a warning if they don't match.
"""

MINARI_REF_SCORES: tuple[float,float]
def check_ref_scores(dataset:MinariDataset):
    global MINARI_REF_SCORES
    metadata = dataset.storage.metadata
    if 'ref_min_score' in metadata:
        metadata_scores = (metadata["ref_min_score"],metadata["ref_max_score"])
    else:
        metadata_scores = None
    for key,saved_scores in MINARI_DICT_REF_SCORES.items():
        if key in dataset.id:
            MINARI_REF_SCORES = saved_scores
            if (metadata_scores is not None 
            and np.linalg.norm(np.array(metadata_scores)-np.array(saved_scores)) > 0.1
            ):
                warnings.warn(
                    "=== WARNING ===\nMinari reference scores found in dataset's metadata "\
                    "do not match manually specified ones.\nWe ignore metadata's reference "\
                    "scores as in July 2025 some were still incorrect "\
                    "https://github.com/Farama-Foundation/Minari/issues/303 or missing in "\
                    "January 2026 https://github.com/Farama-Foundation/Minari/issues/323"
                )
            return
    if metadata_scores is not None:
        MINARI_REF_SCORES = metadata_scores
        return
    raise Exception(
        f"Reference scores for the Minari dataset {dataset.id} were neither "
        f"found in the dataset metadata nor in our dict of reference scores."
    )
    

def minari_normalized_score(acc_reward):
    min_score, max_score = MINARI_REF_SCORES
    return 100 * (acc_reward - min_score) / (max_score - min_score)
    
