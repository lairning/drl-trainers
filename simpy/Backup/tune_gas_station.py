import ray
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

import argparse

from simpy_env import SimpyEnv

if __name__ == "__main__":
    from gas_station_model import N_ACTIONS, OBSERVATION_SPACE, SimModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", type=int, default=1)
    args = parser.parse_args()

    ray.init()

    register_env(
        "SimpyEnv",
        lambda _: SimpyEnv(N_ACTIONS, OBSERVATION_SPACE, SimModel)
    )

    dqn_config = {
        "v_min"      : -30000.0,
        "v_max"      : 1000.0,
        "env"        : "SimpyEnv",
        "hiddens"    : [256, 256],
        "num_workers": 5,
        "framework"    : "torch"
    }

    ppo_config = {
        "env"          : "SimpyEnv",
        "vf_clip_param": 50,  # tune.grid_search([20.0, 100.0]),
        "num_workers"  : 5,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"   : "complete_episodes",
        "framework"    : "torch"
    }

    stop = {
        "training_iteration": args.stop
    }

    results_ppo = tune.run(ppo.PPOTrainer, config=ppo_config, stop=stop)

    ray.shutdown()
