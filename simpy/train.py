import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray.rllib.agents.ppo as ppo

import argparse

from simpy_model import SimpyEnv

parser = argparse.ArgumentParser()
parser.add_argument("--stop", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    register_env(
        "SimpyEnv",
        lambda _: SimpyEnv()
    )

    ppo_config = {
        "vf_clip_param": 50,  # tune.grid_search([20.0, 100.0]),
        "num_workers"  : 5,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"   : "complete_episodes",
        "framework"    : "torch"
    }

    trainer = ppo.PPOTrainer(config=ppo_config, env="SimpyEnv")

    for _ in range(args.stop):
        result = trainer.train()
        print(pretty_print(result))

    ray.shutdown()
