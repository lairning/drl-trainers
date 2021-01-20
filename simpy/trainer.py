
import ray
from ray import tune
from ray.rllib.agents import dqn, a3c, ppo, sac
from ray.tune.registry import register_env

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

    dqn_config = {
        "v_min"       : -30000.0,
        "v_max"       :   1000.0,
        "env"         : "SimpyEnv",
        "hiddens"     : [256, 256],
        "num_workers" : 5
        }

    ppo_config = {
        "env"           : "SimpyEnv",
        "vf_clip_param" : tune.grid_search([20.0, 100.0]),
        "num_workers"   : 5,
        "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"    : "complete_episodes"
    }

    stop = {
        "training_iteration": args.stop
    }


    # results_dqn = tune.run(dqn.DQNTrainer, config=dqn_config, stop=stop)

    results_ppo = tune.run(ppo.PPOTrainer, config=ppo_config, stop=stop)


    ray.shutdown()

