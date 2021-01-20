
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
        "v_min"       : -40000.0,
        "v_max"       :   1200.0,
        "env"         : "SimpyEnv",
        "num_workers" : 5
        }

    ppo_config = {
        "env"           : "SimpyEnv",
        "vf_clip_param" : 10000.0,
        "num_workers"   : 5
    }

    stop = {
        "training_iteration": args.stop
    }


    results_dqn = tune.run(dqn.DQNTrainer, config=dqn_config, stop=stop)

    # results = tune.run("PPO", config=ppo_config, stop=stop)

    # results_sac = tune.run(sac.SACTrainer, config=sac_config, stop=stop)

    ray.shutdown()

