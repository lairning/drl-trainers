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

    result = trainer.train()
    checkpoint = trainer.save()
    best_reward = result['episode_reward_mean']
    best_checkpoint = checkpoint
    print("Mean Reward {}:{}".format(0, result['episode_reward_mean']))
    print("Checkpoint at", checkpoint)
    for i in range(1,args.stop):
        result = trainer.train()
        print("Mean Reward {}:{}".format(i,result['episode_reward_mean']))
        best_reward = max(best_reward,result['episode_reward_mean'])
        if best_reward == ['episode_reward_mean']:
            checkpoint = trainer.save()
            best_checkpoint = checkpoint
            print("Checkpoint at", checkpoint)

    print("BEST Mean Reward  :",best_reward)
    print("BEST Checkpoint at:", checkpoint)

    ray.shutdown()
