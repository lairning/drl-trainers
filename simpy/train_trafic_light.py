import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

import argparse

from simpy_env import SimpyEnv


if __name__ == "__main__":

    from trafic_light_model import N_ACTIONS, OBSERVATION_SPACE, SimModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", type=int, default=1)

    args = parser.parse_args()

    ray.init()

    register_env(
        "SimpyEnv",
        lambda _: SimpyEnv(N_ACTIONS, OBSERVATION_SPACE, SimModel)
    )

    ppo_config = {
        "vf_clip_param": 10,  # tune.grid_search([20.0, 100.0]),
        "num_workers"  : 5,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"   : "complete_episodes",
        "framework"    : "torch"
    }

    dqn_config = {
        "v_min": -1000,
        "v_max": 0,
        "hiddens": [256,256],
        # "noisy": True,
        # "num_atoms": 2,
        # "num_workers"  : 5,
        "framework": "torch"
    }

    trainer = ppo.PPOTrainer(config=ppo_config, env="SimpyEnv")
    #trainer = dqn.DQNTrainer(config=dqn_config, env="SimpyEnv")

    result = trainer.train()
    best_checkpoint = trainer.save()
    best_reward = result['episode_reward_mean']
    print("Mean Reward {}:{}".format(1, result['episode_reward_mean']))

    for i in range(1, args.stop):
        result = trainer.train()
        print("Mean Reward {}:{}".format(i+1, result['episode_reward_mean']))
        best_reward = max(best_reward, result['episode_reward_mean'])
        if best_reward == result['episode_reward_mean']:
            best_checkpoint = trainer.save()

    print("BEST Mean Reward  :", best_reward)
    print("BEST Checkpoint at:", best_checkpoint)

    ray.shutdown()
