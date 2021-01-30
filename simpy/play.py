import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

import argparse

from simpy_env import SimpyEnv
from simpy_model import N_ACTIONS, OBSERVATION_SPACE, SimModel

parser = argparse.ArgumentParser()
parser.add_argument("--stop", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    register_env(
        "SimpyEnv",
        lambda _: SimpyEnv(N_ACTIONS, OBSERVATION_SPACE, SimModel)
    )

    ppo_config = {
        "env"          : "SimpyEnv",
        "vf_clip_param": 50,  # tune.grid_search([20.0, 100.0]),
        "num_workers"  : 0,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"   : "complete_episodes",
        "framework"    : "torch"
    }

    best_checkpoint = "/home/md/ray_results/PPO_2021-01-23_20-53-45/PPO_SimpyEnv_b3e12_00000_0_2021-01-23_20-53-45/checkpoint_120/checkpoint-120"

    agent = ppo.PPOTrainer(config=ppo_config, env="SimpyEnv")
    agent.restore(best_checkpoint)

    # instantiate env class
    he = SimpyEnv(N_ACTIONS, OBSERVATION_SPACE, SimModel)

    for _ in range(args.stop):
        # run until episode ends
        episode_reward = 0
        done = False
        obs = he.reset()
        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = he.step(action)
            episode_reward += reward
        print("Total:",episode_reward)


    ray.shutdown()