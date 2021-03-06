import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

import argparse

from simpy_env import SimpyEnv
from trafic_light_model import N_ACTIONS, OBSERVATION_SPACE, SimModel, print_stats

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

    best_checkpoint = "/home/md/ray_results/PPO_SimpyEnv_2021-02-07_10-19-33rtwyf_gp/checkpoint_195/checkpoint-195"
    agent = ppo.PPOTrainer(config=ppo_config, env="SimpyEnv")
    # best_checkpoint = "/home/md/ray_results/DQN_SimpyEnv_2021-02-06_21-07-35qthgy5z5/checkpoint_95/checkpoint-95"
    # agent = dqn.DQNTrainer(config=dqn_config, env="SimpyEnv")

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
        print_stats(he.sim)


    ray.shutdown()