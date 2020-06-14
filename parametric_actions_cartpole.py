"""Example of handling variable length and/or parametric action spaces.
This is a toy example of the action-embedding based approach for handling large
discrete action spaces (potentially infinite in size), similar to this:
    https://neuro.cs.ut.ee/the-use-of-embeddings-in-openai-five/
This currently works with RLlib's policy gradient style algorithms
(e.g., PG, PPO, IMPALA, A2C) and also DQN.
Note that since the model outputs now include "-inf" tf.float32.min
values, not all algorithm options are supported at the moment. For example,
algorithms might crash if they don't properly ignore the -inf action scores.
Working configurations are given below.
"""

import argparse

import gym
from gym.spaces import Box, Dict, Discrete
import numpy as np
import random

import ray
from ray import tune
from ray.rllib.examples.models.parametric_actions_model import \
    ParametricActionsModel, TorchParametricActionsModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

class ParametricActionsCartPole(gym.Env):
    """Parametric action version of CartPole.
    In this env there are only ever two valid actions, but we pretend there are
    actually up to `max_avail_actions` actions that can be taken, and the two
    valid actions are randomly hidden among this set.
    At each step, we emit a dict of:
        - the actual cart observation
        - a mask of valid actions (e.g., [0, 0, 1, 0, 0, 1] for 6 max avail)
        - the list of action embeddings (w/ zeroes for invalid actions) (e.g.,
            [[0, 0],
             [0, 0],
             [-0.2322, -0.2569],
             [0, 0],
             [0, 0],
             [0.7878, 1.2297]] for max_avail_actions=6)
    In a real environment, the actions embeddings would be larger than two
    units of course, and also there would be a variable number of valid actions
    per step instead of always [LEFT, RIGHT].
    """

    def __init__(self, max_avail_actions):
        # Use simple random 2-unit action embeddings for [LEFT, RIGHT]
        self.left_action_embed = np.random.randn(2)
        self.right_action_embed = np.random.randn(2)
        self.action_space = Discrete(max_avail_actions)
        self.wrapped = gym.make("CartPole-v0")
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(max_avail_actions, )),
            "avail_actions": Box(-10, 10, shape=(max_avail_actions, 2)),
            "cart": self.wrapped.observation_space,
        })

    def update_avail_actions(self):
        self.action_assignments = np.array([[0., 0.]] * self.action_space.n)
        self.action_mask = np.array([0.] * self.action_space.n)
        self.left_idx, self.right_idx = random.sample(
            range(self.action_space.n), 2)
        self.action_assignments[self.left_idx] = self.left_action_embed
        self.action_assignments[self.right_idx] = self.right_action_embed
        self.action_mask[self.left_idx] = 1
        self.action_mask[self.right_idx] = 1

    def reset(self):
        self.update_avail_actions()
        return {
            "action_mask": self.action_mask,
            "avail_actions": self.action_assignments,
            "cart": self.wrapped.reset(),
        }

    def step(self, action):
        if action == self.left_idx:
            actual_action = 0
        elif action == self.right_idx:
            actual_action = 1
        else:
            raise ValueError(
                "Chosen action was not one of the non-zero action embeddings",
                action, self.action_assignments, self.action_mask,
                self.left_idx, self.right_idx)
        orig_obs, rew, done, info = self.wrapped.step(actual_action)
        self.update_avail_actions()
        obs = {
            "action_mask": self.action_mask,
            "avail_actions": self.action_assignments,
            "cart": orig_obs,
        }
        return obs, rew, done, info

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=200)
parser.add_argument("--stop-reward", type=float, default=150.0)
parser.add_argument("--stop-timesteps", type=int, default=100000)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    register_env("pa_cartpole", lambda _: ParametricActionsCartPole(10))
    ModelCatalog.register_custom_model(
        "pa_model", TorchParametricActionsModel
        if args.torch else ParametricActionsModel)

    if args.run == "DQN":
        cfg = {
            # TODO(ekl) we need to set these to prevent the masked values
            # from being further processed in DistributionalQModel, which
            # would mess up the masking. It is possible to support these if we
            # defined a a custom DistributionalQModel that is aware of masking.
            "hiddens": [],
            "dueling": False,
        }
    else:
        cfg = {}

    config = dict({
        "env": "pa_cartpole",
        "model": {
            "custom_model": "pa_model",
        },
        "num_workers": 0,
        "framework": "torch" if args.torch else "tf",
    }, **cfg)

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, stop=stop, config=config)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()