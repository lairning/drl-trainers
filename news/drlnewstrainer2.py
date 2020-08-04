# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:00:16 2020

@author: mario.duarte
"""

import numpy as np
import argparse

import gym
from gym.spaces import Discrete, Box, Tuple

import ray
from ray.rllib.env.external_env import ExternalEnv
from ray.tune.registry import register_env

from ray.rllib.agents import dqn, a3c, ppo, sac

parser = argparse.ArgumentParser()
parser.add_argument("--stop", type=int, default=20)

N_TOPICS = 15
TOPICS = ['T{}'.format(i) for i in range(N_TOPICS)]

CONTEXT_ATTRIBUTES = {'hour':['0-7', '8-9', '10-12','13-14','15-18','19-21','22-23'],
                       'week period': ['Weekday', 'Weekend'],
                       'weather': ['Sunny','Cloudy','Raining'],
                      'device':['mobile ios','mac ios','mobile android','windows']}

OBSERVATION_0 = len(CONTEXT_ATTRIBUTES) * [0] + N_TOPICS * [0]

OBSERVATION_SPACE = Tuple((Discrete(7),
                           Discrete(2),
                           Discrete(3),
                           Discrete(4),
                           Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2),
                           Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2),
                           Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2)
                           ))

# ACTION_SPACE = Box(low=0.0, high=1.0, shape=(N_TOPICS,))

ACTION_SPACE = Tuple( (Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2),
                       Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2),
                       Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2)
                       ))

# Probability of a user click based on the distance bwteen article topics
PROBAB = N_TOPICS*[0]
PROBAB[1:8] = [0.2, 0.5, 0.7, 0.4, 0.3, 0.2, 0.1]

def distance(article1, article2):
    return sum(abs(article1[i]-article2[i]) for i in range(N_TOPICS))

# Start Aticles
N_ARICLES = 5
p = 5 / N_TOPICS
START_ARTICLES = [
    [np.random.choice([0,1],p=[1-p,p]) for _ in range(N_TOPICS)] for _ in range(N_ARICLES)
]
np.random.choice([0,1],p=[1-p,p])

class NewsWorld(gym.Env):


    def __init__(self,config):
        self.observation = OBSERVATION_0
        self.observation_space = OBSERVATION_SPACE
        self.action_space = ACTION_SPACE

    def reset(self):
        self.observation = OBSERVATION_0
        self.observation[0] = np.random.choice([i for i in range(len(CONTEXT_ATTRIBUTES['hour']))])
        self.observation[1] = np.random.choice([i for i in range(len(CONTEXT_ATTRIBUTES['week period']))])
        self.observation[2] = np.random.choice([i for i in range(len(CONTEXT_ATTRIBUTES['weather']))])
        self.observation[3] = np.random.choice([i for i in range(len(CONTEXT_ATTRIBUTES['device']))])
        a = np.random.choice(range(N_ARICLES))
        self.observation[len(CONTEXT_ATTRIBUTES):] = START_ARTICLES[a]
        return self.observation

    def step(self, action: list):
        d = distance(action, self.observation[len(CONTEXT_ATTRIBUTES):])
        p = PROBAB[d]+np.random.normal(0,0.4)
        click = p > 0.5
        done = not click
        reward = 1 if click else 0
        self.observation[len(CONTEXT_ATTRIBUTES):] = action
        return self.observation, reward, done, {}

class ExternalWorld(ExternalEnv):
    def __init__(self, env):
        ExternalEnv.__init__(self, env.action_space, env.observation_space)
        self.env = env

    def run(self):

        while True:
            eid = self.start_episode()
            obs = self.env.reset()
            done = False
            while not done:
                action = self.get_action(eid, obs)
                obs, reward, done, info = self.env.step(action)
                self.log_returns(eid, reward, info=info)
            self.end_episode(eid, obs)

dqn_config = {
    "v_min": 0.0,
    "v_max": 10.0,
    "hiddens": [128],
    "exploration_config": {
        "epsilon_timesteps": 5000,
    },
    "learning_starts": 100,
    "timesteps_per_iteration": 1200,
    'lr': 5e-5,
    "num_atoms": 2
}

ppo_config = {
    "env": NewsWorld
}

a3c_config = {
    "env": NewsWorld
}

sac_config = {
    "env": NewsWorld
}

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=10)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    register_env(
        "NewsLearn",
        #lambda _: HeartsEnv()
        lambda _: ExternalWorld(env=NewsWorld(dict()))
    )

    trainer = a3c.A3CTrainer(env="NewsLearn", config=dict())

    for i in range(args.iterations):
        result = trainer.train()
        print("Iteration {}, Episodes {}, Mean Reward {}, Mean Length {}".format(
            i, result['episodes_this_iter'], result['episode_reward_mean'], result['episode_len_mean']
        ))
        i += 1

    ray.shutdown()
