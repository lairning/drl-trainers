# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:00:16 2020

@author: mario.duarte
"""

import numpy as np
import itertools

import gym
from gym.spaces import Discrete, Box, Tuple

import ray
from ray import tune
from ray.rllib.agents import dqn, a3c, ppo, sac


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
PROBAB[1:8] = [0.3, 0.5, 0.7, 0.5, 0.4, 0.2, 0.1]

def distance(article1, article2):
    return sum(abs(article1[i]-article2[i]) for i in range(N_TOPICS))

# Start Aticles
N_ARICLES = 5
p = 3 / N_TOPICS
START_ARTICLES = [
    [np.random.choice([0,1],p=[1-p,p]) for _ in range(N_TOPICS)] for _ in range(N_ARICLES)
]
np.random.choice([0,1],p=[1-p,p])

class NewsWorld(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

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
        p = PROBAB[d]+np.random.normal(0,1)
        click = p > 0.5
        done = not click
        reward = 1 if click else 0
        self.observation[len(CONTEXT_ATTRIBUTES):] = action
        return self.observation, reward, done, {}

dqn_config = {
    "env": NewsWorld,
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

if __name__ == "__main__":
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))

    stop = {
        "training_iteration": 20
    }

    # results_dqn = tune.run(dqn.DQNTrainer, config=dqn_config, stop=stop)

    results_ppo = tune.run(ppo.PPOTrainer, config=ppo_config, stop=stop)

    # results_a3c = tune.run(a3c.A3CTrainer, config=a3c_config, stop=stop)

    # results_sac = tune.run(sac.SACTrainer, config=sac_config, stop=stop)

    ray.shutdown()
