# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:00:16 2020

@author: mario.duarte
"""

import numpy as np
import argparse
import itertools
import csv
import json

import gym
from gym.spaces import Discrete, Box, Tuple

import ray
from ray.rllib.env.external_env import ExternalEnv
from ray.tune.registry import register_env

# from ray.rllib.agents import dqn, a3c, ppo, sac, marwil
from ray.rllib.agents.marwil import MARWILTrainer
from ray.rllib.agents.a3c import A3CTrainer


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


    def __init__(self, config: dict):
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

first_line = 0

class HistoricalLearn(ExternalEnv):
    def __init__(self, env, file_name: str):
        ExternalEnv.__init__(self, env.action_space, env.observation_space)
        self.csv_file = open(file_name, newline='')

    def run(self):
        global first_line

        csv_file = itertools.islice(self.csv_file, first_line, None)
        reader = csv.reader(csv_file)

        done = True
        eid = None
        while True:
            if done:
                eid = self.start_episode()
            try:
                row = next(reader)
            except StopIteration:
                reader = csv.reader(self.csv_file)
            _, observation, action, new_observation, reward, done = tuple(row)
            observation = json.loads(observation)
            action = json.loads(action)
            new_observation = json.loads(new_observation)
            reward = float(reward)
            done = True if done == 'True' else False
            self.log_action(eid, observation, action)
            self.log_returns(eid, reward)
            if done:
                self.end_episode(eid, new_observation)
                first_line += 1

class OnlineLearn(ExternalEnv):
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

marwil_config = {
    "evaluation_num_workers": 1,
    "evaluation_interval": 1,
    "input_evaluation": ["wis"],
    "evaluation_config": {"input": "sampler"},
    "beta": 1, #tune.grid_search([0, 1])
}

a3c_config = {
    "num_workers": 1,
    "gamma": 0.95,
}

dqn_config = {
    "v_min": 0.0,
    "v_max": 5.0,
    "hiddens": [128],
    "exploration_config": {
        "epsilon_timesteps": 5000,
    },
    # "learning_starts": 100,
    'lr': 5e-5,
    "num_atoms": 2,
}


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--hister", type=int, default=20)
parser.add_argument("--onliter", type=int, default=20)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    register_env(
        "HistoricalLearn",
        #lambda _: HeartsEnv()
        lambda _: HistoricalLearn(NewsWorld(dict()), args.file)
    )

    register_env(
        "OnlinelLearn",
        #lambda _: HeartsEnv()
        lambda _: OnlineLearn(NewsWorld(dict()))
    )

    #trainer = MARWILTrainer(config=marwil_config, env="HistoricalLearn")
    trainer1 = A3CTrainer(config=a3c_config, env="HistoricalLearn")

    for i in range(args.hister):
        result = trainer1.train()
        print("Iteration {}, Episodes {}, Mean Reward {}, Mean Length {}".format(
            i, result['episodes_this_iter'], result['episode_reward_mean'], result['episode_len_mean']
        ))
        i += 1

    checkpoint = trainer1.save()

    trainer2 = A3CTrainer(config=a3c_config, env="OnlineLearn")

    trainer2.restore(checkpoint)

    for i in range(args.onliter):
        result = trainer2.train()
        print("Iteration {}, Episodes {}, Mean Reward {}, Mean Length {}".format(
            i, result['episodes_this_iter'], result['episode_reward_mean'], result['episode_len_mean']
        ))
        i += 1


    ray.shutdown()