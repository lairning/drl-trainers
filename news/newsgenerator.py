# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:00:16 2020

@author: mario.duarte
"""

import numpy as np
import argparse

import gym
from gym.spaces import Discrete, Box, Tuple

import os

import ray.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

parser = argparse.ArgumentParser()
parser.add_argument("--stop", type=int, default=10000)

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

# Probability of a user click based on the distance between article topics
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


    def __init__(self):
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

    def action_space_sample(self):
        dist = np.random.randint(10)
        change_topics = np.random.choice(N_TOPICS,dist,replace=False)
        action = self.observation[len(CONTEXT_ATTRIBUTES):].copy()
        for i in change_topics:
            action[i] = 0 if action[i] else 1
        return action


if __name__ == "__main__":
    args = parser.parse_args()

    batch_builder = SampleBatchBuilder()

    output_path = os.path.join(ray.utils.get_user_temp_dir(), "demo-out")
    writer = JsonWriter(output_path)
    print("OUTPUT IN {}".format(output_path))

    env = NewsWorld()

    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    for eps_id in range(args.stop):
        obs = env.reset()
        prev_action = np.zeros_like(env.action_space_sample())
        prev_reward = 0
        done = False
        t = 0
        while not done:
            action = env.action_space_sample()
            new_obs, rew, done, info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info,
                new_obs=prep.transform(new_obs))
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
        writer.write(batch_builder.build_and_reset())

