# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:00:16 2020

@author: mario.duarte
"""

import numpy as np
import itertools
from scipy.stats import gamma
from datetime import datetime, timedelta

import gym
from gym.spaces import Discrete, Box, Tuple

import ray

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.external_env import ExternalEnv
from ray.tune.registry import register_env

MKT_TEMPLATES = {'eMail': ['mail1', 'mail2', 'mail3', 'mail4'],
                 'webDiscount': ['discount1', 'discount2', 'discount3', 'discount4'],
                 'webInfo': ['info1', 'info2', 'info3', 'info4'],
                 'webPremium': ['premium1', 'premium2', 'premium3', 'premium4']}

CUSTOMER_ATTRIBUTES = {'age': ['<25', '25-45', '>45'],
                       'sex': ['Men', 'Women'],
                       'region': ['Lisbon', 'Oporto', 'North', 'Center', 'South']}

MKT_REWARDS = {'do_nothing': -1,
               'discount_purchase': 2,
               'premium_purchase': 5}

CUSTOMER_BEHAVIOR = {'eMail': ['do_nothing', 'webDiscount', 'webInfo'],
                     'webDiscount': ['do_nothing', 'webInfo', 'discount_purchase'],
                     'webInfo': ['do_nothing', 'webDiscount', 'webPremium'],
                     'webPremium': ['do_nothing', 'webDiscount', 'premium_purchase']
                     }


class World:
    def __init__(self, customer_attributes, customer_journeys, mkt_offers, mkt_rewards):
        self.probab = dict()
        self.rewards = mkt_rewards
        self.journeys = customer_journeys
        self.customer_features = customer_attributes.keys()
        a = list(customer_attributes.values())
        self.customer_segments = list(itertools.product(*a))
        for cs in self.customer_segments:
            dt = dict()
            for t in mkt_offers.keys():
                dt[t] = {mo: np.random.dirichlet(np.ones(len(customer_journeys[t])), size=1)[0] for mo in mkt_offers[t]}
            self.probab[cs] = dt

    def random_customer(self):
        cs = self.customer_segments[np.random.randint(len(self.customer_segments))]
        return dict(zip(self.customer_features, cs))

    def reaction(self, customer_attributes, touch_point, mkt_offer):
        customer_segment = tuple(customer_attributes.values())
        touch_point = np.random.choice(
            self.journeys[touch_point],
            p=self.probab[customer_segment][touch_point][mkt_offer]
        )
        return touch_point

    def customer_behaviour(self, customer_segment, touch_point, mkt_offer):
        return list(zip(
            self.journeys[touch_point],
            self.probab[customer_segment][touch_point][mkt_offer]
        ))

    def first_touch_point(self):
        return list(self.journeys.keys())[0]

    def exit_point(self, touch_point):
        return touch_point in self.rewards.keys()

    def remove_offer(self, touch_point, offer):
        for cs in self.probab.keys():
            if touch_point not in self.probab[cs].keys():
                raise Exception('Unknown Touch Point :' + str(touch_point))
            if offer not in self.probab[cs][touch_point].keys():
                raise Exception('Unknown Offer :' + str(offer))
            del self.probab[cs][touch_point][offer]

    def add_offer(self, touch_point, offer):
        for cs in self.probab.keys():
            self.probab[cs][touch_point][offer] = np.random.dirichlet(np.ones(len(self.journeys[touch_point])), size=1)[
                0]


class MKTWorld(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.probab = dict()
        self.rewards = config["mkt_rewards"]
        self.journeys = config["customer_journeys"]
        self.touch_points = list(config["customer_journeys"].keys()) + list(config["mkt_rewards"].keys())
        self.customer_features = config["customer_attributes"].keys()
        self.customer_values = list(config["customer_attributes"].values())
        self.customer_segments = list(itertools.product(*self.customer_values))
        self.customer_segment = self.customer_segments[0]
        self.mkt_offers = config["mkt_offers"]
        self.observation = list(4 * [0])
        for cs in self.customer_segments:
            dt = dict()
            for t in self.mkt_offers.keys():
                dt[t] = {mo: np.random.dirichlet(np.ones(len(self.journeys[t])), size=1)[0] for mo in
                         self.mkt_offers[t]}
            self.probab[cs] = dt
        self.action_space = Discrete(4)
        self.observation_space = Tuple((Discrete(7), Discrete(3), Discrete(2), Discrete(6)))

    def random_customer(self):
        cs = self.customer_segments[np.random.randint(len(self.customer_segments))]
        return dict(zip(self.customer_features, cs))

    def reset(self):
        cs = self.random_customer()
        self.customer_segment = tuple(cs.values())
        customer_feature = list(self.customer_features)
        self.observation[0] = 0
        self.observation[1] = self.customer_values[0].index(cs[customer_feature[0]])
        self.observation[2] = self.customer_values[1].index(cs[customer_feature[1]])
        self.observation[3] = self.customer_values[2].index(cs[customer_feature[2]])
        return self.observation

    def step(self, action):
        assert action in [0, 1, 2, 3], action
        touch_point = self.touch_points[self.observation[0]]
        mkt_offer = self.mkt_offers[touch_point][action]
        new_touch_point = np.random.choice(
            self.journeys[touch_point],
            p=self.probab[self.customer_segment][touch_point][mkt_offer]
        )
        self.observation[0] = self.touch_points.index(new_touch_point)
        done = new_touch_point in self.rewards.keys()
        reward = self.rewards[new_touch_point] if done else 0
        return self.observation, reward, done, {}


class LAIMKTEngine(ExternalEnv):
    def __init__(self, env, episodes: int):
        ExternalEnv.__init__(self, env.action_space, env.observation_space)
        self.env = env
        self.episodes = episodes

    def run(self):

        for e in range(self.episodes):
            eid = self.start_episode()
            obs = self.env.reset()
            done = False
            while not done:
                action = self.get_action(eid, obs)
                obs, reward, done, info = self.env.step(action)
                self.log_returns(eid, reward, info=info)
            self.end_episode(eid, obs)


env_config = {
    "mkt_rewards": MKT_REWARDS,
    "customer_journeys": CUSTOMER_BEHAVIOR,
    "customer_attributes": CUSTOMER_ATTRIBUTES,
    "mkt_offers": MKT_TEMPLATES
}

dqn_config = {
    "v_min": -1.0,
    "v_max": 5.0,
    "hiddens": [128],
    "exploration_config": {
        "epsilon_timesteps": 4000,
    },
    'lr': 5e-5,
    "num_atoms": 2,
    "learning_starts": 100,
    "timesteps_per_iteration": 1200
}

if __name__ == "__main__":
    ray.init()

    register_env(
        "LAIMKTEngine",
        lambda _: LAIMKTEngine(MKTWorld(env_config), episodes=10000)
    )
    dqn = DQNTrainer(
        env="LAIMKTEngine",
        config=dqn_config
    )

    i = 1
    while True:
        result = dqn.train()
        print("Iteration {}, Episodes {}, Mean Reward {}, Mean Length {}".format(
            i, result['episodes_this_iter'], result['episode_reward_mean'], result['episode_len_mean']
        ))
        i += 1

    ray.shutdown()
