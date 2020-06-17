# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:00:16 2020

@author: mario.duarte
"""

import numpy as np
import itertools

import gym
from gym.spaces import Discrete, Box, Tuple
from ray.tune.logger import pretty_print

import ray
from ray import tune
from ray.rllib.agents import dqn, a3c, ppo

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


dqn_config = {
    "env": MKTWorld,  # or "corridor" if registered above
    "env_config": {
        "mkt_rewards": MKT_REWARDS,
        "customer_journeys": CUSTOMER_BEHAVIOR,
        "customer_attributes": CUSTOMER_ATTRIBUTES,
        "mkt_offers": MKT_TEMPLATES
    },
    "v_min": -1.0,
    "v_max": 5.0,
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
    "env": MKTWorld,  # or "corridor" if registered above
    "env_config": {
        "mkt_rewards": MKT_REWARDS,
        "customer_journeys": CUSTOMER_BEHAVIOR,
        "customer_attributes": CUSTOMER_ATTRIBUTES,
        "mkt_offers": MKT_TEMPLATES
    }
}

a3c_config = {
    "env": MKTWorld,  # or "corridor" if registered above
    "env_config": {
        "mkt_rewards": MKT_REWARDS,
        "customer_journeys": CUSTOMER_BEHAVIOR,
        "customer_attributes": CUSTOMER_ATTRIBUTES,
        "mkt_offers": MKT_TEMPLATES
    }
}

if __name__ == "__main__":
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))

    stop = {
        "training_iteration": 20
    }

    results_dqn = tune.run(dqn.DQNTrainer, config=dqn_config, stop=stop)

    #results_ppo = tune.run(ppo.PPOTrainer, config=ppo_config, stop=stop)

    #results_a3c = tune.run(a3c.A3CTrainer, config=a3c_config, stop=stop)

    ray.shutdown()
