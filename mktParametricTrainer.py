# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:00:16 2020

@author: mario.duarte
"""

import random
import numpy as np

from datetime import datetime
import itertools

import gym
from gym.spaces import Discrete, Tuple, Dict, Box, flatten


import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.impala.impala import ImpalaTrainer
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.env.external_env import ExternalEnv
from ray.tune.registry import register_env
#from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel

def flatten_space(space):
    """Flatten a space into a single ``Box``.
    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    Example::
        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that flattens a discrete space::
        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that recursively flattens a dict::
        >>> space = Dict({"position": Discrete(2),
        ...               "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    if isinstance(space, Box):
        return Box(space.low.flatten(), space.high.flatten())
    if isinstance(space, Discrete):
        return Box(low=0, high=1, shape=(space.n, ))
    if isinstance(space, Tuple):
        space = [flatten_space(s) for s in space.spaces]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, Dict):
        space = [flatten_space(s) for s in space.spaces.values()]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    raise NotImplementedError

MKT_TEMPLATES = {'eMail':['mail1','mail2','mail3','mail4'],
                 'webDiscount':['discount1','discount2','discount3','discount4'],
                 'webPremium':['premium1','premium2','premium3','premium4'],
                'callCenter':['script1','script2','script3','script4']}

MKT_REWARDS = { 'email do nothing':-0.5,
                'call do nothing':-5,
                'call discount purchase':75,
                'call premium purchase':130,
                'web discount purchase':80,
                'web premium purchase':135}

CUSTOMER_BEHAVIOR = {'eMail':['email do nothing', 'callCenter', 'webDiscount','webPremium'],
                     'webDiscount':['email do nothing','webPremium','web discount purchase'],
                     'webPremium':['email do nothing','webDiscount','web premium purchase','callCenter'],
                     'callCenter':['call do nothing','call discount purchase','call premium purchase']
                     }

CUSTOMER_ATTRIBUTES = {'age': ['<25', '25-45', '>45'],
                       'sex': ['Men', 'Women'],
                       'region': ['Lisbon', 'Oporto', 'North', 'Center', 'South']}


def _get_action_mask(actions: list, max_actions: int):
    action_mask = [0] * max_actions
    action_len = len(actions)
    action_mask[:action_len] = [1] * action_len
    return action_mask

tp_actions = MKT_TEMPLATES
max_action_size = max([len(options) for options in MKT_TEMPLATES.values()])
action_mask = {tp_id: _get_action_mask(tp_actions[tp], max_action_size) for tp_id, tp
               in enumerate(tp_actions.keys())}

FLAT_OBSERVATION_SPACE = Box(low=0, high=1, shape=(20,), dtype=np.int64)
REAL_OBSERVATION_SPACE = Tuple((Discrete(10), Discrete(3), Discrete(2), Discrete(5)))

class FlattenObservation(gym.ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = flatten_space(env.observation_space['state'])

    def observation(self, observation):
        return flatten(self.env.observation_space['state'], observation)

class MKTEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(max_action_size)
        self.observation_space = Dict({
            "state": REAL_OBSERVATION_SPACE,
            "action_mask": Box(low=0, high=1, shape=(max_action_size,))
        })

flat = FlattenObservation(MKTEnv())

class MKTWorld(MKTEnv):
    def __init__(self, config):
        super(MKTWorld, self).__init__()
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
        self.observation_space = Dict({
            "state": FLAT_OBSERVATION_SPACE,
            "action_mask": Box(low=0, high=1, shape=(max_action_size,))
        })

    def random_customer(self):
        cs = self.customer_segments[np.random.randint(len(self.customer_segments))]
        return dict(zip(self.customer_features, cs))

    def reset(self):
        cs = self.random_customer()
        self.customer_segment = tuple(cs.values())
        customer_feature = list(self.customer_features)
        self.observation[0] = 0
        for i,_ in enumerate(CUSTOMER_ATTRIBUTES.keys()):
            self.observation[i+1] = self.customer_values[i].index(cs[customer_feature[i]])

        return {'action_mask': action_mask[0], 'state': flat.observation(self.observation)}

    def step(self, action: int):
        touch_point = self.touch_points[self.observation[0]]
        assert action < len(self.mkt_offers[touch_point]), \
            "Action={}, TP={}, OFFERS={}".format(action, touch_point, self.mkt_offers[touch_point])
        mkt_offer = self.mkt_offers[touch_point][action]
        new_touch_point = np.random.choice(
            self.journeys[touch_point],
            p=self.probab[self.customer_segment][touch_point][mkt_offer]
        )
        self.observation[0] = self.touch_points.index(new_touch_point)
        done = new_touch_point in self.rewards.keys()
        reward = self.rewards[new_touch_point] if done else 0
        return {'action_mask': action_mask[self.observation[0]] if not done else [1]*max_action_size, 'state': flat.observation(
            self.observation)}, reward, done, {}

env_config = {
    "mkt_rewards": MKT_REWARDS,
    "customer_journeys": CUSTOMER_BEHAVIOR,
    "customer_attributes": CUSTOMER_ATTRIBUTES,
    "mkt_offers": MKT_TEMPLATES
}


class ExternalMkt(ExternalEnv):
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

tf1, tf, tfv = try_import_tf()

class ParametricActionsModel(DistributionalQTFModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):

        print("{} : [INFO] ParametricActionsModel {}, {}, {}, {}, {}"
             .format(datetime.now(),action_space, obs_space, num_outputs, name, model_config))

        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        # print("####### obs_space {}".format(obs_space))
        # raise Exception("END")

        self.action_param_model = FullyConnectedNetwork(
            FLAT_OBSERVATION_SPACE, action_space, num_outputs,
            model_config, name + "_action_param")
        self.register_variables(self.action_param_model.variables())

    def forward(self, input_dict, state, seq_lens):

        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_param, _ = self.action_param_model({
            "obs": input_dict["obs"]["state"]
        })

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_param + inf_mask, state

    def value_function(self):
        return self.action_param_model.value_function()



if __name__ == "__main__":
    ray.init()

    register_env(
        "ExternalMkt",
        #lambda _: HeartsEnv()
        lambda _: ExternalMkt(MKTWorld(env_config), episodes=1000)
    )

    ModelCatalog.register_custom_model("ParametricActionsModel", ParametricActionsModel)

    ppo_config = {"timesteps_per_iteration": 1000,
                  "model": {"custom_model": "ParametricActionsModel",},
                  "num_workers": 0}

    dqn_config = {"timesteps_per_iteration": 1000,
                  "model": {"custom_model": "ParametricActionsModel"},
                  "num_workers": 0,
                  "hiddens": [],
                  "dueling": False,
                  #"v_min": -26,
                  #"v_max": 26,
                  #"noisy": True
                  }

    trainer = DQNTrainer(env="ExternalMkt", config=dqn_config)
    #trainer = PPOTrainer(env="ExternalHearts", config=ppo_config)

    i = 1
    while True:
        result = trainer.train()
        print("Iteration {}, Episodes {}, Mean Reward {}, Mean Length {}".format(
            i, result['episodes_this_iter'], result['episode_reward_mean'], result['episode_len_mean']
        ))
        i += 1

    ray.shutdown()

