# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:00:16 2020

@author: mario.duarte
"""

import random
import numpy as np

import gym
from gym.spaces import Discrete, Tuple, Dict, Box

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.impala.impala import ImpalaTrainer
from ray.rllib.env.external_env import ExternalEnv
from ray.tune.registry import register_env
from ray.rllib.models.tf import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel

from heartsBasicPlayers import BasicPlayer, RandomPlayer
from card import CARD_2P, CARD_DE, Card, CARD_SET, CARD_LIST, CARDS_PER_PLAYER, HAND_SIZE

tf = try_import_tf()

# Card Status
OTHERS_HAND = 0
MY_HAND = 1
PLAYED = 2

CARD_DE_POINTS = HAND_SIZE
MAX_HAND_POINTS = len([c for c in CARD_SET if c.naipe == "C"]) + CARD_DE_POINTS
CHEAT_POINTS = -MAX_HAND_POINTS

t_episodes = 0


class HeartsEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self):
        self.action_space = Discrete(4 * HAND_SIZE)
        observation_tuple = tuple(Discrete(3) for _ in range(4 * HAND_SIZE))
        self.observation_space = Dict({
            "status": Tuple(observation_tuple),
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n,))
        })
        self.game_status = np.array(4 * HAND_SIZE * [OTHERS_HAND])
        self.players = None
        self.hand_points = 0
        self.players_points = 4 * [0]
        self.first_player = None
        self.status = {"hearts_broken": False,
                       "trick_number": 0,
                       "trick_cards_played": []}

    def _cheating(self, card: Card):
        if (self.status["trick_number"] == 0) and (CARD_2P in self.players[0].cards) and not (card == CARD_2P):
            return True
        if len(self.status["trick_cards_played"]) > 0:
            naipe = self.status["trick_cards_played"][0].naipe
            if card.naipe != naipe and naipe in {c.naipe for c in self.players[0].cards}:
                return True
        else:
            if not self.status['hearts_broken'] and \
                    (card.naipe == 'C' or card == CARD_DE) and \
                    len({c for c in self.players[0].cards if c.naipe != 'C' or not (c == CARD_2P)}) > 0:
                return True
        return False

    def _get_points(self):
        winner_i = 0
        for i in range(1, 4):
            if self.status["trick_cards_played"][i].higher_than(self.status["trick_cards_played"][winner_i]):
                winner_i = i

        points = len([c for c in self.status["trick_cards_played"] if c.naipe == "C"])
        if CARD_DE in self.status["trick_cards_played"]:
            points += CARD_DE_POINTS
        winner = (self.first_player + winner_i) % 4
        return winner, points

    def _play(self, player_i: int):
        if len(self.players[player_i].cards) == 0:
            print("HAND_POINTS {}, STATUS {}".
                  format(self.hand_points, self.status))
            for p in self.players:
                print("    Player name {}, Cards {}".
                      format(p.name, p.cards))
        card = self.players[player_i].play_card(self.status)
        self.players[player_i].cards.remove(card)
        self.status["trick_cards_played"].append(card)
        self.status['hearts_broken'] = card == CARD_DE or card.naipe == "C"
        self.game_status[CARD_LIST.index(card)] = PLAYED

    def _others_play(self):
        n_cards_played = len(self.status["trick_cards_played"])
        player_i = 1
        for i in range(n_cards_played, 4):
            self._play(player_i)
            player_i += 1
        self.first_player, points = self._get_points()
        self.hand_points += points
        self.players_points[self.first_player] += points
        '''
        global t_episodes
        if t_episodes > 5000 and points != 0:
            print("EPISODE {}, WINNER {}, REWARD {}, HAND_POINTS {}".
                  format(t_episodes,self.players[self.first_player].name,points,self.hand_points))
            print(self.status)
        '''
        self.status['trick_number'] += 1
        self.status["trick_cards_played"] = []
        if self.hand_points != MAX_HAND_POINTS:
            if self.first_player != 0:
                for player_i in range(self.first_player, 4):
                    self._play(player_i)
        return self.first_player, points

    def _mask_actions(self, cards: set):
        action_mask = 4 * HAND_SIZE * [0]
        for card in cards:
            action_mask[CARD_LIST.index(card)] = 1
        return np.array(action_mask)

    def _valid_actions(self):
        if len(self.status["trick_cards_played"]) == 0:
            if not self.status["hearts_broken"]:
                possible_cards = {c for c in self.players[0].cards if c.naipe != "C"} - {CARD_DE}
                if len(possible_cards) > 0:
                    return self._mask_actions(possible_cards)

            return self._mask_actions(self.players[0].cards)
        else:
            possible_cards = {c for c in self.players[0].cards if c.naipe == self.status["trick_cards_played"][0].naipe}
            if len(possible_cards) > 0:
                return self._mask_actions(possible_cards)

            if self.status["trick_number"] == 0:
                non_hearts = {c for c in self.players[0].cards if c.naipe != "C"} - {CARD_DE}
                if len(non_hearts) > 0:
                    return self._mask_actions(non_hearts)

            return self._mask_actions(self.players[0].cards)

    def reset(self):
        global t_episodes
        t_episodes += 1
        deck = CARD_SET.copy()
        self.game_status = np.array(4 * HAND_SIZE * [OTHERS_HAND])
        self.players = [RandomPlayer("ME"), RandomPlayer("P2"), RandomPlayer("P3"), RandomPlayer("P4")]
        self.hand_points = 0
        self.players_points = 4 * [0]
        self.status = {"hearts_broken": False,
                       "trick_number": 0,
                       "trick_cards_played": []}
        for p in self.players:
            for i in range(CARDS_PER_PLAYER):
                c = random.sample(deck, 1)[0]
                deck.remove(c)
                p.cards.add(c)
                i = CARD_LIST.index(c)
                if p.name == "ME":
                    self.game_status[i] = MY_HAND
                else:
                    self.game_status[i] = OTHERS_HAND
        self.first_player = 0
        for p in self.players:
            if CARD_2P in p.cards:
                self.first_player = self.players.index(p)
                break
        if self.first_player != 0:
            for player_i in range(self.first_player, 4):
                self._play(player_i)
            action_mask = self._valid_actions()
        else:
            action_mask = np.array(4 * HAND_SIZE * [0])
            action_mask[CARD_LIST.index(CARD_2P)] = 1
        return {'status': self.game_status, "action_mask": action_mask}

    def step(self, action: int):

        card = CARD_LIST[action]
        self.game_status[action] = PLAYED

        self.players[0].cards.remove(card)

        self.status["trick_cards_played"].append(card)
        self.status['hearts_broken'] = card == CARD_DE or card.naipe == "C"

        winner_i, reward = self._others_play()
        done = self.hand_points == MAX_HAND_POINTS
        if winner_i != 0:
            reward = 0
        else:
            reward = - reward

        return {'status': self.game_status, "action_mask": self._valid_actions()}, reward, done, {}


class ExternalHearts(ExternalEnv):
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


class ParametricActionsModel(DistributionalQTFModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        print("####### obs_space {}".format(obs_space))
        raise Exception("END")

        self.action_param_model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs,
            model_config, name + "_action_param")
        self.register_variables(self.action_param_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_param, _ = self.action_param_model({
            "obs": input_dict["obs"]["status"]
        })

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_param + inf_mask, state

    def value_function(self):
        return self.action_param_model.value_function()


dqn_config = {
    "v_min": -MAX_HAND_POINTS,
    "v_max": MAX_HAND_POINTS,
    "exploration_config": {
        "epsilon_timesteps": 1000,
    },
    "hiddens": [3 * 4 * HAND_SIZE],
    "learning_starts": 500,
    "timesteps_per_iteration": 1000
}

if __name__ == "__main__":
    ray.init()

    register_env(
        "ExternalHearts",
        lambda _: ExternalHearts(HeartsEnv(), episodes=200000)
    )

    ModelCatalog.register_custom_model("ParametricActionsModel", ParametricActionsModel)

    ppo_config = {"timesteps_per_iteration": 1000,
                  "model": {"custom_model": "ParametricActionsModel"}
                  }

    trainer = PPOTrainer(
        env="ExternalHearts",
        config=ppo_config
    )

    i = 1
    while True:
        result = trainer.train()
        print("Iteration {}, Episodes {}, Mean Reward {}, Mean Length {}".format(
            i, result['episodes_this_iter'], result['episode_reward_mean'], result['episode_len_mean']
        ))
        i += 1

    ray.shutdown()