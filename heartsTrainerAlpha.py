# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:00:16 2020

@author: mario.duarte
"""

import random
import numpy as np
from copy import deepcopy

import gym
from gym.spaces import Discrete, Tuple, Box, Dict

import ray
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray import tune

from ray.rllib.env.external_env import ExternalEnv
from ray.tune.registry import register_env

from heartsBasicPlayers import BasicPlayer, RandomPlayer
from card import CARD_2P, CARD_DE, Card, CARD_SET, CARD_LIST, CARDS_PER_PLAYER, HAND_SIZE

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
            "obs": Tuple(observation_tuple),
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n,))
        })
        self.game_status = np.array(4 * HAND_SIZE * [OTHERS_HAND])
        self.players = None
        self.hand_points = 0
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
        return {'obs': self.game_status, "action_mask": action_mask}

    def step(self, action: int):

        card = CARD_LIST[action]
        self.game_status[action] = PLAYED

        if card not in self.players[0].cards:
            return self.game_status, CHEAT_POINTS + self.players_points[0], True, {}

        self.players[0].cards.remove(card)

        if self._cheating(card):
            return self.game_status, CHEAT_POINTS + self.players_points[0], True, {}

        self.status["trick_cards_played"].append(card)
        self.status['hearts_broken'] = card == CARD_DE or card.naipe == "C"

        winner_i, reward = self._others_play()
        done = self.hand_points == MAX_HAND_POINTS
        if winner_i != 0:
            reward = 0
        else:
            reward = - reward

        return {'obs': self.game_status, "action_mask": self._valid_actions()}, reward, done, {}



class HeartsEnvWrapper:
    def __init__(self):
        self.env = HeartsEnv()
        self.action_space = Discrete(4 * HAND_SIZE)
        observation_tuple = tuple(Discrete(3) for _ in range(4 * HAND_SIZE))
        self.observation_space = Dict({
            "obs": Tuple(observation_tuple),
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n,))
        })
        self.running_reward = 0

    def reset(self):
        self.running_reward = 0
        return self.env.reset()

    def step(self, action: int):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return obs, score, done, info

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        return {'obs': self.env.game_status, "action_mask": self.env._valid_actions()}

    def get_state(self):
        return deepcopy(self.env), self.running_reward



config = {"timesteps_per_iteration": 1000}

if __name__ == "__main__":
    ray.init()

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    config = {
        "env": HeartsEnvWrapper,
        "timesteps_per_iteration": 1000,
        "model": {
            "custom_model": "dense_model"
        },
        "ranked_rewards": {
            "enable": True,
        },
    }

    stop = {
        "training_iteration": 100
    }

    results = tune.run(AlphaZeroTrainer, config=config, stop=stop)

    ray.shutdown()

'''
if __name__ == "__main__":
    ray.init()

    register_env(
        "HeartsEnv",
        lambda _: HeartsEnvWrapper()
    )

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    trainer = AlphaZeroTrainer(
        env="HeartsEnv",
        config= {
            "timesteps_per_iteration": 1000,
            "model": {"custom_model": "dense_model"}
        }
    )

    i = 1
    while True:
        result = trainer.train()
        print("Iteration {}, Episodes {}, Mean Reward {}, Mean Length {}".format(
            i, result['episodes_this_iter'], result['episode_reward_mean'], result['episode_len_mean']
        ))
        i += 1

    ray.shutdown()
'''