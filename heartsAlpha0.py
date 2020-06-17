"""Example of using training on CartPole."""

import argparse
from copy import deepcopy
import numpy as np
import random

import ray
from ray import tune
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
# from ray.rllib.contrib.alpha_zero.environments.cartpole import CartPole
from ray.rllib.models.catalog import ModelCatalog

import gym
from gym.spaces import Discrete, Box, Dict

from heartsBasicPlayers import BasicPlayer, RandomPlayer
from card import CARD_2P, CARD_DE, Card

HAND_SIZE = 7

CARD_SET = {Card(naipe, n) for naipe in {"E", "C", "O", "P"} for n in list(range(16 - HAND_SIZE, 15)) + [2]}
CARD_LIST = [Card(naipe, n) for naipe in {"E", "C", "O", "P"} for n in list(range(16 - HAND_SIZE, 15)) + [2]]
CARDS_PER_PLAYER = len(CARD_SET) // 4

# Card Status
OTHERS_HAND = -1
MY_HAND = 0
PLAYED = 1
CURRENT_TRICK = 2

CARD_DE_POINTS = HAND_SIZE
MAX_HAND_POINTS = len([c for c in CARD_SET if c.naipe == "C"]) + CARD_DE_POINTS
CHEAT_POINTS = -MAX_HAND_POINTS

t_episodes = 0

# TRUE_OBSERVATION_SPACE = Tuple(tuple(Discrete(3) for _ in range(4 * HAND_SIZE)))
TRUE_OBSERVATION_SPACE = Box(low=-1, high=2, shape=(4 * HAND_SIZE,))


class HeartsEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self):
        self.action_space = Discrete(4 * HAND_SIZE)
        self.observation_space = Dict({
            "obs": TRUE_OBSERVATION_SPACE,
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
        card = self.players[player_i].play_card(self.status)
        self.players[player_i].cards.remove(card)
        self.status["trick_cards_played"].append(card)
        self.status['hearts_broken'] = card == CARD_DE or card.naipe == "C"

    def _others_play(self):
        n_cards_played = len(self.status["trick_cards_played"])
        player_i = 1
        for i in range(n_cards_played, 4):
            self._play(player_i)
            player_i += 1
        for card in self.status["trick_cards_played"]:
            self.game_status[CARD_LIST.index(card)] = PLAYED
        self.first_player, points = self._get_points()
        self.hand_points += points
        self.players_points[self.first_player] += points
        self.status['trick_number'] += 1
        self.status["trick_cards_played"] = []
        if self.hand_points != MAX_HAND_POINTS:
            if self.first_player != 0:
                for player_i in range(self.first_player, 4):
                    self._play(player_i)
        for card in self.status["trick_cards_played"]:
            self.game_status[CARD_LIST.index(card)] = CURRENT_TRICK

        return self.first_player, points

    def _mask_actions(self, cards: set):
        action_mask = 4 * HAND_SIZE * [0]
        for card in cards:
            action_mask[CARD_LIST.index(card)] = 1
        return np.array(action_mask)

    def valid_actions(self):
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
        self.players = [RandomPlayer("ME"), BasicPlayer("P2"), BasicPlayer("P3"), BasicPlayer("P4")]
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
            action_mask = self.valid_actions()
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
            reward = reward / 3
        else:
            reward = - reward

        return {'status': self.game_status, "action_mask": self.valid_actions()}, reward, done, {}


class HeartsEnvWrapper:
    def __init__(self):
        self.env = HeartsEnv()
        self.action_space = Discrete(4 * HAND_SIZE)
        self.observation_space = Dict({
            "obs": TRUE_OBSERVATION_SPACE,
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
        return {'obs': self.env.game_status, "action_mask": self.env.valid_actions()}

    def get_state(self):
        return deepcopy(self.env), self.running_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=6, type=int)
    parser.add_argument("--training-iteration", default=10000, type=int)
    args = parser.parse_args()
    ray.init()

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": args.training_iteration},
        max_failures=0,
        config={
            "env": HeartsEnvWrapper,
            "num_workers": args.num_workers,
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            "sgd_minibatch_size": 64,
            "lr": 1e-4,
            "num_sgd_iter": 1,
            "mcts_config": {
                "puct_coefficient": 1.5,
                "num_simulations": 100,
                "temperature": 1.0,
                "dirichlet_epsilon": 0.20,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
            },
            "ranked_rewards": {
                "enable": True,
            },
            "model": {
                "custom_model": "dense_model",
            },
        },
    )
