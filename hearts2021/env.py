import random
from copy import deepcopy
import numpy as np
import gym
from gym.spaces import Discrete, Box, Tuple, Dict

import card
from card import CARD_DE, CARD_NULL
from basicplayers import RandomPlayer

'''
Gym Environment Used to Train a DRL Agent
A Random Play opponent will be first implemented
'''
N_PLAYERS = 4
HAND_SIZE = 5
CHEAT_POINTS = 2 * HAND_SIZE
MAX_POINTS = 2 * HAND_SIZE
CARD_DE_POINTS = HAND_SIZE

CARD_SET = [card.Card(naipe, number) for naipe in ["P", "O", "E", "C"] for number in
            range(card.ACE - HAND_SIZE + 1, card.ACE + 1)]


class HeartsEnv(gym.Env):

    def __init__(self, other_players: list, hand_size=13):
        self.me = RandomPlayer("ME")
        self.players = [self.me]+other_players
        self.hand_size = hand_size

    def _update_status(self, player_i, played_card):
        self.player_list[player_i].cards.remove(played_card)
        self.status["trick_cards_played"].append(played_card)
        self.observation[-1][player_i] = (self.players.index(self.player_list[player_i]), played_card)
        # self.observation[self.status["trick_number"]][player_i] = (self.players.index(self.player_list[player_i]),
        #                                                            played_card)

    def _start_trick(self, first_player):
        self.status["trick_cards_played"] = []
        fpi = self.players.index(first_player)
        self.player_list = [self.players[i] for i in range(fpi, N_PLAYERS)]
        self.player_list += [self.players[i] for i in range(0, fpi)]
        i = 0
        while self.player_list[i] != self.me:
            played_card = self.player_list[i].play_card(self.status.copy())
            self._update_status(i, played_card)
            i += 1
        self.idx_plist = i  # To be used when the AI Agent Plays
        return (self.observation, self.player_list[self.idx_plist].get_possible_cards(self.status))

    def reset(self):
        self.status = {"hand_points"       : {p.name: 0 for p in self.players},
                       "hand_tricks"       : [],
                       "hearts_broken"     : False,
                       "trick_number"      : 0,
                       "trick_cards_played": [],
                       "trick_players"     : []}
        # A list of lists containing the tricks and for each trick a dict withe
        # The pair (None, CARD_NULL) corresponds to (player, card)
        self.observation = [[(None, CARD_NULL) for j in range(N_PLAYERS)] for i in range(self.hand_size)]
        deck = CARD_SET.copy()
        for p in self.players:
            for i in range(self.hand_size):
                c = random.sample(deck, 1)[0]
                deck.remove(c)
                p.cards.add(c)
            # p.player_list = player_list
        first_player = None
        for p in self.players:
            if card.CARD_2P in p.cards:
                first_player = p
                break
        if first_player == self.me:
            fpi = self.players.index(first_player)
            self.player_list = [self.players[i] for i in range(fpi, N_PLAYERS)]
            self.player_list += [self.players[i] for i in range(0, fpi)]
            self.idx_plist = 0
            obs, _, _, _ = self.step(card.CARD_2P)
            return obs
        return self._start_trick(first_player)

    def step(self, played_card: card.Card):
        # Test Cheating
        if played_card not in self.player_list[self.idx_plist].get_possible_cards(self.status):
            return self.observation, CHEAT_POINTS, True, {}

        self._update_status(self.idx_plist, played_card)

        self.idx_plist += 1
        while self.idx_plist < N_PLAYERS:
            played_card = self.player_list[self.idx_plist].play_card(self.status.copy())
            self._update_status(self.idx_plist, played_card)
            self.idx_plist += 1

        winner_i = 0
        for i in range(1, N_PLAYERS):
            if self.status["trick_cards_played"][i].higher_than(self.status["trick_cards_played"][winner_i]):
                winner_i = i

        points = len([c for c in self.status["trick_cards_played"] if c.naipe == "C"])
        if CARD_DE in self.status["trick_cards_played"]:
            points += CARD_DE_POINTS

        self.status["hand_points"][self.player_list[winner_i].name] += points

        if self.player_list[winner_i] == self.me:
            obs_points = -points
        else:
            obs_points = points

        if not self.status["hearts_broken"]:
            self.status["hearts_broken"] = len([c for c in self.status["trick_cards_played"] if c.naipe == "C" or c
                                                == CARD_DE])
        done = sum(point for point in self.status["hand_points"].values()) == MAX_POINTS

        if not done:
            self.status["trick_number"] += 1
            self.observation[-self.status["trick_number"]-1:-1] = self.observation[-self.status["trick_number"]:]
            self.observation[-1] = [(None, CARD_NULL) for j in range(N_PLAYERS)]
            obs = self._start_trick(self.player_list[winner_i])
        else:
            obs = (self.observation, self.me.get_possible_cards(self.status))

        return obs, obs_points, done, {}

''' Environment Test
he = HeartsEnv(other_players=[RandomPlayer("P1"), RandomPlayer("P2"), RandomPlayer("P3")], hand_size=13)

obs, possible_cards = he.reset()
done = False
while not done:
    #print(he.players[0].cards)
    #print(he.players[1].cards)
    #print(he.players[2].cards)
    #print(he.players[3].cards)

    print(obs)
    # print(possible_cards)
    card = random.choice(list(possible_cards))
    print(card)
    (obs, possible_cards), points, done, _ = he.step(card)
    print(points, he.status["hand_points"], done)
'''

# Simple Environment to test Alpha Trainer
class HeartsEnv0(gym.Env):

    def __init__(self, n_cards):
        super(HeartsEnv0, self).__init__()
        self.n_cards = n_cards
        self.table_card = None
        self.me = []

    def reset(self):
        self.deck = CARD_SET.copy()
        c = random.sample(self.deck, 1)[0]
        self.deck.remove(c)
        self.table_card = c
        self.me = []
        for i in range(self.n_cards):
            c = random.sample(self.deck, 1)[0]
            self.deck.remove(c)
            self.me.append(c)
        return (self.table_card, self.me)

    def step(self, played_card):
        points = 0
        if played_card not in self.me:
            return (self.table_card, self.me), -50, True, {}
        if self.table_card.naipe in {c.naipe for c in self.me} and played_card.naipe != self.table_card.naipe:
            points = -20
        else:
            if played_card.naipe == "C" and self.table_card.naipe == "C":
                if played_card.number >= self.table_card.number:
                    points = -2
                else:
                    points = 2
            elif played_card.naipe == self.table_card.naipe:
                if played_card.number >= self.table_card.number:
                    points = -1
                else:
                    points = 1
            elif played_card.naipe == "C" or self.table_card.naipe == "C":
                points = 1
        self.me.remove(played_card)
        done = len(self.me) == 0
        if not done:
            c = random.sample(self.deck, 1)[0]
            self.deck.remove(c)
            self.table_card = c
        else:
            self.table_card = CARD_NULL
        return (self.table_card, self.me), points, done, {}
'''
he = HeartsEnv0(10)
obs = he.reset()
done = False
while not done:
    c = random.sample(he.me, 1)[0]
    print(he.me, he.table_card)
    obs, points, done, _ = he.step(c)
    print("Card {}, Points {}".format(c,points))
'''
# TRUE_OBSERVATION_SPACE = Box(0,1,shape=(4*HAND_SIZE,))
TRUE_OBSERVATION_SPACE = Box(low=0, high=4*HAND_SIZE-1, shape=(1,))

class HeartsParametricEnv:

    def __init__(self, n_cards):
        self.env = HeartsEnv0(n_cards)
        self.action_space = Discrete(4*HAND_SIZE)
        self.observation_space = Dict({
            "obs": Box(low=0, high=4*HAND_SIZE-1, shape=(1,)), #We are going to create an embbedd
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n, ))
        })

    def _get_mask(self, possible_cards):
        mask = np.zeros(self.action_space.n)
        for c in possible_cards:
            i = CARD_SET.index(c)
            mask[i] = 1
        return mask

    def _encode_card(self,c):
        if c == CARD_NULL:
            return 0
        return [CARD_SET.index(c)]

    def _decode_card(self, i):
        return CARD_SET[i]

    def reset(self):
        (table_card, possible_cards) = self.env.reset()
        return {"obs": self._encode_card(table_card), "action_mask": self._get_mask(possible_cards)}

    def step(self, action):
        c = self._decode_card(action)
        (table_card, possible_cards), rew, done, info = self.env.step(c)
        return {"obs": self._encode_card(table_card), "action_mask": self._get_mask(possible_cards)}, rew, done, info

class HeartsAlphaEnv(HeartsParametricEnv):

    def __init__(self, n_cards):
        super(HeartsAlphaEnv, self).__init__(n_cards)
        self.running_reward = 0

    def reset(self):
        self.running_reward = 0
        return super(HeartsAlphaEnv, self).reset()

    def step(self, action):
        obs, rew, done, info = super(HeartsAlphaEnv, self).step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return obs, score, done, info

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        return {"obs": self._encode_card(self.env.table_card), "action_mask": self._get_mask(self.env.me)}

    def get_state(self):
        return deepcopy(self.env), self.running_reward


'''
he = HeartsParametricEnv(10)
obs = he.reset()
done = False
while not done:
    c = random.sample(he.env.me, 1)[0]
    print(obs['action_mask'])
    print(he.env.me, he.env.table_card, c)
    obs, points, done, _ = he.step(he._encode_card(c))
    print("POINTS:", points)
'''