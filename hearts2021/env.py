import random
from abc import ABC
from copy import deepcopy
import numpy as np
import gym
from gym.spaces import Discrete, Box, Tuple, Dict

import card
from card import CARD_DE, CARD_NULL, CARD_2P, Card
from basicplayers import RandomPlayer, BasicPlayer

'''
Gym Environment Used to Train a DRL Agent
A Random Play opponent will be first implemented
'''
N_PLAYERS = 4
HAND_SIZE = 8
CHEAT_POINTS = 2 * HAND_SIZE
MAX_POINTS = 2 * HAND_SIZE
CARD_DE_POINTS = HAND_SIZE

CARD_SET = [card.Card(naipe, number) for naipe in ["P", "O", "E", "C"] for number in
            range(card.ACE - HAND_SIZE + 2, card.ACE + 1)]
CARD_SET += [card.Card(naipe, 2) for naipe in ["P", "O", "E", "C"]]


class HeartsEnv(gym.Env):

    def __init__(self, other_players: list, hand_size):
        self.me = RandomPlayer("ME")
        self.card_set = [CARD_NULL] + CARD_SET
        self.players = [self.me] + other_players
        self.hand_size = hand_size
        self.idx_plist = None
        self.status = None
        self.observation = None
        self.player_list = None

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
        return self.observation, self.player_list[self.idx_plist].get_possible_cards(self.status)

    # @property
    def reset(self):
        self.status = {"hand_points"       : {p.name: 0 for p in self.players},
                       "hand_tricks"       : [],
                       "hearts_broken"     : False,
                       "trick_number"      : 0,
                       "trick_cards_played": [],
                       "trick_players"     : []}
        # A list of lists containing the tricks and for each trick a dict withe
        # The pair (None, CARD_NULL) corresponds to (player, card)
        self.observation = [[(None, CARD_NULL) for _ in range(N_PLAYERS)] for _ in range(self.hand_size)]
        deck = CARD_SET.copy()
        for p in self.players:
            for i in range(self.hand_size):
                c = random.sample(deck, 1)[0]
                deck.remove(c)
                p.cards.add(c)
            # p.player_list = player_list
        first_player = None
        for p in self.players:
            if CARD_2P in p.cards:
                first_player = p
                break
        if first_player == self.me:
            fpi = self.players.index(first_player)
            self.player_list = [self.players[i] for i in range(fpi, N_PLAYERS)]
            self.player_list += [self.players[i] for i in range(0, fpi)]
            self.idx_plist = 0
            (_, possible_cards_reset), _, _, _ = self.step(CARD_2P)
            return self.observation, possible_cards_reset
        obs_step1 = self._start_trick(first_player)
        return obs_step1

    def step(self, played_card: Card):
        # Test Cheating
        if played_card not in self.player_list[self.idx_plist].get_possible_cards(self.status):
            print("DEBUG1", self.observation[-1], self.me.cards, played_card)
            return (self.observation, set()), CHEAT_POINTS, True, {}

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

        step_points = len([c for c in self.status["trick_cards_played"] if c.naipe == "C"])
        if CARD_DE in self.status["trick_cards_played"]:
            step_points += CARD_DE_POINTS

        self.status["hand_points"][self.player_list[winner_i].name] += step_points

        if self.player_list[winner_i] == self.me:
            obs_points = -step_points
        else:
            obs_points = step_points / 3

        if not self.status["hearts_broken"]:
            self.status["hearts_broken"] = len([c for c in self.status["trick_cards_played"] if c.naipe == "C" or c
                                                == CARD_DE])
        step_done = sum(point for point in self.status["hand_points"].values()) == MAX_POINTS

        if not step_done:
            self.status["trick_number"] += 1
            self.observation[-self.status["trick_number"] - 1:-1] = self.observation[-self.status["trick_number"]:]
            self.observation[-1] = [(None, CARD_NULL) for _ in range(N_PLAYERS)]
            obs_step = self._start_trick(self.player_list[winner_i])
            # print("DEBUG1", obs_step)
        else:
            obs_step = (self.observation, self.me.get_possible_cards(self.status))
        # print("DEBUG2", obs_step)
        return obs_step, obs_points, step_done, {}


# Hearts Environment simplification (only the last trick is considered)
class HeartsEnv1(HeartsEnv):

    def __init__(self, other_players: list, hand_size):
        super(HeartsEnv1, self).__init__(other_players, hand_size)

    def reset(self):
        obs, possible_cards = super(HeartsEnv1, self).reset()
        return (obs[-1], possible_cards)

    def step(self, played_card: Card):
        obs_step, obs_points, done, info = super(HeartsEnv1, self).step(played_card)
        obs, possible_cards = obs_step
        return (obs[-1], possible_cards), obs_points, done, info


# Environment Test
'''
for _ in range(100):
    he = HeartsEnv1(other_players=[RandomPlayer("P1"), RandomPlayer("P2"), RandomPlayer("P3")], hand_size=HAND_SIZE)
    obs, possible_cards = he.reset()
    done = False
    while not done:
        # print(he.players[0].cards)
        # print(he.players[1].cards)
        # print(he.players[2].cards)
        # print(he.players[3].cards)
        # print(obs)
        # print(possible_cards)
        card = random.choice(list(possible_cards))
        # print(card)
        (obs, possible_cards), points, done, _ = he.step(card)
        print(points, he.status["hand_points"], done)

'''
low = np.array([0] * 4 * N_PLAYERS, dtype=np.float32)
high = np.array([1] * 3 * N_PLAYERS + [len(CARD_SET)] * N_PLAYERS, dtype=np.float32)
TRUE_OBSERVATION_SPACE1 = Box(low=low, high=high, dtype=np.float32)
# players_space = Box(low=0, high=1, shape=(3 * N_PLAYERS,))
# cards_space = Box(low=0, high=len(CARD_SET), shape=(N_PLAYERS,))
# TRUE_OBSERVATION_SPACE1 = Dict({'players': players_space, 'cards': cards_space})


class HeartsParametricEnv1:

    def __init__(self, random_players=True):
        if random_players:
            other_players = [RandomPlayer("P1"), RandomPlayer("P2"), RandomPlayer("P3")]
        else:
            other_players = [BasicPlayer("P1"), BasicPlayer("P2"), BasicPlayer("P3")]

        self.env = HeartsEnv1(other_players=other_players, hand_size=HAND_SIZE)
        self.card_set = [CARD_NULL] + CARD_SET
        self.action_space = Discrete(4 * HAND_SIZE)
        self.observation_space = Dict({
            "obs"        : TRUE_OBSERVATION_SPACE1,
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n,))
        })

    def _get_mask(self, possible_cards):
        mask = np.zeros(self.action_space.n)
        for c in possible_cards:
            i = CARD_SET.index(c)
            mask[i] = 1
        return mask

    def _encode_observation(self, table_cards):
        def _encode_card(card_set, c: Card):
            return card_set.index(c)

        def _encode_player(player_i: int):
            player_encode = [0] * 3
            if player_i:
                player_encode[player_i - 1] = 1
            return player_encode

        return [n for player, _ in table_cards for n in _encode_player(player)] + \
                [_encode_card(self.card_set, tc) for _, tc in table_cards]

    def _decode_card(self, i):
        return CARD_SET[i]

    def reset(self):
        table_cards, possible_cards = self.env.reset()
        print()
        print("### DEBUG 11", {p.name: p.cards for p in self.env.players})
        print("DEBUG 01", table_cards, possible_cards)
        obs = self._encode_observation(table_cards)
        # print("DEBUG 02", obs)
        assert self.observation_space['obs'].contains(obs), "{} not in {}".format(obs, self.observation_space['obs'])
        return {"obs": obs, "action_mask": self._get_mask(possible_cards)}

    def step(self, action):
        c = self._decode_card(action)
        print("DEBUG 02", c)
        print("### DEBUG 21", {p.name: p.cards for p in self.env.players})
        (table_cards, possible_cards), rew, done, info = self.env.step(c)
        print("DEBUG 03", table_cards, possible_cards)
        obs = self._encode_observation(table_cards)
        # print("DEBUG 12", obs)
        assert self.observation_space['obs'].contains(obs), "{} not in {}".format(obs, self.observation_space['obs'])
        return {"obs": obs, "action_mask": self._get_mask(possible_cards)}, rew, \
               done, info

'''
import torch
from torch.nn import Embedding

CARD_EMBEDD_SIZE = 3
#emb = Embedding(int(TRUE_OBSERVATION_SPACE1['cards'].high[0])+1, CARD_EMBEDD_SIZE)

N = 1
total_points = 0
for _ in range(N):
    he = HeartsParametricEnv1(random_players=True)
    obs = he.reset()

    # print("DEBUG 20:", obs['obs']['players'], obs['obs']['cards'], )
    # emb_cards = emb(torch.LongTensor(obs['obs']['cards'])).view(-1)
    # t_players = torch.Tensor(obs['obs']['players'])
    # print("DEBUG 21:", t_players, emb_cards)
    # print("DEBUG 22:", torch.cat((t_players,emb_cards)).shape)
    done = False
    while not done:
        possible_idx = [i for i, v in enumerate(obs['action_mask']) if v]
        ci = random.sample(possible_idx, 1)[0]
        # print([(play[0],he.card_set[play[1][0]]) for play in obs['obs']])
        # print([play[1] for play in obs['obs']])
        # print([CARD_SET[i] for i in possible_idx], CARD_SET[ci])
        obs, points, done, _ = he.step(ci)
        total_points += points
print("POINTS:", total_points / N)

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
TRUE_OBSERVATION_SPACE = Box(low=0, high=4 * HAND_SIZE, shape=(1,))


class HeartsParametricEnv:

    def __init__(self, n_cards):
        self.env = HeartsEnv0(n_cards)
        self.card_set = [CARD_NULL] + CARD_SET
        self.action_space = Discrete(4 * HAND_SIZE)
        self.observation_space = Dict({
            "obs"        : Box(low=0, high=len(self.card_set) - 1, shape=(1,)),  # We are going to create an embbedd
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n,))
        })

    def _get_mask(self, possible_cards):
        mask = np.zeros(self.action_space.n)
        for c in possible_cards:
            i = CARD_SET.index(c)
            mask[i] = 1
        return mask

    def _encode_card(self, c):
        return np.array([self.card_set.index(c)])

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
