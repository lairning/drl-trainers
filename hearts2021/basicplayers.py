# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:15:54 2019

@author: mario.duarte
"""

from card import CARD_2P, CARD_DE

class BasicPlayer:
    def __init__(self, name):
        self.name = name
        self.cards = set()

    def __repr__(self):
        return self.name

    def play_card(self, hand_status):
        if len(hand_status["trick_cards_played"]) == 0:
            if hand_status["trick_number"] == 0:
                return CARD_2P
            if not hand_status["hearts_broken"]:
                possible_cards = {c for c in self.cards if c.naipe != "C"} - {CARD_DE}
                if len(possible_cards) == 0:
                    possible_cards = self.cards
            else:
                possible_cards = self.cards
            return min(possible_cards, key=lambda c: c.number)
        else:
            possible_cards = {c for c in self.cards if c.naipe == hand_status["trick_cards_played"][0].naipe}
            if len(possible_cards) == 0:
                if hand_status["trick_number"] == 0:
                    non_hearts = {c for c in self.cards if c.naipe != "C"} - {CARD_DE}
                    if len(non_hearts):
                        return max(non_hearts, key=lambda c: c.number)
                    if CARD_DE in self.cards:
                        return CARD_DE
                    return max(self.cards, key=lambda c: c.number)
                else:
                    if CARD_DE in self.cards:
                        return CARD_DE
                    copas = {c for c in self.cards if c.naipe == "C"}
                    if len(copas) > 0:
                        return max(copas, key=lambda c: c.number)
                    return max(self.cards, key=lambda c: c.number)
            else:
                max_card = max([c for c in hand_status["trick_cards_played"] if
                                c.naipe == hand_status["trick_cards_played"][0].naipe], key=lambda c: c.number)
                if hand_status["trick_cards_played"][0].naipe == "E":
                    if CARD_DE in possible_cards:
                        if max_card.number > CARD_DE.number or len(possible_cards) == 1:
                            return CARD_DE
                        else:
                            possible_cards.remove(CARD_DE)
                            return max(possible_cards, key=lambda c: c.number)
                    else:
                        max_card = CARD_DE
                lower_cards = {c for c in possible_cards if c.number < max_card.number}
                if len(lower_cards):
                    return max(lower_cards, key=lambda c: c.number)
                return max(possible_cards, key=lambda c: c.number)


import random


class RandomPlayer:
    def __init__(self, name):
        self.name = name
        self.cards = set()

    def __repr__(self):
        return self.name

    def get_possible_cards(self, hand_status):

        if len(hand_status["trick_cards_played"]) == 0:
            if hand_status["trick_number"] == 0:
                return {CARD_2P}
            if not hand_status["hearts_broken"]:
                possible_cards = {c for c in self.cards if c.naipe != "C"} - {CARD_DE}
                if len(possible_cards) > 0:
                    return possible_cards

            return self.cards
        else:
            possible_cards = {c for c in self.cards if c.naipe == hand_status["trick_cards_played"][0].naipe}
            if len(possible_cards) > 0:
                return possible_cards

            if hand_status["trick_number"] == 0:
                non_hearts = {c for c in self.cards if c.naipe != "C"} - {CARD_DE}
                if len(non_hearts) > 0:
                    return non_hearts

            return self.cards

    # play and train itself
    def play_card(self, hand_status):

        possible_cards = self.get_possible_cards(hand_status)

        if len(possible_cards) == 0:
            print(self.name, self.cards)
            raise Exception()

        return random.choice(list(possible_cards))
