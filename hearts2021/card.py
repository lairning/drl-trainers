# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:19:47 2019

@author: mario.duarte
"""

CARD_STR = {0:"0", 2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"10", 11:"V",12:"D",13:"R",14:"A"}
STR_CARD = {v:k for k,v in CARD_STR.items()}

# MAX_POINTS = 100
# CHEAT_POINTS = MAX_POINTS

class Card:
    def __init__(self,naipe, number):
        self.naipe = naipe
        self.number = number

    def __repr__(self):
        return CARD_STR[self.number] + str(self.naipe)
    
    def __eq__(self,other):
        if other == None: return False
        return self.naipe == other.naipe and self.number == other.number
    
    def __hash__(self):
        return hash((self.naipe, self.number))
    
    def higher_than(self,other):
        if other == None: return True
        return self.naipe == other.naipe and self.number > other.number
    
    def is_heart(self):
        return self.naipe == 'E'

CLUBS, DIAMONDS, SPADES, HEARTS, NONE = "P","O","E","C", "N"
JACK, QUEEN, KING, ACE = 11, 12, 13, 14

CARD_DE = Card("E",12)
CARD_RE = Card("E",13)
CARD_AE = Card("E",14)
CARD_2P = Card("P",2)
CARD_NULL = Card("N",0)

