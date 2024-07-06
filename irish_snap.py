from collections import deque
from card import Card

# Snap Rules:
# 1. Basic snap: consecutive matching cards
# 2. Top-Bottom snap: if the card played matches the card at the bottom of the pile
# 3. Sandwich: card played matches the one two cards beneath it
# 4. Consecutive cards: in ascending or descending order
# 5. Joker snap
# 6. Three of the same suit
# 7. Same value as the number they say (goes 1-10, J, Q, K doesn't count)
# 8. Cards add to 10 

class IrishSnap:
    def __init__(self) -> None:
        self.pile = deque()
        self.last_called_number = 0
    
    def play_card(self, card_string: str) -> bool:
        new_card = Card(card_string)
        self.pile.append(new_card)
        self.last_called_number = (self.last_called_number % 10) + 1
        return self.check_snap(new_card)
    
    def check_snap(self, new_card: Card) -> bool:
        if len(self.pile) < 2:
            return False
        
        return (
            self.basic_snap(new_card) or
            self.top_bottom_snap(new_card) or
            self.sandwich_snap(new_card) or
            self.consecutive_cards_snap(new_card) or
            self.joker_snap(new_card) or
            self.three_same_suit_snap(new_card) or
            self.same_value_as_called_snap(new_card) or
            self.cards_add_to_ten_snap(new_card)
        )
    
    def basic_snap(self, new_card: Card) -> bool:
        return new_card.__eq__(self.pile[-2])
    
    def top_bottom_snap(self, new_card: Card) -> bool:
        return new_card.__eq__(self.pile[0])
    
    def sandwich_snap(self, new_card: Card) -> bool:
        return len(self.pile) > 2 and new_card.__eq__(self.pile[-3])
    
    def consecutive_cards_snap(self, new_card: Card) -> bool:
        #edge case to consider: king -> ace
        prev_card = self.pile[-2]
        diff = (new_card.value - prev_card.value) % 13
        return diff == 1 or diff == 12
    
    def joker_snap(self, new_card: Card) -> bool:
        return new_card.value == 0
    
    def three_same_suit_snap(self, new_card: Card) -> bool:
        if len(self.pile) > 2:
            return new_card.suit and new_card.suit == self.pile[-2].suit == self.pile[-3].suit
        return False
    
    def same_value_as_called_snap(self, new_card: Card) -> bool:
        return new_card.value == self.last_called_number
    
    def cards_add_to_ten_snap(self, new_card: Card) -> bool:
        return new_card.value + self.pile[-2].value == 10