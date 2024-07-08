from constants import cards

class Card:
    def __init__(self, card_string: str):
        if card_string == "joker":
            self.value = 0
            self.suit = None
        else:
            # format is e.g. "ace of clubs"
            tmp_value, self.suit = card_string.split(" of ")
            self.value = self.parse_value(tmp_value)
    
    def parse_value(self, value: str) -> int:
        """
        Returns the numerical value of the card
        
        Assumptions: Jack=11, Queen=12 and King=13.

        """
        value_map = {
            "jack": 11,
            "queen": 12,
            "king": 13,
            "ace": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10
        }
        return value_map.get(value, 0)
    
    def __eq__(self, other_card) -> bool:
        """
        Returns true if the suit and value are the same
        """
        if isinstance(other_card, Card):
            return self.value == other_card.value and self.suit == other_card.suit
        return False
    
    def equal_value(self, other_card) -> bool:
        """
        Returns true if the values are the same. Does not take into the account the suit.
        """
        if isinstance(other_card, Card):
            return self.value == other_card.value
        return False
    