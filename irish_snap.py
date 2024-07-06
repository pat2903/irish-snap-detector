from collections import deque

# Snap Rules:
# 1. Basic snap: consecutive matching cards
# 2. Top-Bottom snap: if the card played matches the card at the bottom of the pile
# 3. Sandwich: card played matches the one two cards beneath it
# 4. Consecutive cards: in ascending or descending order
# 5. Joker snap
# 6. Three of the same suit
# 7. Same value as the number they say
# 8. Cards add to 10 

class IrishSnap:
    def __init__(self) -> None:
        self.pile = deque()