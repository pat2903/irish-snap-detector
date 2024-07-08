# unit test for the snap rules

import unittest
from irish_snap_rules import IrishSnap

class TestIrishSnap(unittest.TestCase):

    def setUp(self):
        self.game = IrishSnap()
    
    def test_play_card(self):
        result = self.game.play_card("five of clubs")
        self.assertFalse(result)
        self.assertEqual(len(self.game.pile), 1)
        self.assertEqual(self.game.last_called_number, 1)
    
    def test_basic_snap(self):
        self.game.play_card("ace of spades")
        result = self.game.play_card("ace of hearts")
        self.assertTrue(result)

    def test_top_bottom_snap(self):
        self.game.play_card("two of clubs")
        self.game.play_card("five of diamonds")
        result = self.game.play_card("two of diamonds")
        self.assertTrue(result)

    def test_sandwich_snap(self):
        self.game.play_card("three of clubs")
        self.game.play_card("five of diamonds")
        result = self.game.play_card("three of diamonds")
        self.assertTrue(result)

    def test_consecutive_cards_snap_ascending(self):
        self.game.play_card("five of clubs")
        result = self.game.play_card("six of hearts")
        self.assertTrue(result)

    def test_consecutive_cards_snap_descending(self):
        self.game.play_card("six of hearts")
        result = self.game.play_card("five of clubs")
        self.assertTrue(result)

    def test_consecutive_cards_snap_edge_case(self):
        self.game.play_card("king of clubs")
        result = self.game.play_card("ace of hearts")
        self.assertTrue(result)

    def test_joker_snap(self):
        result = self.game.play_card("joker")
        self.assertTrue(result)

    def test_three_same_suit_snap(self):
        self.game.play_card("two of hearts")
        self.game.play_card("five of hearts")
        result = self.game.play_card("eight of hearts")
        self.assertTrue(result)

    def test_same_value_as_called_snap(self):
        self.game.play_card("five of clubs")  
        result = self.game.play_card("two of hearts")  
        self.assertTrue(result)

    def test_cards_add_to_ten_snap(self):
        self.game.play_card("three of clubs")
        result = self.game.play_card("seven of hearts")
        self.assertTrue(result)

    def test_no_snap(self):
        self.game.play_card("two of clubs")
        result = self.game.play_card("five of hearts")
        self.assertFalse(result)
    
    def test_consecutive_cards_snap_queen_king(self):
        self.game.play_card("queen of spades")
        result = self.game.play_card("king of hearts")
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()