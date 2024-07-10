# Irish Snap Detector

## Rules of Irish Snap
A game where players take turns placing cards face-up and race to spot pairs or triplets of cards that correspond to the following rules (Note: these rules vary from person to person-these are the rules I usually play with):

1. Basic Snap: consecutive cards with matching values;
2. Top-Bottom Snap: the card played matches the card at the bottom of the pile;
3. Sandwich Snap: the card played matches the one two cards beneath it;
4. Consecutive Cards Snap: cards are in ascending or descending order;
5. Joker Snap: Any joker played;
6. Suit Triplet: three cards of the same suit;
7. Same value as the number called (goes 1-10, J, Q, K don't count);
8. Cards add to 10. 

Image classification dataset taken from [kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)


Training with EfficientNet B0:
Test accuracy: 0.8164
Test loss: 1.1566

Training with EfficientNet B3:
Test accuracy: 0.9727
Test loss: 0.0878

Training with EfficientNet V2S:
l: 0.9333
Test accuracy: 0.9727
Test loss: 0.5976
Test precision: 0.9722
Test recall: 0.9570
F1 Score: 0.9645