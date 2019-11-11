# Ultimate Tic Tac Toe (UTTT)
Applying the deep learning techniques from ALpha Go to play Ultimate tic-tac-toe.

UTTT does *not* have a clear heuristic of the game state, contrary to for example chess (e.g. counting the pieces).
A neural network is trained to predict the value of a given state of the game.
Monte carlo tree search is used on top of this to improve the engine.


By Lennart and [Mark Boon](https://github.com/Mark-Boon)

[Forked from repo of Daniel Slater](https://github.com/DanielSlater/AlphaToe)

## What is ultimate tic-tac-toe?

<img src="https://upload.wikimedia.org/wikipedia/commons/d/d1/Incomplete_Ultimate_Tic-Tac-Toe_Board.png" width="300">

Ultimate Tic-Tac-Toe (UTTT) is the next level of the very simple game tic-tac-toe.

Imagine each square of a regular 3x3 tic-tac-toe board is anther game of tic-tac-toe, sounds easy right?

Each of the 9 small 3x3 parts of the board is called a **field**.

When making a move, the position within the field you placed your marker determines in which field your opponent can make a move.

More detailed rules can be found on the [wiki](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe).

We calculated an upper-bound for the state-space complexity of 10^37.
Compared to other games:
- tic-tac-toe: 10^3
- checkers: 10^30
- chess: 10^47
- go: 10^170

Making this an extremely more difficult game than regular tic-tac-toe.

## What did we contribute?
- Implementation of the game mechanics of UTTT
- Option for Convolutional Neural Network (CNN)
- Visualization of training progress and results
- Batch normalization
- Baseline

# Usage
## Running
Execute policy_gradient.py

## Unit tests
Test scripts in the tests folder by running "python -m unittest tests\games\test_uttt.py"

To make sure we have the same performance during hyperparameter tuning, all random generators are seeded:
- policy_gradient.py:22 (random)
- train_policy_gradient.py:10 (tensorflow)
- network_helpers.py:6 (numpy
