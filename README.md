# UTTT
Applying the deep learning techniques from to play ultimate tic-tac-toe

Original repo by Daniel Slater: https://github.com/DanielSlater/AlphaToe

Then run the file file policy_gradient.py

## testing
Test scripts in the tests folder by running "python -m unittest tests\games\test_uttt.py"
to make sure we have the same performance during hyperparameter tuning, all random generators are seeded:
policy_gradient.py:22 (random)
train_policy_gradient.py:10 (tensorflow)
network_helpers.py:6 (numpy)