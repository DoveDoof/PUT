"""
Builds and trains a neural network that uses policy gradients to learn to play Tic-Tac-Toe.

The input to the network is a vector with a number for each space on the board. If the space has one of the networks
pieces then the input vector has the value 1. -1 for the opponents space and 0 for no piece.

The output of the network is a also of the size of the board with each number learning the probability that a move in
that space is the best move.

The network plays successive games randomly alternating between going first and second against an opponent that makes
moves by randomly selecting a free space. The neural network does NOT initially have any way of knowing what is or is not
a valid move, so initially it must learn the rules of the game.

I have trained this version with success at 3x3 tic tac toe until it has a success rate in the region of 75% this maybe
as good as it can do, because 3x3 tic-tac-toe is a theoretical draw, so the random opponent will often get lucky and
force a draw.
"""
import functools
import pdb
import common.visualisation as plt
import random
random.seed(1)

from common.network_helpers import create_network
from games.uttt import UltimateTicTacToeGameSpec
from techniques.train_policy_gradient import train_policy_gradients

# BATCH_SIZE = 100  # every how many games to do a parameter update?
# LEARN_RATE = 1e-3
# PRINT_RESULTS_EVERY_X = 100  # every how many games to print the results
# NETWORK_FILE_PATH = None#'current_network.p'  # path to save the network to
# NUMBER_OF_GAMES_TO_RUN = 1000
# HIDDEN_LAYERS = (150, 150, 150)

BATCH_SIZE = 100 # every how many games to do a parameter update?
LEARN_RATE = 1e-3
PRINT_RESULTS_EVERY_X = 500 # every how many games to print the results
save_network_file_path = 'networks/ep-3_81_81_81/net.p' # path to save a network file
NETWORK_FILE_PATH = None # path to load a network file (change to above variable to continue)
NUMBER_OF_GAMES_TO_RUN = 5000
HIDDEN_LAYERS = (81, 81, 81)


# to play a different game change this to another spec, e.g TicTacToeXGameSpec or ConnectXGameSpec, to get these to run
# well may require tuning the hyper parameters a bit
game_spec = UltimateTicTacToeGameSpec()

input_layer = game_spec.board_squares()
output_layer = game_spec.outputs()

create_network_func = functools.partial(create_network, input_layer, HIDDEN_LAYERS, output_layer)

res = train_policy_gradients(game_spec, create_network_func, NETWORK_FILE_PATH,
							 number_of_games=NUMBER_OF_GAMES_TO_RUN,
							 batch_size=BATCH_SIZE,
							 learn_rate=LEARN_RATE,
							 print_results_every=PRINT_RESULTS_EVERY_X,
							 save_network_file_path = save_network_file_path)


parameters = {	'batch_size':BATCH_SIZE, 
				'learn_rate':LEARN_RATE, 
				'print_results_every':PRINT_RESULTS_EVERY_X,
				'network_file_path':NETWORK_FILE_PATH,
				'number_of_games':NUMBER_OF_GAMES_TO_RUN,
				'input_layer':input_layer,
				'hidden_layers':HIDDEN_LAYERS,
				'output_layer':output_layer,
				'save_network_file_path':save_network_file_path,
				'results':res[2]
				}

plt.save(parameters, save_network_file_path)


# pdb.set_trace()