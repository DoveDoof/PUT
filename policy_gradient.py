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
import common.custom_cnn as cnn

from techniques.train_policy_gradient_historic import train_policy_gradients_vs_historic
from common.network_helpers import create_network
from games.uttt import UltimateTicTacToeGameSpec
from techniques.train_policy_gradient import train_policy_gradients


# Import config file
import yaml
with open("_config.yml", 'r') as f:
	config = yaml.load(f)

game_spec = UltimateTicTacToeGameSpec()

input_layer = game_spec.board_squares()
output_layer = game_spec.outputs()

config['input_layer'] = input_layer
config['output_layer'] = output_layer

if config['cnn_on']:
	create_network_func = functools.partial(cnn.create_network,
											config['filter_shape'],
											config['filter_depth'],
											config['dense_width'])
else:
	create_network_func = functools.partial(create_network,
											config['input_layer'],
											config['hidden_layers'],
											config['output_layer'])

if config['historic']:
	print("Historic network is being used")
	res = train_policy_gradients_vs_historic(game_spec,
											 create_network_func,
											 load_network_file_path = config['load_network_file_path'],
											 save_network_file_path = config['save_network_file_path'],
											 number_of_historic_networks = 1,
											 historic_network_base_path = config['historic_network_base_path'],
											 number_of_games = config['number_of_games'],
											 update_opponent_winrate = config['update_opponent_winrate'],
											 print_results_every = config['print_results_every'],
											 learn_rate = config['learn_rate'],
											 batch_size = config['batch_size'],
											 cnn_on = config['cnn_on'],
											 eps = config['eps'],
											 deterministic = config['deterministic'],
											 mcts = config['mcts'],
											 min_win_ticks = config['min_win_ticks'])
else:
	res = train_policy_gradients(game_spec,
											 create_network_func,
											 load_network_file_path = config['load_network_file_path'],
											 number_of_games = config['number_of_games'],
											 batch_size = config['batch_size'],
											 learn_rate = config['learn_rate'],
											 print_results_every = config['print_results_every'],
											 save_network_file_path = config['save_network_file_path'],
											 cnn_on = config['cnn_on'],
								 			 eps = config['eps'],
								 			 deterministic = config['deterministic'],
								 			mcts=config['mcts'])


config["results"] = res[2]
plt.save(config)


# pdb.set_trace()