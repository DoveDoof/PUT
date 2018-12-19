import unittest as test
from pprint import pprint

import functools
import games.uttt as ut
import tensorflow as tf
import time
import json
import os
import re
import numpy as np

from techniques.monte_carlo import monte_carlo_tree_search
import common.custom_cnn as cnn
from common.network_helpers import load_network, get_stochastic_network_move, create_network, get_deterministic_network_move
from common.visualisation import load_results

class TestUltimateTicTacToe(test.TestCase):
	# Test case
	# Manually playing single game against a network
	# run with: py -m unittest tests\games\test_uttt_network.py

	def test_game_network(self):

		network_file_path = r'C:\Users\User\APH\1B 2017 2018\Advanced Machine Learning\Resit\Git\QLUT\networks\regnn_50_50_50_e-3_stoch_mcts\\'
		n = 100
		steps_games = 10
		cnn_on = False
		mcts = True
		filter_shape = [3, 3]
		filter_depth = [10, 10, 10]
		dense_width = []
		input_layer = 90
		hidden_layers = [51, 51, 51]
		output_layer = 81

		f = []
		for (dirpath, dirnames, filenames) in os.walk(network_file_path):
			f.extend(filenames)
			break

		netlist_hist = []
		raw_netlist = []
		for file in f:
			p = re.compile('net_ep\d+_.+\.p')
			if 'config' in file:
				pass
			elif 'hist' in file:
				historic_net = file
			elif p.search(file) is None:
				netlist_hist.append(file)
			else:
				raw_netlist.append(file)

		nr_games = []
		for i, name in enumerate(raw_netlist):
			nr_games.append((int(name[6:-20]), i))
		nr_games.sort()

		netlist = [raw_netlist[i[1]] for i in nr_games]
		gamefiles = [ netlist[(i+1)*steps_games - 1] for i in range(0,int(len(netlist)/steps_games))]
		network_games = [nr_games[(i+1)*steps_games - 1][0] for i in range(0,int(len(netlist)/steps_games))]
		print(gamefiles)
		print(gamefiles[0][1])

		# if resultfile exists:
		# network_info = load_results(network_file_path, results_only = False)
		# input_layer = network_info['input_layer']
		# hidden_layers = network_info['hidden_layers']
		# output_layer = network_info['output_layer']
		# otherwise look up values



		game_spec = ut.UltimateTicTacToeGameSpec()

		if mcts:
			opponent_func = game_spec.get_monte_carlo_player_func(number_of_samples=27)
		else:
			opponent_func = game_spec.get_random_player_func()


		# opponent_func = game_spec.get_manual_player_func()

		def player_func(board_state, side):
			if mcts:
				_, move = monte_carlo_tree_search(game_spec, board_state, side, 27, session,
												  input_layer, output_layer, True, cnn_on, True)
			else:
				move = get_deterministic_network_move(session, input_layer, output_layer, board_state, side, valid_only = True, game_spec = game_spec)

			move_for_game = np.asarray(move)  # The move returned to the game is in a different configuration than the CNN learn move
			return game_spec.flat_move_to_tuple(move_for_game.argmax())

		if cnn_on:
			create_network_func = functools.partial(cnn.create_network,
											   filter_shape,
											   filter_depth,
											   dense_width)
		else:
			create_network_func = functools.partial(create_network, input_layer, hidden_layers, output_layer)


		input_layer, output_layer, variables, _ = create_network_func()
		results = {}
		for i in range(len(gamefiles)):
			t = time.perf_counter()
			with tf.Session() as session:
				session.run(tf.global_variables_initializer())
				print("loading pre-existing network")
				load_network(session, variables, network_file_path + gamefiles[i]) #\\

				results_X = [game_spec.play_game(player_func, opponent_func) for _ in range(int(n/2))]
				results_O = [game_spec.play_game(opponent_func, player_func) for _ in range(int(n/2))]
				elapsed_time = time.perf_counter() - t

				results[network_games[i]] = results_X + [-j for j in results_O]
				print('network ' + gamefiles[i])
				"""
				print('Elapsed time:', elapsed_time)
				print('Network as X (%s games):' % (n/2))
				print('Player X wins: ', results_X.count(1)/n*2)
				print('Player O wins: ', results_X.count(-1)/n*2)
				print('Draws        : ', results_X.count(0)/n*2)
				print('Winrate X    : ', 0.5 + 1.*sum(results_X)/n*2)

				print('Network as O (%s games):' % (n/2))
				print('Player O wins: ', results_O.count(-1)/n*2)
				print('Player X wins: ', results_O.count(1)/n*2)
				print('Draws        : ', results_O.count(0)/n*2)
				print('Winrate O    : ', 0.5 - 1.*sum(results_O)/n*2)

				"""

		with open(network_file_path +'_benchmark_vs_rand.json', 'w') as outfile:
			json.dump(results, outfile)


if __name__ == '__main__':
	test.main()