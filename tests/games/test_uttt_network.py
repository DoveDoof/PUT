import unittest as test
from pprint import pprint


import functools
import games.uttt as ut
import tensorflow as tf
import time

from common.network_helpers import load_network, get_stochastic_network_move, create_network
from common.visualisation import load_results

class TestUltimateTicTacToe(test.TestCase):
	# Test case
	# Manually playing single game against a network
	# run with: py -m unittest tests\games\test_uttt_network.py
	def test_game_network(self):

		network_file_path = 'networks/regularization_51_e-3b100/net_ep54000.p'
		n = 1000

		# if resultfile exists:
		# network_info = load_results(network_file_path, results_only = False)
		# input_layer = network_info['input_layer']
		# hidden_layers = network_info['hidden_layers']
		# output_layer = network_info['output_layer']
		# otherwise look up values
		input_layer = 90
		hidden_layers = [51]
		output_layer = 81

		game_spec = ut.UltimateTicTacToeGameSpec()

		opponent_func = game_spec.get_random_player_func()
		# opponent_func = game_spec.get_manual_player_func()

		def player_func(board_state, side):
			move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side, valid_only = True, game_spec = game_spec)
			return game_spec.flat_move_to_tuple(move.argmax())

		create_network_func = functools.partial(create_network, input_layer, hidden_layers, output_layer)
		input_layer, output_layer, variables, _ = create_network_func()

		t = time.perf_counter()	
		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			print("loading pre-existing network")
			load_network(session, variables, network_file_path)

			results_X = [game_spec.play_game(player_func, opponent_func) for i in range(int(n/2))]
			results_O = [game_spec.play_game(opponent_func, player_func) for i in range(int(n/2))]
			elapsed_time = time.perf_counter() - t
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