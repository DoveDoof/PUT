import unittest as test
from pprint import pprint


import functools
import games.uttt as ut
import tensorflow as tf

from common.network_helpers import load_network, get_stochastic_network_move, create_network
from common.visualisation import load_results

class TestUltimateTicTacToe(test.TestCase):
	# Test case
	# Manually playing single game against a network
	# run with: py -m unittest tests\games\test_uttt_network.py
	def test_game_network(self):

		network_file_path = 'networks/ep-3_81_81_81/net_ep6500_2018-07-27_113928.p'
		

		network_info = load_results(network_file_path, results_only = False)
		input_layer = network_info['input_layer']
		hidden_layers = network_info['hidden_layers']
		output_layer = network_info['output_layer']

		game_spec = ut.UltimateTicTacToeGameSpec()

		# opponent_func = game_spec.get_random_player_func()
		opponent_func = game_spec.get_manual_player_func()

		def player_func(board_state, side):
			move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side, valid_only = True, game_spec = game_spec)
			return game_spec.flat_move_to_tuple(move.argmax())

		create_network_func = functools.partial(create_network, input_layer, hidden_layers, output_layer)
		input_layer, output_layer, variables = create_network_func()

		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			print("loading pre-existing network")
			load_network(session, variables, network_file_path)

			# switch functions to change who begins the game
			game_spec.play_game(player_func, opponent_func, log=2)