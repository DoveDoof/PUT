import unittest as test

import games.uttt as ut
import techniques.monte_carlo as mc


class TestUltimateTicTacToe(test.TestCase):
	@test.skip
	def test_move(self):
		game_spec = ut.UltimateTicTacToeGameSpec()


		# generate board with 10 random moves
		random_func = game_spec.get_random_player_func()
		board_state = ut._new_board()
		side = 1
		for _ in range(10):
			move = random_func(board_state, side)
			board_state = game_spec.apply_move(board_state, move, side)
			side = -1*side
		
		print("")
		ut.print_board_state(board_state, side)

		result, move = mc._monte_carlo_sample(game_spec, board_state, side)
		print("result: ", result)
		print("move: ", move)
		mc_func = game_spec.get_monte_carlo_player_func()
		result, move = mc.monte_carlo_tree_search(game_spec, board_state, side, 100)
		print(result)
		print(move)



	# @test.skip
	def test_game(self):
		game_spec = ut.UltimateTicTacToeGameSpec()

		player_func = game_spec.get_monte_carlo_player_func()
		opponent_func = game_spec.get_random_player_func()
		# opponent_func = game_spec.get_manual_player_func()

		game_spec.play_game(player_func, opponent_func, log = 1)