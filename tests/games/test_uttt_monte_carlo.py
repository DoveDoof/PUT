import unittest as test
import time

import games.uttt as ut
import techniques.monte_carlo as mc


class TestUltimateTicTacToe(test.TestCase):
	@test.skip
	def test_move(self):
		# Test single move of monte carlo tree search algorithm
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



	@test.skip
	def test_game(self):
		# test game against monte carlo tree search algorithm
		game_spec = ut.UltimateTicTacToeGameSpec()

		player_func = game_spec.get_monte_carlo_player_func()
		opponent_func = game_spec.get_random_player_func()
		# opponent_func = game_spec.get_manual_player_func()

		game_spec.play_game(player_func, opponent_func, log = 1)

	def test_performance(self):
		# test performance of mcts against random bot
		game_spec = ut.UltimateTicTacToeGameSpec()

		n = 10
		number_of_samples = 27

		mcts_func = game_spec.get_monte_carlo_player_func(number_of_samples)
		rand_func = game_spec.get_random_player_func()

		t = time.perf_counter()
		ut.play_game(mcts_func, rand_func)
		elapsed_time = time.perf_counter() - t
		print('One game takes %s seconds, so %s will take %s seconds' % (elapsed_time, n, n*elapsed_time))

		t = time.perf_counter()
		resultsX = [ut.play_game(mcts_func, rand_func) for i in range(n)]
		resultsO = [ut.play_game(rand_func, mcts_func) for i in range(n)]
		elapsed_time = time.perf_counter() - t

		print('Elapsed time:', elapsed_time)
		print('mcts as X:')
		print('mcts wins: ', resultsX.count(1))
		print('random wins: ', resultsX.count(-1))
		print('Draws        : ', resultsX.count(0))
		print('Winrate      : ', 0.5 + 1.*sum(resultsX)/2/n)

		print('mcts as O:')
		print('mcts wins: ', resultsO.count(-1))
		print('random wins: ', resultsO.count(1))
		print('Draws        : ', resultsO.count(0))
		print('Winrate      : ', 0.5 - 1.*sum(resultsO)/2/n)

if __name__ == '__main__':
	test.main()