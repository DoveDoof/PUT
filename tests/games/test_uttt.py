import unittest as test
from pprint import pprint

import time
import numpy as np
import games.uttt as ut

class TestUltimateTicTacToe(test.TestCase):
	@test.skip
	def test_moves(self):
		b = ut._new_board()
		pprint(ut.apply_move(b, (8,3), 1))

	# @test.skip
	def test_game(self):
		ut.play_game(ut.random_player, ut.random_player, log = 2)
	
	@test.skip
	def test_games(self):
		n = 1000

		t = time.perf_counter()	
		results = [ut.play_game(ut.random_player, ut.random_player) for i in range(n)]
		elapsed_time = time.perf_counter() - t
		print('Elapsed time:', elapsed_time)
		print('Player X wins: ', results.count(1)/n)
		print('Player O wins: ', results.count(-1)/n)
		print('Draws        : ', results.count(0)/n)
		print('Winrate      : ', 0.5 + 1.*sum(results)/n)

	@test.skip
	def test_macroboard(self):
		b = (( 0, -1,  1,  1,  0,  1,  0, -1, -1),
			 ( 0, -1,  1, -1,  1, -1,  1,  1, -1),
			 (-1, -1, -1, -1, -1,  1, -1,  1, -1),
			 (-1, -1,  0,  1,  1, -1,  1,  1, -1),
			 ( 0, -1,  0,  1,  0, -1,  1,  1,  0),
			 ( 1, -1,  1,  1, -1,  0,  0,  0,  1),
			 ( 1,  1,  0,  0,  1, -1,  1, -1,  1),
			 ( 1,  0, -1, -1, -1, -1, -1, -1,  1),
			 ( 1,  0, -1,  1,  1,  0,  1,  1, -1))
		mb = b[-1]
		pprint(mb)
		mb =((-1,  1, -1),
			 (-1,  1,  1), 
			 ( 1, -1,  None))
		print(ut._winner_microboard(mb))