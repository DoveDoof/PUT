from unittest import TestCase
from pprint import pprint

import random
import time
import numpy as np
import games.uttt as ut

class TestUltimateTicTacToe(TestCase):

	# b = ut._new_board()
	# pprint(ut.apply_move(b, (8,3), 1))


	# last_move = (5,6)
	# last_move = (6,3)
	# next_move = (2,0)
	# b = (( 1,  0,  0,  1,  0,  0,  0,  1,  0),
	# 	 ( 1,  0,  1, -1,  1,  0,  0,  1,  0),
	# 	 ( 1, -1,  0,  0,  0, -1,  0,  1,  0),
	# 	 (-1,  1,  0, -1, -1,  0,  0, -1,  0),
	# 	 (-1, -1,  1,  1, -1,  0, -1,  0,  1),
	# 	 (-1,  1,  1,  0,  1,  1, -1, -1,  0),
	# 	 ( 0,  0,  0, -1,  1, -1, -1, -1,  0),
	# 	 ( 1,  0, -1,  1,  0,  1, -1, -1,  0),
	# 	 (-1, -1,  1,  1, -1, -1,  0,  1,  1))
	# print(tuple(ut.available_moves(b)))

	random.seed(1)

	# ut.play_game(ut.random_player, ut.random_player, log = 2)

	n = 1000

	t = time.perf_counter()	
	results = [ut.play_game(ut.random_player, ut.random_player) for i in range(n)]
	elapsed_time = time.perf_counter() - t
	print('Elapsed time:', elapsed_time)
	print('Player X wins: ', results.count(1)/n)
	print('Player O wins: ', results.count(-1)/n)
	print('Draws        : ', results.count(0)/n)
	print('Winrate      : ', 0.5 + 1.*sum(results)/n)

	# b = ((0, -1, 1, 1, 0, 1, 0, -1, -1),
	# 	 (0, -1, 1, -1, 1, -1, 1, 1, -1),
	# 	 (-1, -1, -1, -1, -1, 1, -1, 1, -1),
	# 	 (-1, -1, 0, 1, 1, -1, 1, 1, -1),
	# 	 (0, -1, 0, 1, 0, -1, 1, 1, 0),
	# 	 (1, -1, 1, 1, -1, 0, 0, 0, 1),
	# 	 (1, 1, 0, 0, 1, -1, 	 1, -1,  1),
	# 	 (1, 0, -1, -1, -1, -1, -1, -1,  1),
	# 	 (1, 0, -1, 1, 1, 0, 	 1,  1, -1))
	# mb = ut._get_macroboard(b)
	# pprint(mb)
	# mb =((-1,  1, -1),
	# 	 (-1,  1,  1), 
	# 	 ( 1, -1,  None))
	# print(ut._winner_microboard(mb))