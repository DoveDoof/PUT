"""
Quick script to simulate the training process results that were not stored due to interrupting the training program.
Only works if hist0 is still the first net and is never overwritten because the winrate was large enough 3 times.
This is equivalent to not having a file 'net_ep####.p' in the folder.

ignores deterministic and is always stochastic, with and without mcts
"""


import functools
import pdb
import common.visualisation as plt
import json
import random
import common.custom_cnn as cnn
import collections
import os
import time
import numpy as np
import tensorflow as tf
import re

random.seed(1)

from common.network_helpers import get_stochastic_network_move, load_network, save_network, get_deterministic_network_move, get_random_network_move
from common.network_helpers import create_3x3_board_states
from common.network_helpers import create_network
from common.visualisation import load_results
from techniques.monte_carlo import monte_carlo_tree_search
from techniques.train_policy_gradient_historic import train_policy_gradients_vs_historic
from games.uttt import UltimateTicTacToeGameSpec


netloc = r'C:\Users\User\APH\1B 2017 2018\Advanced Machine Learning\Resit\Git\QLUT\networks\cnn_50_50_50_e-3_stoch_mcts\\'
nr_testgames = 100



# Import config file
import yaml
with open(netloc + '_config.yml', 'r') as f:
	config = yaml.load(f)

game_spec = UltimateTicTacToeGameSpec()

config['input_layer'] = game_spec.board_squares()
output_layer = game_spec.outputs()

config['output_layer'] = output_layer

if config['cnn_on']:
	create_network = functools.partial(cnn.create_network,
											config['filter_shape'],
											config['filter_depth'],
											config['dense_width'])
else:
	create_network = functools.partial(create_network,
											config['input_layer'],
											config['hidden_layers'],
											config['output_layer'])

load_network_file_path = config['load_network_file_path']
save_network_file_path = config['save_network_file_path']
number_of_historic_networks = 1
historic_network_base_path = config['historic_network_base_path']
number_of_games = config['number_of_games']
update_opponent_winrate = config['update_opponent_winrate']
print_results_every = config['print_results_every']
learn_rate = config['learn_rate']
batch_size = config['batch_size']
cnn_on = config['cnn_on']
eps = config['eps']
deterministic = config['deterministic']
mcts = config['mcts']
min_win_ticks = config['min_win_ticks']
beta = config['beta']


input_layer, output_layer, variables, weights = create_network()

current_historical_index = 0
historical_networks = []


f = []
for (dirpath, dirnames, filenames) in os.walk(netloc):
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

if len(netlist_hist) == 0:
	netlist_hist.append(historic_net)

nr_games = []
for i, name in enumerate(raw_netlist):
	nr_games.append((int(name[6:-20]), i))
nr_games.sort()
nr_games = nr_games[::-1]

netlist = [raw_netlist[i[1]] for i in nr_games]

print(netlist)

opponent_index = 0

historical_input_layer, historical_output_layer, historical_variables, _ = create_network()

networks = []
for _ in netlist:
	net_input_layer, net_output_layer, net_variables, _ = create_network()
	networks.append((net_input_layer, net_output_layer, net_variables))

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	print('location: '+netloc)
	load_network(session, historical_variables, netloc+historic_net)
	print('loaded historic net: ' + historic_net)

	def make_move_historical(net, board_state, side):
		if mcts:
			_,move = monte_carlo_tree_search(game_spec, board_state, side, 27, session,
											input_layer, output_layer, True, cnn_on, True)
		else:
			# move = get_deterministic_network_move(session, net[0], net[1], board_state, side,
			# 										valid_only = True, game_spec = game_spec, cnn_on = cnn_on)
			move = get_stochastic_network_move(session, net[0], net[1], board_state, side,
			                                 valid_only=True, game_spec=game_spec, cnn_on=cnn_on)

		move_for_game = np.asarray(move) # move must be an array, mcts doesn't return this
		return game_spec.flat_move_to_tuple(move_for_game.argmax())

	results = []
	for i, network in enumerate(networks):
		load_network(session, network[2], netloc+netlist[i])
		print('loaded network: ' + netlist[i])

		make_move_hist = functools.partial(make_move_historical, (historical_input_layer, historical_output_layer, historical_variables))
		make_move_testnet = functools.partial(make_move_historical, network)

		results.append([])
		for j in range(nr_testgames):
			if bool(random.getrandbits(1)):
				# testnet as X
				reward = game_spec.play_game(make_move_testnet, make_move_hist)
			else:
				# testnet as O
				reward = -game_spec.play_game(make_move_hist, make_move_testnet)
			results[-1].append(reward)
			print('game %i/%i done'%(j+1,nr_testgames), end="\r")

		with open(netloc+'_postresults_benchmarking'+str(nr_games[i][0])+'.json', 'w') as outfile:
			json.dump(results[-1], outfile)
		print('network done, winrate: %f, results:' % round(np.mean(results[-1])/2.0 + 0.5, 3))
		print(results[-1])

	with open(netloc+'_postresults_benchmarking.json', 'w') as outfile:
		json.dump(results, outfile)