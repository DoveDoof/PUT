import collections
import functools
import os
import random
import time

import numpy as np
import tensorflow as tf

from common.network_helpers import get_stochastic_network_move, load_network, save_network, get_deterministic_network_move
from common.network_helpers import create_3x3_board_states
from common.visualisation import load_results

def train_policy_gradients_vs_historic(game_spec, create_network, network_file_path,
                                       save_network_file_path = None,
                                       number_of_historic_networks = 1,
                                       historic_network_base_path = 'historic_network',
                                       number_of_games = 100000,
                                       update_opponent_winrate = 0.65,
                                       print_results_every = 1000,
                                       learn_rate = 1e-4,
                                       batch_size = 100,
                                       cnn_on = False):
    """Train a network against itself and over time store new version of itself to play against.

    Args:
        historic_network_base_path (str): Base path to save new historic networks to a number for the network "slot" is
            appended to the end of this string.
        save_historic_every (int): We save a version of the learning network into one of the historic network
            "slots" every x number of games. We have number_of_historic_networks "slots"
        number_of_historic_networks (int): We keep this many old networks to play against
        save_network_file_path (str): Optionally specifiy a path to use for saving the network, if unset then
            the network_file_path param is used.
        game_spec (games.base_game_spec.BaseGameSpec): The game we are playing
        create_network (->(input_layer : tf.placeholder, output_layer : tf.placeholder, variables : [tf.Variable])):
            Method that creates the network we will train.
        network_file_path (str): path to the file with weights we want to load for this network
        number_of_games (int): number of games to play before stopping
        update_opponent_winrate (float): the required winrate before updating the opponent to the newer agent
        print_results_every (int): Prints results to std out every x games, also saves the network
        learn_rate (float):
        batch_size (int):

    Returns:
        [tf.Variables] : trained variables used in the final network
    """
    save_network_file_path = save_network_file_path or network_file_path
    # create folder if it does not exist
    if save_network_file_path:
        split = save_network_file_path.split('/')
        directory = '/'.join(split[:-1]) or '.'
        if not os.path.isdir(directory):
            os.makedirs(directory)
            print("created directory " + directory)

    reward_placeholder = tf.placeholder("float", shape=(None,))
    actual_move_placeholder = tf.placeholder("float", shape=(None, game_spec.outputs()))

    input_layer, output_layer, variables = create_network()

    policy_gradient = tf.reduce_sum(tf.reshape(reward_placeholder, (-1, 1)) * actual_move_placeholder * output_layer)
    train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(-policy_gradient) # Why is this one different from the other train policy grad?

    current_historical_index = 0 # We will (probably) not use this: we always train against the most recent agent
    historical_networks = []

    mini_batch_board_states, mini_batch_moves, mini_batch_rewards = [], [], []
    results = collections.deque(maxlen=print_results_every)

    for _ in range(number_of_historic_networks):
        historical_input_layer, historical_output_layer, historical_variables = create_network()
        historical_networks.append((historical_input_layer, historical_output_layer, historical_variables))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        base_episode_number = 0
        winrates = []

        if network_file_path and os.path.isfile(network_file_path):
            print("loading pre-existing network")
            load_network(session, variables, network_file_path)
            base_episode_number, winrates = load_results(network_file_path)

        def make_move_historical(historical_network_index, board_state, side):
            net = historical_networks[historical_network_index]
            #move = get_stochastic_network_move(session, net[0], net[1], board_state, side,
            #                                  valid_only=True, game_spec=game_spec, CNN_ON=cnn_on)
            move = get_deterministic_network_move(session, net[0], net[1], board_state, side,
                                                valid_only = True, game_spec = game_spec, cnn_on = cnn_on)

            return game_spec.flat_move_to_tuple(move.argmax())

        def make_training_move(board_state, side):
            if cnn_on:
                np_board_state = create_3x3_board_states(board_state)
            else:
                np_board_state = np.array(board_state)

            mini_batch_board_states.append(np_board_state * side)
            #move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side,
            #                                   valid_only=True, game_spec=game_spec, CNN_ON=cnn_on)
            move = get_deterministic_network_move(session, input_layer, output_layer, board_state, side,
                                                  valid_only = True, game_spec = game_spec, cnn_on = cnn_on)
            move_for_game = move  # The move returned to the game is in a different configuration than the CNN learn move
            if cnn_on:
                # Since the mini batch states is saved the same way it should enter the neural net (the adapted board state),
                # the same should happen for the mini batch moves
                move = create_3x3_board_states(np.reshape(move,[9,9]))   # The function requires a 9x9 array
                mini_batch_moves.append(move[0:81])
            else:
                mini_batch_moves.append(move)
            return game_spec.flat_move_to_tuple(move_for_game.argmax())
        """-------------------------------"""
#        if os.path.isfile(network_file_path):
#            print("loading pre existing weights")
#            load_network(session, variables, network_file_path)
#        else:
#            print("Could not find previous weights so initialising randomly")

        for i in range(number_of_historic_networks):
            if os.path.isfile(historic_network_base_path + str(i) + '.p'):
                load_network(session, historical_networks[i][2], historic_network_base_path + str(i) + '.p')
            elif os.path.isfile(str(network_file_path)):
                # if we can't load a historical file use the current network weights
                """------------------------------Add random bot for the first try------------------------------------"""
                print('Warning: loading historical file failed. Current net is being used.')
                load_network(session, historical_networks[i][2], network_file_path)

        first_bot = True  # When true, the agent did not yet achieve high enough winrate, so it plays against a random bot
        win_ticks = 0 # registers the amount of times the agent has a high enough winrate to update its opponent
        for episode_number in range(1, number_of_games):
            opponent_index = random.randint(0, number_of_historic_networks - 1)
            make_move_historical_for_index = functools.partial(make_move_historical, opponent_index)
            if first_bot:
                make_move_historical_for_index = game_spec.get_random_player_func()

            # randomize if going first or second
            if bool(random.getrandbits(1)):
                reward = game_spec.play_game(make_training_move, make_move_historical_for_index)
            else:
                reward = -game_spec.play_game(make_move_historical_for_index, make_training_move)

            results.append(reward)

            # we scale here so winning quickly is better winning slowly and loosing slowly better than loosing quick
            last_game_length = len(mini_batch_board_states) - len(mini_batch_rewards)

            reward /= float(last_game_length)

            mini_batch_rewards += ([reward] * last_game_length)

            episode_number += 1

            if episode_number % batch_size == 0:
                normalized_rewards = mini_batch_rewards - np.mean(mini_batch_rewards)

                rewards_std = np.std(normalized_rewards)
                if rewards_std != 0:
                    normalized_rewards /= rewards_std
                else:
                    print("warning: got mini batch std of 0.")

                np_mini_batch_board_states = np.array(mini_batch_board_states) \
                    .reshape(len(mini_batch_rewards), *input_layer.get_shape().as_list()[1:])

                session.run(train_step, feed_dict={input_layer: np_mini_batch_board_states,
                                                   reward_placeholder: normalized_rewards,
                                                   actual_move_placeholder: mini_batch_moves})

                # clear batches
                del mini_batch_board_states[:]
                del mini_batch_moves[:]
                del mini_batch_rewards[:]

#            if episode_number % print_results_every == 0:
#                print("episode: %s average result: %s" % (episode_number, np.mean(results)))
#                #if np.mean(results)

            if episode_number % print_results_every == 0:
                winrate = _win_rate(print_results_every, results)
                if winrate == 0:
                    print('DEBUG TEST')
                winrates.append([base_episode_number+episode_number, winrate])
                print("episode: %s win_rate: %s" % (base_episode_number+episode_number, winrate))
                if save_network_file_path:
                    save_network(session, variables, time.strftime(save_network_file_path[:-2]+"_ep"+str(base_episode_number+episode_number)+"_%Y-%m-%d_%H%M%S.p"))

            # Update opponent when winrate is high enough and it happens for a longer period
            if (episode_number % print_results_every == 0) and (winrate >= update_opponent_winrate):
                win_ticks += 1
                if win_ticks >= 3:
                    win_ticks = 0
                    first_bot = False
                    print("saving historical network %s", current_historical_index) # Overwrite historic opponent with current network
                    save_network(session, variables, historic_network_base_path + str(current_historical_index) + '.p')
                    load_network(session, historical_networks[current_historical_index][2],
                                 historic_network_base_path + str(current_historical_index) + '.p')

                    # also save to the main network file
                    save_network(session, variables, save_network_file_path or network_file_path)

                    current_historical_index += 1 # Not used when we only have 1 historic network
                    current_historical_index %= number_of_historic_networks

        # save our final weights
        save_network(session, variables, save_network_file_path or network_file_path)

    return variables, _win_rate(print_results_every, results), winrates

def _win_rate(print_results_every, results):
    return 0.5 + sum(results) / (print_results_every * 2.)