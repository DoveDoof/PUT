import collections
import os
import random
import time

import numpy as np
import tensorflow as tf

from common.network_helpers import create_3x3_board_states
from common.network_helpers import load_network, get_stochastic_network_move, save_network
from common.visualisation import load_results


def train_policy_gradients(game_spec,
                           create_network,
                           network_file_path,
                           save_network_file_path=None,
                           opponent_func=None,
                           number_of_games=10000,
                           print_results_every=1000,
                           learn_rate=1e-4,
                           batch_size=100,
                           randomize_first_player=True):
    """Train a network using policy gradients

    Args:
        save_network_file_path (str): Optionally specifiy a path to use for saving the network, if unset then
            the network_file_path param is used.
        opponent_func (board_state, side) -> move: Function for the opponent, if unset we use an opponent playing
            randomly
        randomize_first_player (bool): If True we alternate between being the first and second player
        game_spec (games.base_game_spec.BaseGameSpec): The game we are playing
        create_network (->(input_layer : tf.placeholder, output_layer : tf.placeholder, variables : [tf.Variable])):
            Method that creates the network we will train.
        network_file_path (str): path to the file with weights we want to load for this network
        number_of_games (int): number of games to play before stopping
        print_results_every (int): Prints results to std out every x games, also saves the network
        learn_rate (float):
        batch_size (int):

    Returns:
        (variables used in the final network : list, win rate: float)
    """
    save_network_file_path = save_network_file_path or network_file_path
    # create folder if it does not exist
    if save_network_file_path:
        split = save_network_file_path.split('/')
        directory = '/'.join(split[:-1]) or '.'
        if not os.path.isdir(directory):
            os.makedirs(directory)
            print("created directory " + directory)

    opponent_func = opponent_func or game_spec.get_random_player_func()
    reward_placeholder = tf.placeholder("float", shape=(None,))
    actual_move_placeholder = tf.placeholder("float", shape=(None, game_spec.outputs()))

    input_layer, output_layer, variables = create_network()

    policy_gradient = tf.log(
        tf.reduce_sum(tf.multiply(actual_move_placeholder, output_layer), axis=1)) * reward_placeholder

    train_step = tf.train.AdamOptimizer(learn_rate).minimize(-policy_gradient)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # load existing network and keep track of number of games played
        base_episode_number = 0
        winrates = []
        if network_file_path and os.path.isfile(network_file_path):
            print("loading pre-existing network")
            load_network(session, variables, network_file_path)
            base_episode_number, winrates = load_results(network_file_path)

        mini_batch_board_states, mini_batch_moves, mini_batch_rewards = [], [], []
        results = collections.deque(maxlen=print_results_every)

        def make_training_move(board_state, side):
            # We must have the first 3x3 board as first 9 entries of the list, second 3x3 board as next 9 entries etc.
            # This is required for the CNN. The CNN takes the first 9 entries and forms a 3x3 board etc.
            #np_board_state = np.array(board_state)
            """If the 10 split 3x3 boards are desired, use create_3x3_board_states(board_state) here"""
            np_board_state = create_3x3_board_states(board_state)

            mini_batch_board_states.append(np_board_state * side) # append all states are used in the minibatch (+ and - determine which player's state it was)
            move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side, valid_only = True, game_spec = game_spec)
            mini_batch_moves.append(move)
            return game_spec.flat_move_to_tuple(move.argmax())

        for episode_number in range(1, number_of_games+1):
            # randomize if going first or second
            if (not randomize_first_player) or bool(random.getrandbits(1)):
                reward = game_spec.play_game(make_training_move, opponent_func)     # In this line one game is played.
            else:
                reward = -game_spec.play_game(opponent_func, make_training_move)

            results.append(reward)

            # we scale here so winning quickly is better winning slowly and losing slowly better than losing quickly
            last_game_length = len(mini_batch_board_states) - len(mini_batch_rewards)

            reward /= float(last_game_length)

            mini_batch_rewards += ([reward] * last_game_length)

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

            if episode_number % print_results_every == 0:
                winrate = _win_rate(print_results_every, results)
                if winrate == 0:
                    print('DEBUG TEST')
                winrates.append([base_episode_number+episode_number, winrate])
                print("episode: %s win_rate: %s" % (base_episode_number+episode_number, winrate))
                if save_network_file_path:
                    save_network(session, variables, time.strftime(save_network_file_path[:-2]+"_ep"+str(base_episode_number+episode_number)+"_%Y-%m-%d_%H%M%S.p"))

                """
                tf.trainable_variables() #contains the name and shape of the weights of the layers.
                print(tf.trainable_variables())
                w1 = [v for v in tf.trainable_variables() if v.name == "network/output_weights:0"][0]
                b1 = [v for v in tf.trainable_variables() if v.name == "network/output_bias:0"][0]
                init_weights = tf.global_variables_initializer()
                with tf.Session() as sesstest:
                    sesstest.run(init_weights)
                    v1 = sesstest.run(w1)
                    print('\n Output layer weights: \n')
                    print(v1)
                    v2 = sesstest.run(b1)
                    print('\n Biases: \n')
                    print(v2)
                
                # tf.trainable_variables() contains the name and shape of the weights of the layers.
                print(tf.trainable_variables())
                w_all = [v for v in tf.trainable_variables()] # This will give all weights and biases.
                # To look only at one layer of weights, check trainable_variables for the name and use f.e.
                # w1 = [v for v in tf.trainable_variables() if v.name == "network/weights_1:0"][0]
                init_weights = tf.global_variables_initializer()
                # To actually print the the weights on screen, a Session must be used:
                with tf.Session() as sesstest:
                    sesstest.run(init_weights)
                    v = sesstest.run(w_all)
                    print('\n Weights \n')
                    print(v)
            """
        if save_network_file_path:
            save_network(session, variables, save_network_file_path)

    return variables, _win_rate(print_results_every, results), winrates


def _win_rate(print_results_every, results):
    return 0.5 + sum(results) / (print_results_every * 2.)

