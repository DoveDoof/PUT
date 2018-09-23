"""
Full code for running a game of ultimate tic-tac-toe on a 9x9 board

for game rules, see https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe

We play on a 9x9 board and because the last move determines where the next player is allowed to play, 
we add an extra row at the bottom which is the flattened macroboard. 
A one indicates that you are allowed to play in the smaller 3x3 area on the main board, whereas a zero indicates that you cannot.

On the board, a zero indicates no player has played there yet, a 1 and -1 are the players.

"""



import itertools
import random
import numpy as np

from functools import partial
from common.base_game_spec import BaseGameSpec
from techniques.min_max import evaluate
import techniques.monte_carlo as mc


_mb_unfinished = 8
_mb_available = 7

# do not change these, so we can sum results for winrate
_mb_draw = 0
_mb_X = 1
_mb_O = -1

def _new_board():
    return tuple((_mb_available,)*9 if i==9 else (_mb_unfinished,)*9 for i in range(10))
    # return ((0,)*9,)*10

def apply_move(board_state, move, side):
    move_x, move_y = move

    # update board
    updated_row = board_state[move_x][:move_y] + (side,) + board_state[move_x][move_y+1:]
    updated_board = board_state[:move_x] + (updated_row,) + board_state[move_x+1:-1]

    # update macroboard
    changed_index = move_x//3*3 + move_y//3 # in range 0-9
    next_move_index = move_x%3*3 + move_y%3 # in range 0-9
    
    def get_macroboard():
        macroboard_changed = _winner_microboard(_get_microboard(updated_board, (move_x//3, move_y//3)))
        if next_move_index == changed_index:
            macroboard_next_move = macroboard_changed
        else:
            macroboard_next_move = board_state[9][next_move_index]

        if macroboard_next_move in (_mb_unfinished, _mb_available):
            # replace mb_next_move with _mb_available
            macroboard_next_move = _mb_available
            # replace other values with _mb_unfinished or actual value
            replace_value = _mb_unfinished
        else:
            # mb_next_move is occupied, leave it as is
            # macroboard_next_move = macroboard_next_move
            # replace other values with _mb_available or actual value
            replace_value = _mb_available
        
        def get_tuples(i, replace_value):
            if i == next_move_index:
                return macroboard_next_move
            tocheck = macroboard_changed if i==changed_index else board_state[9][i]
            if tocheck in (_mb_available, _mb_unfinished):
                return replace_value
            else:
                return tocheck


        macroboard = tuple(get_tuples(i, replace_value) for i in range(9))
        return macroboard

    return updated_board + (get_macroboard(),)

def _get_microboard(board, pos):
    # returns copy of microboard selected by pos = (0-2, 0-2)
    def get_tuples():
        for x in range(pos[0]*3, pos[0]*3+3):
            yield board[x][pos[1]*3:(pos[1]*3+3)]
    return tuple(get_tuples())

def _has_3_in_a_line(line):
    return all(x == _mb_O for x in line) | all(x == _mb_X for x in line)

def _is_full(microboard):
    return not(any(_mb_available in row or _mb_unfinished in row for row in microboard))

def _winner_microboard(microboard):
    for x in range(3):
        if _has_3_in_a_line(microboard[x]):
            return microboard[x][0]
    # check columns
    for y in range(3):
        if _has_3_in_a_line([i[y] for i in microboard]):
            return microboard[0][y]

    # check diagonals
    if _has_3_in_a_line([microboard[i][i] for i in range(3)]):
        return microboard[0][0]
    if _has_3_in_a_line([microboard[2 - i][i] for i in range(3)]):
        return microboard[0][2]

    # draw
    if _is_full(microboard):
        return _mb_draw

    # otherwise, unfinished
    return _mb_unfinished


def available_moves(board_state):
    board = board_state[:-1]
    macroboard = board_state[-1]
    for x, y in itertools.product(range(9), range(9)):
        if (board[x][y]==_mb_unfinished and macroboard[x//3*3+y//3] == _mb_available):
            yield (x,y)


def has_winner(board_state):
    # +1 for player X, -1 for player O, 0 for draw, None for no winner yet
    macroboard = tuple(board_state[-1][i*3:(i+1)*3] for i in range(3))
    winner = _winner_microboard(macroboard)
    if winner == _mb_unfinished:
        return None
    else:
        return winner


def evaluate(board_state):
    """An evaluation function for this game, gives an estimate of how good the board position is for the plus player.
    There is no specific range for the values returned, they just need to be relative to each other.

    Args:
        board_state (tuple): State of the board

    Returns:
        number
    """
    raise NotImplementedError()


def play_game(plus_player_func, minus_player_func, log=0):
    # int: 1 if the plus_player_func won, -1 if the minus_player_func won and 0 for a draw
    board_state = _new_board()
    player_turn = _mb_X

    last_move = None
    while True:
        _available_moves = list(available_moves(board_state))

        if len(_available_moves) == 0:
            # draw
            if log:
                print("no moves left, game ended a draw")
            return _mb_draw

        if log==2:
            import numpy as np
            print_board_state(board_state, '')
            input('Press Enter to continue')

        if player_turn > 0:
            move = plus_player_func(board_state, player_turn)
        else:
            move = minus_player_func(board_state, player_turn)

        # a new move is determined, so update last_move
        last_move = move
        
        if move not in _available_moves:
            # if a player makes an invalid move the other player wins
            if log:
                print("illegal move ", move)
            return -player_turn

        board_state = apply_move(board_state, move, player_turn)

        winner = has_winner(board_state)
        if winner != None:
            if winner == _mb_X:
                winner_text = 'X ('+str(_mb_X)+')'
            elif winner == _mb_O:
                winner_text = 'O ('+str(_mb_O)+')'
            else:
                winner_text = 'no winner'

            if log:
                print_board_state(board_state, '')
                print("we have a winner, side: %s" % winner_text)
            return winner
        player_turn = -player_turn

def random_player(board_state, _):
    moves = list(available_moves(board_state))
    return random.choice(moves)

def manual_player(board_state, side):
    moves = list(available_moves(board_state))
    print("Player "+str(side)+", choose any of the available moves (0-"+str(len(moves)-1)+"):\n")
    print(moves)
    mv = input()
    try:
        return moves[int(mv)]
    # catch non integer and out of range inputs
    except (ValueError, IndexError):
        print("Invalid input, playing random move.")
        return random.choice(moves)


def print_board_state(board_state, side, print_mb = True):
    for i in range(9):
        pretty_row = [" " if x==_mb_unfinished else x for x in board_state[i]]
        pretty_row = ['X' if x==_mb_X else x for x in pretty_row]
        pretty_row = ['O' if x==_mb_O else x for x in pretty_row]
        pretty_row.insert(3, "|")
        pretty_row.insert(7, "|")
        print(" ".join(pretty_row))
        if i==2 or i==5:
            print('------|-------|------')
    if print_mb:
        print("Macroboard:")
        pretty_mb = [" " if x==_mb_unfinished else x for x in board_state[-1]]
        pretty_mb = ["X" if x==_mb_X else x for x in pretty_mb]
        pretty_mb = ["O" if x==_mb_O else x for x in pretty_mb]
        pretty_mb = ["." if x==_mb_available else x for x in pretty_mb]
        pretty_mb = ["D" if x==_mb_draw else x for x in pretty_mb]
        print("|".join(pretty_mb[:3]), )
        print("------")
        print("|".join(pretty_mb[3:6]))
        print("------")
        print("|".join(pretty_mb[6:]))

class UltimateTicTacToeGameSpec(BaseGameSpec):
    def __init__(self):
        self.available_moves = available_moves
        self.has_winner = has_winner
        self.new_board = _new_board
        self.apply_move = apply_move
        self.evaluate = evaluate
        self.play_game = play_game

    def get_random_player_func(self):
        return random_player

    def get_manual_player_func(self):
        return manual_player

    def monte_carlo_player(self, board_state, side, uct, number_of_samples):
        if uct:
            _, move = mc.monte_carlo_tree_search_uct(self, board_state, side, number_of_samples)
        else:
            _, move = mc.monte_carlo_tree_search(self, board_state, side, number_of_samples)
        return move

    def get_monte_carlo_player_func(self, number_of_samples, uct = True):
        return partial(self.monte_carlo_player, uct = uct, number_of_samples = number_of_samples)

    def board_dimensions(self):
        return 10, 9

    def outputs(self):
        return 9*9

    def flat_move_to_tuple(self, move_index):
        return int(move_index / 9), move_index % 9

    def tuple_move_to_flat(self, tuple_move):
        return tuple_move[0] * 9 + tuple_move[1]



if __name__ == '__main__':
    # example of playing a game
    play_game(random_player, random_player, log=True)
