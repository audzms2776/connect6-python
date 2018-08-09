#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from logger import MoveLogger
from rules import Referee
from bot import RandomBot, Player, TFBot

# constants
STONE_CHAR = ['.', 'O', 'X']
STONE_NAME = ['', 'White (O)', 'Black (X)']
CHAR_TO_X = {chr(ord('A') + i): i for i in range(19)}
X_TO_CHAR = {i: chr(ord('A') + i) for i in range(19)}
win_list = []


# console helper methods
def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def darktext(str):
    return str if os.name == 'nt' else '\x1b[0;30m{}\x1b[0m'.format(str)


def draw_board(board, player=0, nth_move=0):
    # cls()
    print('Move : {}'.format(nth_move))
    print('{} turn.'.format(STONE_NAME[player]))
    print()
    print('       A B C D E F G H I J K L M N O P Q R S ')
    print('     +---------------------------------------+')

    for y in range(19):
        print('  {:>2d} |'.format(y + 1), end='')  # line no.
        for x in range(19):
            stone = board[y][x]
            if stone != 0:
                print(' ' + STONE_CHAR[board[y][x]], end='')
            else:
                print(darktext(' ' + X_TO_CHAR[x].lower()), end='')
        print(' |')

    print('     +---------------------------------------+')
    print()


def exit_game(logger: MoveLogger, won_bot=None):
    if won_bot is not None:
        logger.log_winner(won_bot.player)
        print('{} ({}) won!!'.format(STONE_NAME[won_bot.player], won_bot.bot_kind))
    else:
        print('No one won.')

    logger.save_to_file()


def main(episode, bots):
    bots[1].reset_data()
    # to align index with player variable.
    bot_set = [None] + bots

    board = [[0 for x in range(19)] for y in range(19)]
    referee = Referee(board)

    nth_move = 1
    player = 2  # 1=white 2=black. black moves first
    player_moved_count = 1  # at first time, black can only move once.
    logger = MoveLogger()

    while True:
        # draw_board(board, player, nth_move)
        # input loop.
        while True:
            try:
                x, y = bot_set[player].move(board, nth_move)
                able_to_place, msg = referee.can_place(x, y)
                if not able_to_place:
                    # print('{} shit'.format(player))
                    continue
                break

            except KeyboardInterrupt:
                print('\n' + 'Bye...')
                exit_game(logger)
                return

            except Exception as e:
                raise e
                continue

        # place stone
        board[x][y] = player
        # logger.log(x, y, player)
        referee.update(x, y, player)

        won_player = referee.determine()

        if won_player == 1:
            bot_set[2].rewards[-1][0] = -1.0
        elif won_player == 2:
            reward = 1.0
        else:
            reward = 0

        if bot_set[player].player == 2:
            bot_set[player].reward_sum += reward
            bot_set[player].rewards = np.vstack([bot_set[player].rewards, reward])

        if won_player is not None:
            win_list.append(won_player)
            episode_loss = bot_set[2].train()
            exit_game(logger, bot_set[won_player])

            # print("[Episode {}] Won player: {} Loss: {}".format(episode, won_player, episode_loss))
            return

        player_moved_count += 1
        if player_moved_count == 2:
            # Change turn : a player can move 2 times per turn.
            nth_move += 1
            player_moved_count = 0
            player = 1 if player == 2 else 2


if __name__ == '__main__':
    print('Choose player slot. (1=Player 2=AI)')

    whitebot = RandomBot(1)
    blackbot = TFBot(2)

    for episode in range(1000):
        main(episode, [whitebot, blackbot])

        if episode % 100 == 0:
            print('random win count: {}'.format(win_list[-100:].count(1)))
            print('tf     win count: {}'.format(win_list[-100:].count(2)))        
    
    print('random win count: {}'.format(win_list.count(1)))
    print('tf     win count: {}'.format(win_list.count(2)))
