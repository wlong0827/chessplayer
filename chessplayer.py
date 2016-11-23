# Author: James Baskerville, Vinay Iyengar, Will Long
# MRU: Nov 23 2016
# Description: Our CS 182 Final Project is a python program that 
#	utilizes python-chess to analyze and play chess

# python-chess docs: https://python-chess.readthedocs.io/en/latest/

import chess
from chess import svg

f = open('out.svg', 'w')

board = chess.Board()

board.push_san("e4")
board.push_san("e5")

f.write(chess.svg.board(board = board))

f.close()
