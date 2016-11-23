# Author: James Baskerville, Vinay Iyengar, Will Long
# MRU: Nov 23 2016
# Description: Our CS 182 Final Project is a python program that 
#	utilizes python-chess to analyze and play chess

# python-chess docs: https://python-chess.readthedocs.io/en/latest/

import chess
from chess import svg
from chess import pgn
import random

class ChessPlayer:

	def __init__(self):
		self.file = open('out.svg', 'w')
		self.board = chess.Board()

	def GameOver(self):
		return self.board.is_game_over()

	def RandomPlayer(self):
		board = self.board
		#game_moves = []

		while not self.GameOver():
			legal_moves = list(board.legal_moves)
			rand_move = random.randint(0, len(legal_moves) - 1)
			#game_moves.append(legal_moves[rand_move])
			board.push(legal_moves[rand_move])

		game = chess.pgn.Game.from_board(self.board)
		print game
		self.file.write(chess.svg.board(board=self.board))
		self.file.close()

#class ClassificationPlayer(ChessPlayer):
#	def move(self):


cp = ChessPlayer()
cp.RandomPlayer()
