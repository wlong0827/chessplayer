# Author: James Baskerville, Vinay Iyengar, Will Long
# MRU: Nov 23 2016
# Description: Our CS 182 Final Project is a python program that 
#	utilizes python-chess to analyze and play chess

# python-chess docs: https://python-chess.readthedocs.io/en/latest/

import chess
from chess import svg
import random

class ChessPlayer:
	"""
	The ChessPlayer class contains all of the chess engines
	that we will write as well as any helper functions useful
	to writing the engines. It includes the svg file to write
	our output to as well as the chess board itself
	"""

	def __init__(self):
		self.file = open('out.svg', 'w')
		self.board = chess.Board()

		# I can't seem to find a piece value dict so I wrote one
		self.values = {'P': 1, 'R': 5, 'N': 3, 'B': 3, 'Q': 9, 'K': 5}

	def GameOver(self):
		return self.board.is_game_over()

	# move is a chess.Move object
	def MoveValue(self, move):
		target = move.to_square()
		piece = self.board.piece_at(target)
		
		if piece:
			# piece could be upper or lower case
			return self.values[piece.upper()]

		return 0

	"""
	RandomPlayer plays a random legal move at each step until
	the game is over, then writes the final position to self.file
	"""
	def RandomPlayer(self):
		board = self.board

		while not self.GameOver():
			legal_moves = list(board.legal_moves)
			rand_move = random.randint(0, len(legal_moves) - 1)
			board.push(legal_moves[rand_move])

		self.file.write(chess.svg.board(board = self.board))
		self.file.close()

cp = ChessPlayer()
cp.RandomPlayer()
