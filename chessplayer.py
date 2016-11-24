# Author: James Baskerville, Vinay Iyengar, Will Long
# MRU: Nov 23 2016
# Description: Our CS 182 Final Project is a python program that 
#	utilizes python-chess to analyze and play chess

# python-chess docs: https://python-chess.readthedocs.io/en/latest/

import chess
from chess import svg
import random
import time
import sys

class Tree:
    def __init__(self, cargo, left=None, right=None):
        self.cargo = cargo
        self.left  = left
        self.right = right

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
		target = move.to_square
		piece = self.board.piece_at(target)
		
		if piece:
			# piece could be upper or lower case
			return self.values[str(piece).upper()]

		return 0

	def RandomMove(self, legal_moves):
		rand_move = random.randint(0, len(legal_moves) - 1)

		return legal_moves[rand_move]

	def GreedyMove(self, legal_moves):
		
		# best_move = (move, value)
		best_move = (None, -float('inf'))

		for move in legal_moves:
			value = self.MoveValue(move)
			if value > best_move[1]:
				best_move = (move, value)

		# If we can't take anything, return random move
		if best_move[1] == 0:
			return self.RandomMove(legal_moves)
		
		return best_move[0]

	def Write(self):
		self.file.write(chess.svg.board(board = self.board))
	
	def Close(self):	
		self.file.close()

	"""
	RandomPlayer plays a random legal move at each step until
	the game is over, then writes the final position to self.file
	"""
	def RandomPlayer(self):
		board = self.board

		while not self.GameOver():
			legal_moves = list(board.legal_moves)
			move = self.RandomMove(legal_moves)
			board.push(move)

		print board.result()
		self.Write()
		self.Close()

	def GreedyPlayer(self):
		board = self.board

		while not self.GameOver():
			legal_moves = list(board.legal_moves)
			move = self.GreedyMove(legal_moves)
			board.push(move)

		print board.result()
		self.Write()
		self.Close()

	def RandomvsGreedy(self):
		board = self.board

		while not self.GameOver():
			legal_moves = list(board.legal_moves)

			if not board.turn:
				move = self.RandomMove(legal_moves)
				board.push(move)
			else:
				move = self.GreedyMove(legal_moves)
				board.push(move)

			print board

			time.sleep(1)

		print board.result()
		self.Write()
		self.Close()

	def MinimaxPlayer(self, depth):
		board = self.board

		while not self.GameOver():
			legal_moves = list(board.legal_moves)
			move = self.GreedyMove(legal_moves)
			board.push(move)

		print board.result()
		self.Write()

cp = ChessPlayer()
cp.RandomvsGreedy()
