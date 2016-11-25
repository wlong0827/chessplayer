# Author: James Baskerville, Vinay Iyengar, Will Long
# MRU: Nov 23 2016
# Description: Our CS 182 Final Project is a python program that 
#   utilizes python-chess to analyze and play chess

# python-chess docs: https://python-chess.readthedocs.io/en/latest/

import chess
from chess import svg
from chess import pgn
import random
from subprocess import call
import time

class Tree:
    def __init__(self, cargo):
        self.cargo = cargo
        self.children = []

    def addChild(self, cargo):
    	self.children.append(cargo)

class ChessPlayer:
    """
    The ChessPlayer class contains all of the chess engines
    that we will write as well as any helper functions useful
    to writing the engines. It opens the svg file to write
    our output to as well as the chess board itself
    """
    def __init__(self, outfile):
        # outfile should be a path to a svg
        self.file = open(outfile, 'w')
        self.board = chess.Board()
        self.game = chess.pgn.Game()

    def isGameOver(self):
        return self.board.is_game_over()

    # move is a chess.Move object
    def getMoveValue(self, move):
        target = move.to_square
        piece = self.board.piece_at(target)
        
        if piece:
            # piece could be upper or lower case
            return self.values[str(piece).upper()]

        return 0

    # overwritten by engines
    def move(self):
        pass

    def write(self, file):
        self.file.write(chess.svg.board(board = self.board))
        self.file.close()

    def play(self):
        board = self.board

        while not self.isGameOver():
            legal_moves = list(board.legal_moves)
            move = self.move(legal_moves)
            board.push(move)

        self.game = chess.pgn.Game.from_board(self.board)

        self.write(self.file)

    def printGame(self):
        print self.game

"""
    RandomPlayer plays a random legal move at each step until
    the game is over, then writes the final position to self.file
"""
class RandomPlayer(ChessPlayer):

    def move(self, legal_moves):
        return legal_moves[random.randint(0, len(legal_moves) - 1)]


class GreedyPlayer(ChessPlayer):

    def __init__(self, outfile):
        self.file = open(outfile, 'w')
        self.board = chess.Board()

        # I can't seem to find a piece value dict so I wrote one
        self.values = {'P': 1, 'R': 5, 'N': 3, 'B': 3, 'Q': 9, 'K': 5}

    def move(self, legal_moves):   
        # best_move = (move, value)
        best_move = (None, -float('inf'))

        for move in legal_moves:
            value = self.getMoveValue(move)
            if value > best_move[1]:
                best_move = (move, value)

        # If we can't take anything, return random move
        if best_move[1] == 0:
            return legal_moves[random.randint(0, len(legal_moves) - 1)]
        
        return best_move[0]

class MinimaxPlayer(ChessPlayer):

	def __init__(self, outfile):
		self.file = open(outfile, 'w')
		self.board = chess.Board()
		self.values = {'P': 1, 'R': 5, 'N': 3, 'B': 3, 'Q': 9, 'K': 1000}
		self.tree = Tree(self.board)
		self.calculations = 0

	def boardValue(self, board):
		value = 0
		for square in range(64):
			piece = board.piece_at(square)
			if piece:
				p = str(piece)
				
				if p.isupper():
					value += self.values[p]
				else:
					value -= self.values[p.upper()]
		return value

	def maxMove(self, depth, player, board):
		legal_moves = list(board.legal_moves)
		value = -float('inf')
		print len(legal_moves)
		for move in legal_moves:
			board_copy = board.copy()
			self.calculations += 1
			board_copy.push(move)
    		value = max(value, self.move(board_copy, depth - 1, player))
		return value

	def minMove(self, depth, player, board):
		legal_moves = list(board.legal_moves)
		print len(legal_moves)
		value = float('inf')
		for move in legal_moves:
			board_copy = board.copy()
			self.calculations += 1
			board_copy.push(move)
    		value = min(value, self.move(board_copy, depth - 1, player))
		return value

	def move(self, board, depth = 5, player = True):
		print board
		if depth == 0 or self.isGameOver():
			value = self.boardValue(board)
			board.pop()
			print "value", value
			print "boards", self.calculations
			return value
		if board.turn == player: 
			return self.maxMove(depth, player, board)
		else:
			return self.minMove(depth, player, board)

class HumanPlayer(ChessPlayer):

	def move(self, legal_moves):
		move = raw_input("Input your move\n")
		formatted_move = self.board.parse_san(str(move))

		while formatted_move not in legal_moves:
			move = raw_input("Incorrect input. Try again\n")

		return formatted_move

#class ClassificationPlayer(ChessPlayer):
#   def move(self):

def PlayAgents(Player1, Player2):
	board = Player1.board

	while not Player1.isGameOver():
		legal_moves = list(board.legal_moves)
		print board
		print "\n"

		if not board.turn:
			move = Player1.move(legal_moves)
			board.push(move)
		else:
			move = Player2.move(legal_moves)
			board.push(move)

		#print board
		#time.sleep(1)

	print board.result()
	Player1.write('out.svg')

"""
-------------- Test Code -------------------------
"""
# rp = RandomPlayer('out.svg')
# gp = GreedyPlayer('out.svg')
# hp = HumanPlayer('out.svg')
# PlayAgents(gp, hp)
mp = MinimaxPlayer('out.svg')
mp.move(mp.board)

# rp = RandomPlayer('random.svg')
# rp.play()
# rp.printGame()
# call(["open", "random.svg"])

# gp = GreedyPlayer('greedy.svg')
# gp.play()
# gp.printGame()
# call(["open", "greedy.svg"])
