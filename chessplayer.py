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
        self.half_moves = 0

    def isGameOver(self, board):
        return board.is_game_over()

    # move is a chess.Move object
    def getMoveValue(self, move):
        target = move.to_square
        piece = self.board.piece_at(target)
        
        if piece:
            # piece could be upper or lower case
            return self.values[str(piece).upper()]

        return 0

    # overwritten by engines
    def move(self, board):
        pass

    def write(self, file, chessboard):
    	f = open(file, 'w')
        f.write(chess.svg.board(board = chessboard))
        f.close()

    def play(self):
        board = self.board

        while not self.isGameOver(board):
            move = self.move(board)
            board.push(move)

        self.game = chess.pgn.Game.from_board(self.board)

        self.write(self.file)

    def printGame(self):
        print self.game

    def exit(self):
        close(self.file)

"""
    RandomPlayer plays a random legal move at each step until
    the game is over, then writes the final position to self.file
"""
class RandomPlayer(ChessPlayer):

    def move(self, board):
        legal_moves = list(board.legal_moves)
        return legal_moves[random.randint(0, len(legal_moves) - 1)]


class GreedyPlayer(ChessPlayer):

    def __init__(self, outfile):
        self.file = open(outfile, 'w')
        self.board = chess.Board()

        # I can't seem to find a piece value dict so I wrote one
        self.values = {'P': 1, 'R': 5, 'N': 3, 'B': 3, 'Q': 9, 'K': 5}

    def move(self, board):   
        # best_move = (move, value)
        legal_moves = list(board.legal_moves)
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

    def __init__(self, outfile, player = chess.WHITE):
        self.file = open(outfile, 'w')
        self.board = chess.Board()
        self.values = {'P': 1, 'R': 5, 'N': 3, 'B': 3, 'Q': 9, 'K': 1000}
        self.calculations = 0
        # start at 0 if white, 1 if black
        self.half_moves = int(not player)  

    def boardValue(self, board):
        value = 0
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                p = piece.symbol()
                if p.isupper():
                    value += self.values[p]
                    # print value
                else:
                    value -= self.values[p.upper()]
                    # print value
        # if value != 0:
        #     print "VALUEEEE"
        return value

    def maxMove(self, board, depth, player):
        legal_moves = list(board.legal_moves)
        
        value = (-float('inf'), None)
        # print len(legal_moves)
        for move in legal_moves:
            board_copy = board.copy()
            self.calculations += 1
            board_copy.push(move)
            #value = max(value, self.move(board_copy, depth - 1, player))
            new_value = self.value(board_copy, depth - 1, player)
            if new_value[0] > value[0]:
            	value = new_value
        #print "max value", value[0]
        return value

    def minMove(self, board, depth, player):
        legal_moves = list(board.legal_moves)
        # print len(legal_moves)
        value = (float('inf'), None)
        for move in legal_moves:
            board_copy = board.copy()
            self.calculations += 1
            board_copy.push(move)
            # value = min(value, self.move(board_copy, depth - 1, player))
            new_value = self.value(board_copy, depth - 1, player)
            if new_value[0] < value[0]:
            	value = new_value
        #print "min value", value[0]
        return value

    def value(self, board, depth, player):
        if depth == 0 or self.isGameOver(board):
            value = self.boardValue(board)
            return (value, board.move_stack)
        if board.turn == player: 
        	#print "asking max"
        	return self.maxMove(board, depth, player)
        else:
        	#print "asking min"
        	return self.minMove(board, depth, player)

    def move(self, board, depth = 3, player = chess.WHITE):
    	value, moves = self.value(board, depth, player)
    	move = moves[self.half_moves]
    	print moves
    	print move
    	print "final value", value
    	self.half_moves += 2
    	return move

class ReinforcementLearningPlayer(ChessPlayer):
    def __init__(self):
        self.file = open(outfile, 'w')
        self.board = chess.Board()

class HumanPlayer(ChessPlayer):

    def move(self, board):
        legal_moves = list(board.legal_moves)
      	moves_str = "Choose a move: "
        for i, move in enumerate(legal_moves):
            if i == len(legal_moves) - 1:
                moves_str += "{}.".format(move)
            else:
                moves_str += "{}, ".format(move)
        print moves_str + "\n"
        
        inp = raw_input("Input your move:\n")

        while True:
            try:
                move = chess.Move.from_uci(inp)
            except ValueError:
                inp = raw_input("Invalid input. Try again.\n")
                continue
            if move not in legal_moves:
                inp = raw_input("Illegal move. Try again.\n")
            else:
                break

        return move


"""
    PlayAgents(Player1, Player2)
    Plays a game of chess, with Player1 (white) vs. Player2 (black)
"""

def PlayAgents(BlackPlayer, WhitePlayer):
    board = BlackPlayer.board

    while not BlackPlayer.isGameOver(board):
    	BlackPlayer.write('out.svg', board)
        print board
        print "\n"
        BlackPlayer.board = board
        WhitePlayer.board = board

        if board.turn == chess.BLACK:
            move = BlackPlayer.move(board)
            print "Black's move: {}".format(move)
            board.push(move)
        else:
            move = WhitePlayer.move(board)
            print "White's move: {}".format(move)
            board.push(move)

        #print board
        #time.sleep(1)

    print board.result()
    BlackPlayer.exit()
    WhitePlayer.exit()

"""
-------------- Test Code -------------------------
"""
# rp = RandomPlayer('out.svg')
gp = GreedyPlayer('out.svg')
hp = HumanPlayer('out.svg')
<<<<<<< HEAD
mp1 = MinimaxPlayer('out.svg', chess.BLACK)
mp2 = MinimaxPlayer('out.svg', chess.WHITE)

PlayAgents(mp1, mp2)
=======
mp = MinimaxPlayer('out.svg', chess.WHITE)

PlayAgents(hp, mp)
>>>>>>> c6eaa389944a938fbcec70cdb3982d610176d6ab

# mp.move()

# rp = RandomPlayer('random.svg')
# rp.play()
# rp.printGame()
# call(["open", "random.svg"])

# gp = GreedyPlayer('greedy.svg')
# gp.play()
# gp.printGame()
# call(["open", "greedy.svg"])
