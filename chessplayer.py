# Author: James Baskerville, Vinay Iyengar, Will Long
# MRU: Nov 23 2016
# Description: Our CS 182 Final Project is a python program that 
#   utilizes python-chess to analyze and play chess

# python-chess docs: https://python-chess.readthedocs.io/en/latest/

import chess
from chess import svg
from chess import pgn
from chess import polyglot
from chess import syzygy
import random
from subprocess import call
import time
import os
import sys

class ChessPlayer:
    """
    The ChessPlayer class is the parent class for all of
    our various engines. It includes several more general
    functions and checks that are shared across engines.
    """

    # INITIALIZATION
    # -------------------------
    def __init__(self, outfile):
        self.outfile = outfile
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.reader = None
        self.tbs = None
        self.half_moves = 0

    def initOpeningBook(self, book="gm2600"):
        path = "opening_books/" + book + ".bin"
        if os.path.isfile(path):
            self.reader = chess.polyglot.open_reader(path)
        else:
            self.reader = None

    def initTablebases(self, directory="syzygy/3-4-5/"):
        if os.path.isdir(directory):
            self.tbs = chess.syzygy.open_tablebases(directory)
        else:
            self.tbs = None

    # SIMPLE FUNCTIONS
    # -------------------------
    def isGameOver(self, board):
        return board.is_game_over()

    def writeBoard(self, outfile, chessboard):
        f = open(outfile, 'w') 
        f.write(chess.svg.board(board = chessboard))
        f.close()

    def printGame(self):
        print self.game

    def exit(self):
        if self.reader:
            self.reader.close()
        if self.tbs:
            self.tbs.close()

    # move is a chess.Move object
    def getMoveValue(self, move):
        target = move.to_square
        piece = self.board.piece_at(target)
        
        if piece:
            # piece could be upper or lower case
            return self.values[str(piece).upper()]
        return 0

    # naive evaluation function -- just material
    def getBoardValue(self, board):
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
        return value

    def getNumPieces(self, board):
        count = 0
        for square in range(64):
            if board.piece_at(square):
                count += 1
        return count

    def getEndgameMoves(self, board, legal_moves):
        wins = []
        losses = []
        draws = []
        for move in legal_moves:
            board.push(move)
            dtz = self.tbs.probe_dtz(board)
            if dtz == None:
                pass
            elif dtz > 0:
                # winning move
                wins.append((move, dtz))
            elif dtz == 0:
                # drawing move
                draws.append(move)
            else:
                # losing move
                losses.append((move, dtz))
            board.pop()

        half_moves_left = 100 - board.halfmove_clock
        if wins:
            if dtz >= half_moves_left - 4:
                moves = sorted(wins, key=lambda win: win[1])[0][0]
            else:
                moves = [win[0] for win in wins]
            moves = wins
        elif draws:
            moves = draws
        elif losses:
            moves = sorted(losses, key=lambda loss: loss[1], reverse=True)[0][0]
        else:
            moves = []
        return moves

    def getGoodMoves(self, board):
        moves = []
        legal_moves = list(board.legal_moves)
        # endgame tablebases
        if self.tbs and self.getNumPieces(board) <= 5:
            moves = self.getEndgameMoves(board, moves)
            #if moves:
                #print "used endgame:", moves
        # opening book
        elif self.reader and self.board.fullmove_number <= 10:
            moves = [entry.move() for entry in self.reader.find_all(board)
                        if entry.move() in legal_moves]
            #print "used opening:", moves
        else:
            moves = legal_moves
            #print "standard"

        if moves:
            return moves
        else:
            #print "jk, used standard"
            return legal_moves

    # overwritten by engines
    def move(self, board):
        pass

    # def play(self):
    #     board = self.board

    #     while not self.isGameOver(board):
    #         move = self.move(board)
    #         board.push(move)

    #     self.game = chess.pgn.Game.from_board(self.board)

    #     self.writeBoard(self.file)

"""
    RandomPlayer plays a random legal move at each step until
    the game is over, then writes the final position to self.file
"""
class RandomPlayer(ChessPlayer):

    def move(self, board):
        legal_moves = list(board.legal_moves)
        return legal_moves[random.randint(0, len(legal_moves) - 1)]


class GreedyPlayer(ChessPlayer):

    def __init__(self, outfile, book="", directory=""):
        self.outfile = outfile
        self.board = chess.Board()

        # I can't seem to find a piece value dict so I wrote one
        self.values = {'P': 2, 'R': 5, 'N': 3, 'B': 3.5, 'Q': 9, 'K': 5}

        if isinstance(book, basestring) and book != "":
            self.initOpeningBook(book)
        elif book == True:
            self.initOpeningBook()
        else:
            self.reader = None

        if isinstance(directory, basestring) and directory != "":
            self.initTablebases(directory)
        elif directory == True:
            self.initTablebases()
        else:
            self.tbs = None

    def move(self, board):   
        # best_move = (move, value)
        moves = self.getGoodMoves(board)
        best_move = (None, -float('inf'))

        # takes move that maximizes immediate board value
        for move in moves:
            board.push(move)
            value = self.getBoardValue(board)
            if value > best_move[1]:
                best_move = (move, value)
            board.pop()
        
        return best_move[0]

class MinimaxPlayer(ChessPlayer):

    def __init__(self, outfile, player=chess.WHITE, book="", directory=""):
        self.outfile = outfile
        self.board = chess.Board()
        self.values = {'P': 2, 'R': 5, 'N': 3, 'B': 3.5, 'Q': 9, 'K': 1000}
        self.calculations = 0
        self.legal_moves = 0
        # start at 0 if white, 1 if black
        self.half_moves = 0 if player == chess.WHITE else 1

        if isinstance(book, basestring) and book != "":
            self.initOpeningBook(book)
        elif book == True:
            self.initOpeningBook()
        else:
            self.reader = None

        if isinstance(directory, basestring) and directory != "":
            self.initTablebases(directory)
        elif directory == True:
            self.initTablebases()
        else:
            self.tbs = None

    def maxMove(self, board, depth, player, alpha, beta):
        moves = self.getGoodMoves(board)
        value = (-float('inf'), None)
        for move in moves:
            board_copy = board.copy()
            #self.calculations += 1
            board_copy.push(move)
            #value = max(value, self.move(board_copy, depth - 1, player))
            new_value = self.value(board_copy, depth - 1, player)
            if new_value[0] > value[0]:
                value = new_value
            if value[0] >= beta:
                #print "max cut"
                return value
            alpha = max(alpha, value[0])
        #print "max value", value[0]
        return value

    def minMove(self, board, depth, player, alpha, beta):
        moves = self.getGoodMoves(board)
        value = (float('inf'), None)
        for move in moves:
            board_copy = board.copy()
            #self.calculations += 1
            board_copy.push(move)
            # value = min(value, self.move(board_copy, depth - 1, player))
            new_value = self.value(board_copy, depth - 1, player)
            if new_value[0] < value[0]:
                value = new_value
            if value[0] <= alpha:
                #print "min cut"
                return value
            beta = min(beta, value[0])
        #print "min value", value[0]
        return value

    def value(self, board, depth, player):
        alpha = -float('inf')
        beta = float('inf')

        if depth == 0 or self.isGameOver(board):
            value = self.getBoardValue(board)
            self.calculations += 1
            return (value, board.move_stack)
        if board.turn == player: 
            #print "asking max"
            return self.maxMove(board, depth, player, alpha, beta)
        else:
            #print "asking min"
            return self.minMove(board, depth, player, alpha, beta)

    def move(self, board, depth=3, player=chess.WHITE):
        value, moves = self.value(board, depth, player)
        move = moves[self.half_moves]
        #print "final value", value
        #print "calcs", self.calculations
        self.half_moves += 2
        return move

class GreedyNNPlayer(GreedyPlayer):

    def getBoardValue(self, board):
        pass

class MinMaxNNPlayer(MinimaxPlayer):

    def getBoardValue(self, board):
        pass

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
def PlayAgents(WhitePlayer, BlackPlayer, debug=False):
    board = WhitePlayer.board
    if debug:
        print board
        print '\n'

    while not WhitePlayer.isGameOver(board):
        WhitePlayer.writeBoard('out.svg', board)

        # Update both Players' boards
        WhitePlayer.board = board
        BlackPlayer.board = board

        if board.turn == chess.WHITE:
            if debug:
                print "Move:", board.fullmove_number
            move = WhitePlayer.move(board)
            if debug:
                print "White's move: {}".format(move)
            board.push(move)
        else:
            move = BlackPlayer.move(board)
            if debug:
                print "Black's move: {}".format(move)
            board.push(move)

        if debug:
            print board
            print '\n'
            time.sleep(1)

    if debug:
        print board
        print board.result()
        print chess.pgn.Game.from_board(board)

    return board.result()
