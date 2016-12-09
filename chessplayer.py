# Author: James Baskerville, Vinay Iyengar, Will Long
# MRU: Dec 07 2016
# Description: Our CS 182 Final Project is a python program that 
#   utilizes python-chess to analyze and play chess

# python-chess docs: https://python-chess.readthedocs.io/en/latest/

import chess
from chess import svg
from chess import pgn
from chess import polyglot
from chess import syzygy
import random
import os

# depth for search
MAX_DEPTH = 3

# stuff for evaluation function
import pstbs
MATE_VALUE = 10000
CASTLE_VALUE = 100

# weights
mat_weight = 30.0
pos_weight = 4.0
cas_weight = 50.0

class ChessPlayer:
    """
    The ChessPlayer class is the parent class for all of
    our various engines. It includes several more general
    functions and checks that are shared across engines.
    """

    # INITIALIZATION
    # -------------------------
    def __init__(self, outfile, fen=chess.STARTING_FEN):
        self.outfile = outfile
        self.board = chess.Board(fen)
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
    # ----------------
    def isGameOver(self, board):
        return board.is_game_over(claim_draw=True)

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
            return self.values[piece.piece_type]
        return 0

    def getNumPieces(self, board):
        count = 0
        for square in chess.SQUARES:
            if board.piece_at(square):
                count += 1
        return count

    # EVALUATION FUNCTION
    # -------------------

    # evaluation function: positive if White is winning
    def getBoardValue(self, board):
        if board.is_checkmate():
            if board.result() == "1-0":
                return MATE_VALUE
            else:
                return -MATE_VALUE

        material, endgame = self.evalMaterial(board)

        # only evaluate position after opening
        if board.fullmove_number > 5:
            position = self.evalPosition(board, endgame)
        else:
            position = 0.0

        #castle = self.evalCastle(board)

        # normalize (to range -1 to 1)
        material = material / 39.0
        position = position / (self.getNumPieces(board) * 50.0)
        #castle = castle / CASTLE_VALUE

        #print "material total:", (mat_weight * material)
        #print "position total:", (pos_weight * position)
        #print "castle total:", (cas_weight * castle)
        #print ""

        return ((mat_weight * material) + 
                (pos_weight * position))# +
                #(cas_weight * castle))

    def evalMaterial(self, board):
        white_mat = 0.0
        black_mat = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece != chess.KING:
                if piece.color == chess.WHITE:
                    white_mat += self.values[piece.piece_type]
                else:
                    black_mat += self.values[piece.piece_type]

        material = white_mat - black_mat
        endgame = True if white_mat <= 13 and black_mat <= 13 else False

        return (material, endgame)

    def lookupPST(self, square, piece, endgame):
        # king PST depends on game stage and color
        if piece.piece_type == chess.KING:
            if endgame:
                val = pstbs.tables['end K'][square]
            elif piece.color == chess.WHITE:
                val = pstbs.tables['white mid K'][square]
            else:
                val = pstbs.tables['black mid K'][square]

        # pawn PST depends on color
        elif piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE:
                val = pstbs.tables['white P'][square]
            else:
                val = pstbs.tables['black P'][square]

        # all other pieces have symmetrical PSTs
        else:
            val = pstbs.tables[piece.piece_type][square]

        return val

    def evalPosition(self, board, endgame):
        white_pos = 0.0
        black_pos = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    white_pos += self.lookupPST(square, piece, endgame)
                else:
                    black_pos += self.lookupPST(square, piece, endgame)

        position = white_pos - black_pos

        return position

    # def evalCastle(self, board):
    #     move = board.pop()
    #     val = 0

    #     if board.is_castling(move):
    #         if board.turn == chess.WHITE:
    #             val = CASTLE_VALUE
    #         else:
    #             val = -CASTLE_VALUE

    #     board.push(move)
    #     return val

    # def evalKingSafety(self, board):
    #     pass


    # MOVING FUNCTIONS
    # ----------------

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
            if moves:
                print "used endgame:", moves
        # opening book
        elif self.reader and self.board.fullmove_number <= 10:
            moves = [entry.move() for entry in self.reader.find_all(board)
                        if entry.move() in legal_moves]
        else:
            moves = legal_moves

        if moves:
            return moves
        else:
            return legal_moves

    # overwritten by engines
    def move(self, board):
        pass

"""
    RandomPlayer plays a random legal move at each step until
    the game is over, then writes the final position to self.file
"""
class RandomPlayer(ChessPlayer):

    def move(self, board):
        legal_moves = list(board.legal_moves)
        return legal_moves[random.randint(0, len(legal_moves) - 1)]


class GreedyPlayer(ChessPlayer):

    def __init__(self, outfile, fen=chess.STARTING_FEN, player=chess.WHITE, book="", directory=""):
        self.outfile = outfile
        self.board = chess.Board(fen)

        # I can't seem to find a piece value dict so I wrote one
        self.values = { chess.PAWN : 1, 
                        chess.ROOK : 5, 
                        chess.KNIGHT : 3, 
                        chess.BISHOP : 3, 
                        chess.QUEEN: 9, 
                        chess.KING : MATE_VALUE }
        self.value_sign = 1 if player == chess.WHITE else -1

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
            value = self.getBoardValue(board) * self.value_sign
            if value > best_move[1]:
                best_move = (move, value)
            board.pop()

        # act randomly if no value change is possible
        if self.getBoardValue(board) == best_move[1]:
            return moves[random.randint(0, len(moves) - 1)]
        
        return best_move[0]

class MinimaxPlayer(ChessPlayer):

    def __init__(self, outfile, fen=chess.STARTING_FEN, player=chess.WHITE, book="", directory=""):
        self.outfile = outfile
        self.board = chess.Board(fen)
        self.values = { chess.PAWN : 1, 
                        chess.ROOK : 5, 
                        chess.KNIGHT : 3, 
                        chess.BISHOP : 3,
                        chess.QUEEN: 9,
                        chess.KING : MATE_VALUE }
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
            assert(board.is_legal(move))
            board_copy = board.copy()
            #self.calculations += 1
            board_copy.push(move)
            #value = max(value, self.move(board_copy, depth - 1, player))
            new_val = self.value(board_copy, depth - 1, player, alpha, beta)
            if new_val[0] > value[0]:
                value = new_val
            elif (new_val[0] == value[0] and 
                len(new_val[1]) < len(value[1]) and new_val[0] > 0):
                value = new_val
            if value[0] >= beta:
                return value
            alpha = max(alpha, value[0])
            if (value[0] == MATE_VALUE and 
                board_copy.is_checkmate() and
                depth == MAX_DEPTH):
                print "yes"
                return value
        return value

    def minMove(self, board, depth, player, alpha, beta):
        moves = self.getGoodMoves(board)
        value = (float('inf'), None)
        for move in moves:
            assert(board.is_legal(move))
            board_copy = board.copy()
            board_copy.push(move)
            new_val = self.value(board_copy, depth - 1, player, alpha, beta)
            if new_val[0] < value[0]:
                value = new_val
            elif (new_val[0] == value[0] and 
                len(new_val[1]) < len(value[1]) and new_val[0] < 0):
                value = new_val
            if value[0] <= alpha:
                return value
            beta = min(beta, value[0])
            if (value[0] == -MATE_VALUE and 
                board_copy.is_checkmate() and
                depth == MAX_DEPTH):
                return value
        return value

    def value(self, board, depth, player, alpha, beta):
        if depth == 0 or board.is_game_over():
            value = self.getBoardValue(board)
            self.calculations += 1
            return (value, board.move_stack)
        if board.turn == player: 
            return self.maxMove(board, depth, player, alpha, beta)
        else:
            return self.minMove(board, depth, player, alpha, beta)

    def move(self, board, depth=MAX_DEPTH, player=chess.WHITE):
        alpha = -float('inf')
        beta = float('inf')
        value, moves = self.value(board, depth, player, alpha, beta)
        print "minimax value: ", value
        move = moves[self.half_moves]
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
                move = board.parse_uci(inp)
                if move == chess.Move.null():
                    inp = raw_input("Illegal move. Try again.\n")
                    continue
            except ValueError:
                try:
                    move = board.parse_san(inp)
                    if move == chess.Move.null():
                        inp = raw_input("Illegal move. Try again.\n")
                        continue
                except ValueError:
                    inp = raw_input("Invalid input. Try again.\n")
                    continue

            break

        return move
