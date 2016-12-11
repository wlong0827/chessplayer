# -*- encoding: utf-8 -*-

import chess
from prettytable import PrettyTable

import chessplayer
import runners

"""
Create Chessplayer Instances
----------------------------
"""

rp = chessplayer.RandomPlayer('rp-out.svg')

gpw = chessplayer.GreedyPlayer('gp-out.svg')
gpb = chessplayer.GreedyPlayer('gp-out.svg', player=chess.BLACK)
gpw_book = chessplayer.GreedyPlayer('gp-book-out.svg', book=True)
gpb_book = chessplayer.GreedyPlayer('gp-book-out.svg', player=chess.BLACK, book=True)
gpw_dir = chessplayer.GreedyPlayer('gp-dir-out.svg', player=chess.WHITE, book=True, directory=True)
gpb_dir = chessplayer.GreedyPlayer('gp-dir2-out.svg', player=chess.BLACK, book=True, directory=True)

mpw = chessplayer.MinimaxPlayer('mpw-out.svg', player=chess.WHITE)
mpb = chessplayer.MinimaxPlayer('mpb-out.svg', player=chess.BLACK)

mpw_book = chessplayer.MinimaxPlayer('mpw-book-out.svg', player=chess.WHITE, book=True)
mpb_book = chessplayer.MinimaxPlayer('mpb-book-out.svg', player=chess.BLACK, book=True)

mpw_dir = chessplayer.MinimaxPlayer('mpw-dir-out.svg', player=chess.WHITE, book=True, directory=True)
mpb_dir = chessplayer.MinimaxPlayer('mpb-dir-out.svg', player=chess.BLACK, book=True, directory=True)


"""
Do Testing
----------
"""

start_fen = chess.STARTING_FEN

mpw_test = chessplayer.MinimaxPlayer(   'mpw-test.svg', 
                                        fen         = start_fen,
                                        player      = chess.WHITE,
                                        book        = True,
                                        directory   = True )

mpb_test = chessplayer.MinimaxPlayer(   'mpb-test.svg', 
                                        fen         = start_fen,
                                        player      = chess.BLACK,
                                        book        = True,
                                        directory   = True )

gpb_test = chessplayer.GreedyPlayer(   'gp-test.svg', 
                                        fen         = start_fen,
                                        player      = chess.BLACK,
                                        book        = True,
                                        directory   = True )

#runners.PlayAgents(rp, gpb_test, debug=True)

hp = chessplayer.HumanPlayer('hp-out.svg')

#runners.PlayAgents(mpw_test, hp, debug=True)

runners.PlayAgents(hp, mpb_test, debug=True)

"""
Generate Statistics
--------------
"""

# stats = []

# # stats.append(['Greedy vs. Random'] + runners.calcStats((gpw, rp), (rp, gpb), 100))
# # stats.append(['Greedy with Book vs. Random'] + runners.calcStats((gpw_book, rp), (rp, gpb_book), 100))
# # stats.append(['Greedy with Book/TBs vs. Random'] + runners.calcStats((gpw_dir, rp), (rp, gpb_dir), 100))
# # stats.append(['Greedy with Book vs. Greedy'] + runners.calcStats((gpw_book, gpb), (gpw, gpb_book), 100))
# # stats.append(['Greedy with Book/TBs vs. Greedy'] + runners.calcStats((gpw_dir, gpb), (gpw, gpb_dir), 100))
# stats.append(['Minimax vs. Greedy'] + runners.calcStats((mpw, gpb), (gpw, mpb), 60, trunc=False, log=True))
# stats.append(['Minimax with Book/TBs vs. Greedy'] + runners.calcStats((mpw_dir, gpb), (gpw, mpb_dir), 60, trunc=False, log=True))

# longest_title = max([len(x[0]) for x in stats])
# header = ['Matchup', 'Total Games', 'Wins', 'Losses', 'Draws', 'Win Percentage', 'Loss Percentage']

# table = PrettyTable(header)
# for row in stats:
#     table.add_row(row)

# print table
