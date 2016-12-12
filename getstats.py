
import chess
from prettytable import PrettyTable

import chessplayer
import runners

"""
Create Chessplayer Instances
----------------------------
"""

rp = chessplayer.RandomPlayer()

gpw = chessplayer.GreedyPlayer()
gpb = chessplayer.GreedyPlayer(player=chess.BLACK)
gpw_book = chessplayer.GreedyPlayer(book=True)
gpb_book = chessplayer.GreedyPlayer(player=chess.BLACK, book=True)
gpw_dir = chessplayer.GreedyPlayer(player=chess.WHITE, book=True, directory=True)
gpb_dir = chessplayer.GreedyPlayer(player=chess.BLACK, book=True, directory=True)

mpw = chessplayer.MinimaxPlayer(player=chess.WHITE)
mpb = chessplayer.MinimaxPlayer(player=chess.BLACK)

mpw_book = chessplayer.MinimaxPlayer(player=chess.WHITE, book=True)
mpb_book = chessplayer.MinimaxPlayer(player=chess.BLACK, book=True)

mpw_dir = chessplayer.MinimaxPlayer(player=chess.WHITE, book=True, directory=True)
mpb_dir = chessplayer.MinimaxPlayer(player=chess.BLACK, book=True, directory=True)

# only one NN player instance allowed at a time
nnw = chessplayer.GreedyNNPlayer(player=chess.WHITE, book=True, directory=True)
#nnb = chessplayer.GreedyNNPlayer(player=chess.BLACK, book=True, directory=True)

"""
Generate Statistics
--------------
"""

# stats = []

# # Comment out lines as appropriate to generate statistics on certain matchups.
# # Also, adjust number of games as needed.
# # NOTE: MAY TAKE HOURS TO RUN, DEPENDING ON PLAYERS AND NUMBER OF GAMES

# stats.append(['Greedy vs. Random'] + runners.calcStats((gpw, rp), (rp, gpb), 200))
# stats.append(['Greedy with Book vs. Random'] + runners.calcStats((gpw_book, rp), (rp, gpb_book), 200))
# stats.append(['Greedy with Book/TBs vs. Random'] + runners.calcStats((gpw_dir, rp), (rp, gpb_dir), 200))
# stats.append(['Greedy with Book vs. Greedy'] + runners.calcStats((gpw_book, gpb), (gpw, gpb_book), 100))
# stats.append(['Greedy with Book/TBs vs. Greedy'] + runners.calcStats((gpw_dir, gpb), (gpw, gpb_dir), 100))
# stats.append(['Minimax vs. Greedy'] + runners.calcStats((mpw, gpb), (gpw, mpb), 150, trunc=False, log=True))
# stats.append(['Minimax with Book/TBs vs. Greedy'] + runners.calcStats((mpw_dir, gpb), (gpw, mpb_dir), 150, trunc=False, log=True))
# stats.append(['ANN vs. Greedy'] + runners.calcStats((nnw, gpb), (nnw, gpb), 75, trunc=False))

# longest_title = max([len(x[0]) for x in stats])
# header = ['Matchup', 'Total Games', 'Wins', 'Losses', 'Draws', 'Win Percentage', 'Loss Percentage']

# table = PrettyTable(header)
# for row in stats:
#     table.add_row(row)

# print table
