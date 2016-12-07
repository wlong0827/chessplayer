import chess
from prettytable import PrettyTable

import chessplayer
import runners

"""
Create Chessplayer Instances
----------------------------
"""

rp = chessplayer.RandomPlayer('rp-out.svg')

gp = chessplayer.GreedyPlayer('gp-out.svg')
gp_book = chessplayer.GreedyPlayer('gp-book-out.svg', book=True)
gp_dir = chessplayer.GreedyPlayer('gp-dir-out.svg', book=True, directory=True)
gp_dir2 = chessplayer.GreedyPlayer('gp-dir2-out.svg', book=True, directory=True)

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
runners.PlayAgents(gp, gp_dir2, debug=True)


"""
Run Statistics
--------------
"""

# stats = []

# stats.append(['Greedy vs. Random'] + runners.calcStats((gp, rp), (rp, gp), 100))
# stats.append(['Greedy with Book vs. Random'] + calcStats((gp_book, rp), (rp, gp_book), 100))
# stats.append(['Greedy with Book/TBs vs. Random'] + calcStats((gp_dir, rp), (rp, gp_dir), 100))
# stats.append(['Minimax vs. Greedy'] + calcStats((mpw, gp), (gp, mpb), 5))
# stats.append(['Minimax with Book vs. Greedy'] + calcStats((mpw_book, gp), (gp, mpb_book), 5))
# stats.append(['Minimax with Book/TBs vs. Greedy'] + calcStats((mpw_dir, gp), (gp, mpb_dir), 5))

# longest_title = max([len(x[0]) for x in stats])
# header = ['Matchup', 'Total Games', 'Wins', 'Losses', 'Draws', 'Win Percentage']

# table = PrettyTable(header)
# for row in stats:
#     table.add_row(row)

# print table

