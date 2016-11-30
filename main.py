import chessplayer
import chess
from prettytable import PrettyTable

def calcStats((white1, black1), (white2, black2), games_per_side):
    wins = 0
    draws = 0
    losses = 0
    total = 0
    for _ in xrange(games_per_side):
        # white1 vs. black1
        result = chessplayer.PlayAgents(white1, black1)
        #print result
        if result == '1-0':
            wins += 1
        elif result == '1/2-1/2':
            draws += 1
        else:
            losses += 1
        total += 1

        for agent in [white1, black1]:
            agent.board.reset()

        # white2 vs. black2
        result = chessplayer.PlayAgents(white2, black2)
        #print result
        if result == '1-0':
            wins += 1
        elif result == '1/2-1/2':
            draws += 1
        else:
            losses += 1
        total += 1

        for agent in [white2, black2]:
            agent.board.reset()

        #print "2 games"
    return [total, wins, losses, draws, float(wins) / total]

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

#chessplayer.PlayAgents(gp_dir, gp_dir2, debug=True)
#chessplayer.PlayAgents(mpw_dir, mpb_dir, debug=True)

stats = []

stats.append(['Greedy vs. Random'] + calcStats((gp, rp), (rp, gp), 100))
stats.append(['Greedy with Book vs. Random'] + calcStats((gp_book, rp), (rp, gp_book), 100))
stats.append(['Greedy with Book/TBs vs. Random'] + calcStats((gp_dir, rp), (rp, gp_dir), 100))
#stats.append(['Minimax vs. Greedy'] + calcStats((mpw, gp), (gp, mpb), 5))
#stats.append(['Minimax with Book vs. Greedy'] + calcStats((mpw_book, gp), (gp, mpb_book), 5))
#stats.append(['Minimax with Book/TBs vs. Greedy'] + calcStats((mpw_dir, gp), (gp, mpb_dir), 5))

longest_title = max([len(x[0]) for x in stats])
header = ['Matchup', 'Total Games', 'Wins', 'Losses', 'Draws', 'Win Percentage']

table = PrettyTable(header)
for row in stats:
    table.add_row(row)

print table

"""
-------------- Test Code -------------------------
We should probably put this in a new file, eventually.
Sorry for the merge conflicts here!
"""
# rp = RandomPlayer('out.svg')
#gp = GreedyPlayer('out.svg')
#hp = HumanPlayer('out.svg')
#mp = MinimaxPlayer('out1.svg', player=chess.BLACK, book=False)
# mp2 = MinimaxPlayer('out2.svg', player=chess.WHITE, book="Formula12")
#mp2 = MinimaxPlayer('out2.svg', player=chess.BLACK, book="Formula12")
#mp = MinimaxPlayer('out.svg', player=chess.BLACK, book="gm2600")

#PlayAgents(mp2, hp)
#PlayAgents(mp, mp2)

# mp.move()




