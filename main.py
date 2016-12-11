# -*- encoding: utf-8 -*-
import getopt
import sys
import chess
import chessplayer
import runners

"""
Play a Game!
    Note: You can't play ANN vs. ANN because you can only restore one
    TensorFlow session from the data at a time.
----------
"""
def play(argv):

    start_fen = chess.STARTING_FEN
    svgfile = ""

    white_type = 'mp'
    black_type = 'gp'

    try:
        opts, args = getopt.getopt(argv, "hw:b:s:f:", ["white=", "black=", "fen="])
    except getopt.GetoptError:
        print 'Usage: main.py -w <whiteplayer> -b <blackplayer> -s <svgfile> -f <FEN>]'
        print '   Players: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -w <whiteplayer> -b <blackplayer> -s <svgfile> -f <FEN>]'
            print '   Players: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)'
            sys.exit()
        elif opt in ("-w", "--white"):
            if arg not in ("hp", "rp", "gp", "mp", "nn"):
                print 'White player type not recognized. Use the shortcode (e.g. "hp").'
                print '   Players: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)'
                sys.exit(2)
            else:
                white_type = arg
        elif opt in ("-b", "--black"):
            if arg not in ("hp", "rp", "gp", "mp", "nn"):
                print 'Black player type not recognized. Use the shortcode (e.g. "hp").'
                print '   Players: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)'
                sys.exit(2)
            else:
                black_type = arg
        elif opt in ("-s", "--svgfile"):
            svgfile = arg
        elif opt in ("-f", "--fen"):
            start_fen = arg

    print start_fen

    print ""
    if white_type == 'hp':
        print "White Player: Human"
        white = chessplayer.HumanPlayer(fen=start_fen)
    elif white_type == 'rp':
        print "White Player: Random"
        white = chessplayer.RandomPlayer(fen=start_fen)
    elif white_type == 'gp':
        print "White Player: Greedy"
        white = chessplayer.GreedyPlayer(player=chess.WHITE, fen=start_fen, book=True, directory=True)
    elif white_type == 'mp':
        print "White Player: Minimax"
        white = chessplayer.MinimaxPlayer(player=chess.WHITE, fen=start_fen, book=True, directory=True)
    elif white_type == 'nn':
        print "White Player: Greedy ANN"
        white = chessplayer.GreedyNNPlayer(player=chess.WHITE, fen=start_fen, book=True, directory=True)

    if black_type == 'hp':
        print "Black Player: Human"
        black = chessplayer.HumanPlayer(fen=start_fen)
    elif black_type == 'rp':
        print "Black Player: Random"
        black = chessplayer.RandomPlayer(fen=start_fen)
    elif black_type == 'gp':
        print "Black Player: Greedy"
        black = chessplayer.GreedyPlayer(player=chess.BLACK, fen=start_fen, book=True, directory=True)
    elif black_type == 'mp':
        print "Black Player: Minimax"
        black = chessplayer.MinimaxPlayer(player=chess.BLACK, fen=start_fen, book=True, directory=True)
    elif black_type == 'nn':
        print "Black Player: Greedy ANN"
        black = chessplayer.GreedyNNPlayer(player=chess.BLACK, fen=start_fen, book=True, directory=True)
    print ""

    runners.PlayAgents(white, black, outfile=svgfile, debug=True)

if __name__ == "__main__":
   play(sys.argv[1:])