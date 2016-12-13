# -*- encoding: utf-8 -*-
import getopt
import sys
import chess
import chessplayer
import runners

"""
Play a Game!
    Usage: python main.py -w <white/book/tbs> -b <black/book/tbs> -s <svgfile> -f <fen>

    Example: python main.py -w hp -b mp/F/F -s game.svg -f '8/8/8/4p3/8/4k3/8/4K3 b - - 0 1'

        Option      Description                     Default
        ---------   -----------------------------   ---------------------
        <white>     the player type of White        mp
        <black>     the player type of Black        gp
        <book>      opening book usage (T/F)        T
        <tbs>       endgame tablebase usage (T/F)   T
        <svgfile>   outfile for svgs                "" (no svg)
        <fen>       FEN for custom board position   normal starting board

    Player types: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)

    Notes: 
    1.  If you don't use the <svgfile> option, the game will just be printed
        out to the console.
    2.  For the Human player, you can enter moves in Standard Algebraic Notation
        (e.g. e4, Bxf3, or O-O-O) or UCI Notation (e.g. e2e4, g4f3, e1c1).
    3.  You can't play ANN vs. ANN because you can only restore one
        TensorFlow session from the data at a time.
"""
def play(argv):

    start_fen = chess.STARTING_FEN
    svgfile = ""

    white_type = 'mp'
    white_book = True
    white_dir = True

    black_type = 'gp'
    black_book = True
    black_dir = True


    # parse the options and complain on error
    try:
        opts, args = getopt.getopt(argv, "hw:b:s:f:", ["white=", "black=", "fen="])
    except getopt.GetoptError:
        print 'Usage: python main.py -w <white/book/dir> -b <black, book, dir> -s <svgfile> -f <fen>'
        print '   Players: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)'
        sys.exit(2)

    # set variables according to options, if necessary
    for opt, arg in opts:
        if opt == '-h':
            print 'python python main.py -w <white/book/dir> -b <black/book/dir> -s <svgfile> -f <fen>'
            print '   Players: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)'
            sys.exit()
        elif opt in ("-w", "--white"):
            args = arg.split('/')
            if args[0] not in ("hp", "rp", "gp", "mp", "nn"):
                print 'White player type not recognized. Use the shortcode (e.g. "hp").'
                print '   Players: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)'
                sys.exit(2)
            else:
                white_type = args[0]
                if len(args) == 3:
                    white_book = False if (args[1] in ('F', 'False', 'f', 'false')) else True
                    white_dir = False if (args[2] in ('F', 'False', 'f', 'false')) else True
        elif opt in ("-b", "--black"):
            args = arg.split('/')
            if args[0] not in ("hp", "rp", "gp", "mp", "nn"):
                print 'Black player type not recognized. Use the shortcode (e.g. "hp").'
                print '   Players: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)'
                sys.exit(2)
            else:
                black_type = args[0]
                if len(args) == 3:
                    black_book = False if (args[1] in ('F', 'False', 'f', 'false')) else True
                    black_dir = False if (args[2] in ('F', 'False', 'f', 'false')) else True
        elif opt in ("-s", "--svgfile"):
            svgfile = arg
        elif opt in ("-f", "--fen"):
            start_fen = arg

    print start_fen

    # create instances of chess agents and play them
    print ""
    if white_type == 'hp':
        print "White Player: Human"
        white = chessplayer.HumanPlayer(fen=start_fen)
    elif white_type == 'rp':
        print "White Player: Random"
        white = chessplayer.RandomPlayer(fen=start_fen)
    elif white_type == 'gp':
        print "White Player: Greedy"
        white = chessplayer.GreedyPlayer(player=chess.WHITE, fen=start_fen, book=white_book, directory=white_dir)
    elif white_type == 'mp':
        print "White Player: Minimax"
        white = chessplayer.MinimaxPlayer(player=chess.WHITE, fen=start_fen, book=white_book, directory=white_dir)
    elif white_type == 'nn':
        print "White Player: Greedy ANN"
        white = chessplayer.GreedyNNPlayer(player=chess.WHITE, fen=start_fen, book=white_book, directory=white_dir)

    if black_type == 'hp':
        print "Black Player: Human"
        black = chessplayer.HumanPlayer(fen=start_fen)
    elif black_type == 'rp':
        print "Black Player: Random"
        black = chessplayer.RandomPlayer(fen=start_fen)
    elif black_type == 'gp':
        print "Black Player: Greedy"
        black = chessplayer.GreedyPlayer(player=chess.BLACK, fen=start_fen, book=black_book, directory=black_dir)
    elif black_type == 'mp':
        print "Black Player: Minimax"
        black = chessplayer.MinimaxPlayer(player=chess.BLACK, fen=start_fen, book=black_book, directory=black_dir)
    elif black_type == 'nn':
        print "Black Player: Greedy ANN"
        black = chessplayer.GreedyNNPlayer(player=chess.BLACK, fen=start_fen, book=black_book, directory=black_dir)
    print ""

    runners.playAgents(white, black, outfile=svgfile, debug=True)

if __name__ == "__main__":
   play(sys.argv[1:])