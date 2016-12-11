import chessplayer
import chess
import time
from subprocess import call

"""
    playAgents:
    Used to play two players against each other.

    If debug is set to True, the game plays one move every 2 seconds,
    prints the board after each move, and reports the result.
"""
def PlayAgents(WhitePlayer, BlackPlayer, outfile="", debug=False, truncate=False):
    board = WhitePlayer.board

    # for input FENs
    if board.turn == chess.BLACK:
        WhitePlayer.half_moves = 1
        BlackPlayer.half_moves = 0

    if debug:
        if outfile != "":
            WhitePlayer.writeBoard(outfile, board)
            call(['open', outfile])
        WhitePlayer.printGame(board)
        print '\n'
        time.sleep(2)

    stop_cond = board.is_game_over(claim_draw=True)
    if truncate:
        stop_cond = (board.fullmove_number > 20 or stop_cond)

    while not stop_cond:

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
            if outfile != "":
                WhitePlayer.writeBoard(outfile, board)
                call(['open', outfile])
            WhitePlayer.printGame(board)
            print '\n'
            time.sleep(2)

        stop_cond = board.is_game_over(claim_draw=True)
        if truncate:
            stop_cond = (board.fullmove_number > 20 or stop_cond)

    if not truncate:
        result = board.result(claim_draw=True)
    elif WhitePlayer.getBoardValue(board) > 1:
        result = "1-0"
    elif WhitePlayer.getBoardValue(board) < 1:
        result = "0-1"
    else:
        result = "1/2-1/2"

    if debug:
        print result
        print chess.pgn.Game.from_board(board)

    for player in [WhitePlayer, BlackPlayer]:
        player.board.reset()
    
    WhitePlayer.half_moves = 0
    BlackPlayer.half_moves = 1

    return result

"""
    calcStats:
    Used to generate statistics about the matchup of two players.

    white1 is assumed to be the same type of player as black2.
    white2 is assumed to be the same type of player as black1.

    Plays 2*games_per_side games in total and reports statistics
    on those games. Win percentage is calculated from the cumulative
    wins of white1 and black2 (assumed to be the same).
"""
def calcStats((white1, black1), (white2, black2),
                games_per_side, log=False, trunc=False):
    if log:
        logfile = open('/Users/jwbaskerv/Desktop/ResultsLog.txt', 'a')
        logfile.write("Running Tests: {} vs {}\n".format(white1, black1))
    wins = 0
    draws = 0
    losses = 0
    total = 0
    for _ in xrange(games_per_side):
        # white1 vs. black1
        result = PlayAgents(white1, black1, truncate=trunc)

        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1
        total += 1

        if log:
            logfile.write(str([total, wins, losses, draws, float(wins) / total, float(losses) / total])+ '\n')

        # white2 vs. black2
        result = PlayAgents(white2, black2, truncate=trunc)

        if result == '1-0':
            losses += 1
        elif result == '0-1':
            wins += 1
        else:
            draws += 1
        total += 1

        if log:
            logfile.write(str([total, wins, losses, draws, float(wins) / total, float(losses) / total]) + '\n')

    if log:
        logfile.write("Done with {} vs. {}\n".format(white1, black1))
        logfile.close()

    return [total, wins, losses, draws, float(wins) / total, float(losses) / total]

