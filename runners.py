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
def PlayAgents(WhitePlayer, BlackPlayer, debug=False, truncate=False):
    board = WhitePlayer.board
    if debug:
        WhitePlayer.printGame(board)
        WhitePlayer.writeBoard('out.svg', board)
        #call(['open', 'out.svg'])
        print '\n'
        time.sleep(2)

    stop_cond = board.is_game_over(claim_draw=True)
    if truncate:
        stop_cond = (board.fullmove_number > 15 or board.is_game_over(claim_draw=True))

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
            WhitePlayer.printGame(board)
            WhitePlayer.writeBoard('out.svg', board)
            #call(['open', 'out.svg'])
            print '\n'
            time.sleep(2)

        stop_cond = board.is_game_over(claim_draw=True)
        if truncate:
            stop_cond = (board.fullmove_number > 20 or board.is_game_over(claim_draw=True))

    if not truncate:
        result = board.result
    elif WhitePlayer.getBoardValue(board) > 1:
        result = "1-0"
    elif BlackPlayer.getBoardValue(board) > 1:
        result = "0-1"
    else:
        result = "1/2-1/2"

    if debug:
        WhitePlayer.printGame(board)
        print result
        print chess.pgn.Game.from_board(board)

    for player in [WhitePlayer, BlackPlayer]:
        player.board.reset()
        player.half_moves = 0

    return result

# def playAgents(WhitePlayer, BlackPlayer, debug=False):
#     board = WhitePlayer.board
#     if debug:
#         print board
#         print '\n'

#     while not board.is_game_over(claim_draw=True):
#         WhitePlayer.writeBoard('out.svg', board)

#         if board.turn == chess.WHITE:
#             if debug:
#                 print "Move:", board.fullmove_number
#             move = WhitePlayer.move(board)
#             if debug:
#                 print "White's move: {}".format(move)
#             board.push(move)
#         else:
#             move = BlackPlayer.move(board)
#             if debug:
#                 print "Black's move: {}".format(move)
#             board.push(move)

#         if debug:
#             print board
#             print '\n'
#             time.sleep(1)

#     if debug:
#         print board
#         print board.result()
#         print chess.pgn.Game.from_board(board)

    

#     return board.result()

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
        logfile = open('stats.log', 'a')
        logfile.write("Running Tests: {} vs {}", white1, black1)
    wins = 0
    draws = 0
    losses = 0
    total = 0
    for _ in xrange(games_per_side):
        # white1 vs. black1
        result = chessplayer.PlayAgents(white1, black1, truncate=trunc)

        if result == '1-0':
            wins += 1
        elif result == '1/2-1/2':
            draws += 1
        else:
            losses += 1
        total += 1

        if log:
            logfile.write(str([total, wins, losses, draws, float(wins) / total]))

        # white2 vs. black2
        result = chessplayer.PlayAgents(white2, black2, truncate=trunc)

        if result == '1-0':
            wins += 1
        elif result == '1/2-1/2':
            draws += 1
        else:
            losses += 1
        total += 1

        if log:
            logfile.write([total, wins, losses, draws, float(wins) / total])

    if log:
        logfile.write("Done with {} vs. {}", white1, black1)
        logfile.close()

    return [total, wins, losses, draws, float(wins) / total]

