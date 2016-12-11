# chessplayer
## CS 182 Final Project

### Requirements
Run `pip install -r requirements.txt` to install all code requirements.

As an alternative to a web broswer, you may install Gaplin (http://gapplin.wolfrosch.com/) to open svgs.

### Playing a Game
Usage: python main.py -w \<white> -b \<black> -s \<svgfile> -f \<fen>

    Option      Description                     Default
    ---------   -----------------------------   ---------------------
    <white>     the player type of White        mp
    <black>     the player type of Black        gp
    <svgfile>   outfile for svgs                "" (no svg)
    <fen>       FEN for custom board position   normal starting board

Player types: Human (hp), Random (rp), Greedy (gp), Minimax (mp), ANN (nn)

Example: python main.py -w hp -b mp -s game.svg -f '8/8/8/4p3/8/4k3/8/4K3 b - - 0 1'

Notes: 
1.  If you don't use the \<svgfile> option, the game will just be printed
    out to the console.
2.  For the Human player, you can enter moves in Standard Algebraic Notation
    (e.g. e4, Bxf3, or O-O-O) or UCI Notation (e.g. e2e4, g4f3, e1c1).
3.  You can't play ANN vs. ANN because you can only restore one
    TensorFlow session from the data at a time.