import chess

values = {'P': float(0.1), 'R': float(0.2), 'N': float(0.3), 'B': float(0.4), 'Q': float(0.5), 'K': float(0.6),
'p': float(-0.1), 'r': float(-0.2), 'n': float(-0.3), 'b': float(-0.4), 'q': float(-0.5), 'k': float(-0.6)}

def encode(board):

	result = []
	#print "------Start--------"
	#print board
	for square in range(64):
		piece = board.piece_at(square)
		#print str(piece)
		if str(piece) in values.keys():
			v = float(values[str(piece)])
			#print "piece append", v
			result.append(v)
		else:
			#print "append 0"
			result.append(0)
	
	result.append(int(board.turn))
	#print result
	#print "----------End---------"
	return result

f = open('games.txt', "r")
f2 = open('scores.csv', "r")
f3 = open('train.txt', "w+")

file = f.readlines()
file2 = f2.readlines()

games = []
scores = []

for line in file:
	if line[0] !=  '\n':
		games.append(line)

for line in file2:
	if line[0] !=  '\n':
		l = line.split(',')
		scores.append(l[1])

scores = scores[1:]

assert(len(scores) == len(games))
for i in range(len(scores)):
	b = chess.Board()
	positions = []

	s = scores[i]
	g = games[i]
	s = s.split(" ")
	g = g.split(" ")
	g.pop()
	s[len(s)-1] = s[len(s)-1].rstrip()

	assert(len(s) == len(g))
	for j in range(len(s)):
		b.push(chess.Move.from_uci(g[j]))
		positions.append((int(s[j]), b.copy()))

	for position in positions:
		if not position[0] == 0:
			result = encode(position[1])
			string = str(position[0]) + "," + str(result) + "\n"
			f3.write(string)

f3.close()



