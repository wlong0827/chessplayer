import chess

def onehot(i):
	result = [0 for _ in range(13)]
	result[i] = 1
	return result

values = {'P': onehot(1), 'R': onehot(2), 'N': onehot(3), 'B': onehot(4), 'Q': onehot(5), 'K': onehot(6),
'p': onehot(7), 'r': onehot(8), 'n': onehot(9), 'b': onehot(10), 'q': onehot(11), 'k': onehot(12)}
	
def encode(board):
	result = []
	for square in range(64):
		piece = board.piece_at(square)
		if str(piece) in values.keys():
			v = values[str(piece)]
			result.append(v)
		else:
			result.append(onehot(0))
	
	result.append([int(board.turn)])

	result = [item for sublist in result for item in sublist]
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
count = 0

assert(len(scores) == len(games))

for i in range(len(scores)):
	print "Progress", i
	b = chess.Board()
	positions = []

	s = scores[i]
	g = games[i]
	s = s.split(" ")
	g = g.split(" ")
	g.pop()
	s[len(s)-1] = s[len(s)-1].rstrip()

	messed_up = 0
	if len(s) == len(g):
		for j in range(len(s)):
			try:
				b.push(chess.Move.from_uci(g[j]))
				positions.append((int(s[j]), b.copy()))
			except:
				messed_up += 1

		for position in positions:
			if not position[0] == 0:
				result = encode(position[1])
				string = str(position[0]) + str(result) + "\n"
				count += 1
				f3.write(string)
	else:
		messed_up += 1

print "%i messed up" % messed_up
print "%i worked out" % count

f.close()
f2.close()
f3.close()



