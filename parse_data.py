import chess

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
			string = str(position)
			f3.write(string[1:-1] + "\n")

f3.close()
