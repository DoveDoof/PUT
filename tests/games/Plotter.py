import matplotlib.pyplot as plt
import numpy as np


d = {
	"5000": [-1, 1, 0, 1, 0, 0, 1, -1, 0, -1, 1, 0, 1, 1, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 1, 1, 0, -1, -1, 0, 0, -1, 0, -1, 1, 1, 0, 1, -1, 0, 0, 1, -1, -1, -1, 0, 1, -1, -1, -1, -1, -1, 1, -1, 0, 0, -1, -1, -1, -1, 1, 0, 1, 0, 0, 0, -1, 1, 0, 1, -1, -1, 0, -1, 1, -1, 0, 0, 0, -1, -1, 0, 1, -1, 0, 1, 0, 1, -1, -1, 1, 0, -1, 0, 1, 1, 1, 0, -1],
	"10000": [0, 1, 1, 1, -1, 0, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 0, 0, -1, -1, 0, 1, 0, 1, 0, 0, 1, -1, -1, 0, -1, 1, 0, 1, 0, -1, -1, -1, -1, -1, -1, 0, -1, 0, 1, 1, 1, 0, 1, -1, 1, -1, 1, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 1, 0, 1, 1, -1, -1, 0, -1, 0, 1, 0, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1, -1, -1, 1, -1, 1, 0, -1, -1, -1, 0],
	"15000": [1, 1, 0, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 0, 1, 1, 0, -1, 1, -1, 1, 1, -1, -1, 1, 0, -1, 1, -1, 1, 1, 1, 0, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 0, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 0, 1, 0, -1, -1, 1, 1, 1, 0, -1, 1, 1, -1, 0, 0, 0, 0, -1, 0, 1, -1, 0, 1, 0, 1, 1, -1, 1, 0],
	"20000": [0, 1, 1, 0, 1, 1, 0, -1, 1, 0, 1, 0, -1, -1, -1, 0, 1, 0, 0, -1, 0, 0, 1, -1, 0, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1, 1, 0, 1, 0, 0, -1, 0, 0, -1, 1, 0, 1, 1, 1, -1, 1, 0, 1, 1, 1, 1, 1, 0, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 0, 0, 1, 1, -1, 0, -1, -1, -1, 1, 1, -1, 0, -1, 1, 0, 0, -1, 0, 1, -1, 1],
	"25000": [1, 0, 1, 1, 1, 0, 0, 0, 0, -1, -1, 0, -1, 0, 1, 0, 1, -1, -1, 1, 0, -1, 1, 1, 1, 0, -1, -1, -1, 1, 1, -1, 0, -1, -1, 0, 0, 1, 1, 1, 1, -1, 0, -1, 0, 0, -1, 1, -1, -1, 1, 0, -1, 1, 1, 0, 1, 1, -1, -1, 1, 1, 0, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, -1, 1, 0, -1, 1, 1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, -1],
	"30000": [-1, 1, 1, 1, 0, 1, -1, 1, -1, -1, -1, 0, 0, 1, 1, -1, 0, 1, 1, 1, -1, 0, -1, 0, 0, 1, 1, 0, -1, -1, -1, 1, -1, 1, 0, 1, 0, 1, 1, -1, -1, 1, 1, 0, -1, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 1, 1, 1, -1, -1, 1, -1, 0, 1, -1, -1, 0, 1, -1, 0, -1, 1, 1, -1, 0, 0, 1, -1, 0, -1, -1, 1, 1, 1, 1, 1, -1, 1, 0, 1, -1, 0, 0, 0, -1, 1, -1, -1, -1, 1],
	"35000": [1, 1, 1, 1, -1, 0, 1, 1, 1, 1, -1, 1, 0, -1, -1, 1, -1, 0, 1, 1, 1, 1, -1, -1, -1, 0, -1, -1, -1, -1, 0, 1, 1, 0, 1, -1, -1, 0, -1, -1, 1, 1, 1, -1, 0, 1, -1, 0, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 0, -1, 1, 1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, 1, 0, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 0, -1],
	"40000": [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, -1, 0, -1, 0, 1, 1, 1, 0, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1, 1, -1, 0, 0, 1, -1, 1, -1, 1, 1, 0, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 0, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 0, 1, 0, 1, 0, 1],
	"45000": [-1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 0, -1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 0, 1, 1, 1, -1, -1, 1, 1, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, -1, 1, -1, 0, 1, -1, 1, 0, 1, 1, -1, 0, 0, -1, -1, 1, 0, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 0, 0, 1, 1],
	"50000": [0, 1, 1, -1, 0, 0, 1, 1, 1, 1, -1, 0, -1, 1, -1, -1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, -1, 0, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, 1, 0, 1, 0, -1, 1, 0, 0, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 0, -1, 0],
	"55000": [1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 0, 1, 1, 1, -1, -1, 0, -1, -1, 1, 1, 1, -1, -1, 0, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 0, 1, 0, 1, 1, -1, -1, 1, 1, 0, 0, 1, 0, 1, 1],
	"60000": [1, 0, 1, -1, 1, 1, 0, 0, 0, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 0, 0, -1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, -1, -1, 0, -1, -1, 0, 1, -1, -1, 1, -1, 1, 0, 1, 1, 1, 1, 0, 1, -1, -1, 1, -1, 0, 1, 0, 1, 0, 1, 1, 1, -1, 1, -1],
	"65000": [-1, -1, -1, 1, 1, 0, 1, 1, -1, 1, 1, -1, 0, 0, 1, -1, -1, 1, 1, 1, 1, -1, 0, -1, -1, 0, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 0, 1, 1, 0, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 0, 1, 1, 0, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1],
	"70000": [0, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 0, -1, 0, 0, 1, -1, 1, -1, 1, -1, 0, 1, 0, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 0, 1, 1, 1, -1, -1, 0, -1, -1, 1, -1, -1, 0, 0, -1, 1, 0, 0, 0, -1, 1, 1, -1, 0, 0, -1, -1, 0, 1, 1, -1, 1, 0, 1, 0, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, 0, -1, -1, 0, 0, 0],
	"75000": [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 0, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 1, 1, -1, 1, 0, -1, 0, 1, -1, 1, 1, 1, 1, -1, -1, 0, -1, 0, 0, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 0, 0, -1, -1, 1, 1, 1, 1, -1, -1, 0, 1, -1, -1, -1, 1, 1, 0, -1, 1, -1, -1, -1, 1, -1, 0, -1, 0, -1, -1, 1, 1, 1, 1, 0, 1, 1, 1, 1, -1],
	"80000": [-1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 0, 1, 0, 0, -1, 1, 1, 1, 0, 1, -1, 0, 1, -1, 1, 1, -1, -1, -1, 0, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 0, -1, -1, 1, 1, 1, 0, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 0, 1, 1, 1, -1, 1, 0, -1, 1, -1, -1, 1, 1, 0, -1, 1, -1, 1, 1, -1, 1, 0, 1, 1, 1, -1, 1, -1],
	"85000": [1, 0, -1, -1, 0, 0, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 0, 0, -1, 1, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, 0, -1, 1, 0, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 0, 1, 1, -1, -1, 1, -1, 0, -1, 1, -1, -1, 1, 1, -1, 0, 1, 1, 1, -1, -1, 1, 0, -1, -1],
	"90000": [1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 0, 1, 0, 0, 1, -1, 1, 1, 0, 1, -1, -1, 1, 1, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, -1, 0, -1, -1, 1, -1, 1, 1, 0, 1, -1, -1, 1, 0, -1, 1, -1, 0, -1, 1, 0, 0, -1, 0, -1, 1, -1, 1, 0, 1, 1, -1, 1, -1, 0, 1, -1, 0, -1, -1, -1, -1, 0, 1, 0, 1, 1, 0, 1, -1, 1, 1, 0, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1],
	"95000": [1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 0, 1, 0, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 0, -1, 0, 0, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 0, 1, 1, -1, -1, -1, 1, 0, 1, 0, -1, -1, 0, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 0, 0, 1, 0, -1, 0, 1, -1, 1, -1, 0, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1],
	"100000": [-1, 0, 1, 1, 1, 1, -1, 1, -1, -1, 1, 0, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 0, 1, 1, -1, -1, 0, -1, 1, 0, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, 0, -1, 1, -1, 0, 1, 0, 0, -1, 0, 1, 1, -1, -1, 0, -1, -1, -1, -1, 1, 0, 1, 1, 1, 1, -1, 1, -1, 1, 0, -1, -1, 0, -1, -1, 0, -1, 0, -1, -1, 0, -1, 1, 1, 1, 1, 0, 1, -1, 1, -1, -1, 1, 1, -1]
}

winrate = []
lossrate = []
drawrate = []
games = []

for key,value in d.items():
    winrate.append(value.count(1)/len(value))
    lossrate.append(value.count(-1) / len(value))
    drawrate.append(value.count(0) / len(value))
    games.append(key)

fig, ax = plt.subplots()
plt.ticklabel_format( style='sci', axis='x', scilimits=(0,0))
plt.plot(np.asfarray(games), np.array(winrate))
plt.legend(['win rate','loss rate','draw rate'])

plt.title("Benchmark")
plt.show()
