import matplotlib.pyplot as plt
import json
import time
from os import listdir
import pprint as pp

def plot(res, title='Winrate over time vs random player'):
	games = [i[0] for i in res]
	winrates = [i[1] for i in res]
	plt.plot(games, winrates)
	plt.ylabel('Winrate')
	plt.xlabel('Number of games')
	plt.ylim([0, 1])
	plt.title(title)
	plt.show()

def save(parameters, network_folder):
	# save results from games over time in /temp/
	filename = time.strftime("./"+network_folder+"/results_%Y-%m-%d_%H%M%S.json")
	with open(filename, 'w') as outfile:
		json.dump(parameters, outfile)

def load(file):
	# loads and returns results using json
	with open(file) as json_data:
		data = json.load(json_data)

	return data
	


def plot_last(dir = './temp/'):
	# scans the temp folder
	files = listdir(dir)

	if (len(files)>0):
		# selects most recent file
		files.sort()

		# loads the results from it
		data = load(dir+files[-1])
		results = data['results']
		del data['results']

		print('Loaded file: ' + files[-1])
		pp.pprint(data)
		# plots the results
		plot(results)
	else:
		raise ValueError('No files available in /temp/')