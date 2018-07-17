import matplotlib.pyplot as plt
import json
import time
from os.path import isdir
import glob
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
	# check if file was given instead of folder
	if '.' in network_folder:
		network_folder = '/'.join(network_folder.split('/')[:-1]) or '.'
	if not isdir(network_folder):
		print('Folder does not exist: ' + network_folder)
	else:
		filename = time.strftime("./"+network_folder+"/results_%Y-%m-%d_%H%M%S.json")
		with open(filename, 'w') as outfile:
			json.dump(parameters, outfile)

def load(file):
	# loads and returns results using json
	with open(file) as json_data:
		data = json.load(json_data)

	return data
	


def plot_last(directory = './networks/'):
	# replace /*/ with /**/ to make recursive
	paths = glob.glob('networks/*/*.json', recursive=True)
	
	if (len(paths)>0):
		files = [(i.split('\\')[-1], i) for i in paths]
		# sort on filename
		sorted(files, key = lambda x:x[0])
		
		# loads the results from it
		data = load(files[-1][1])
		results = data['results']
		del data['results']

		print('Loaded file: ' + files[-1][1])
		pp.pprint(data)
		# plots the results
		plot(results)
	else:
		raise ValueError('No result files available')