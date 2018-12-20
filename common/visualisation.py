import matplotlib.pyplot as plt
import numpy as np
import json
import time
from os.path import isdir
import re
import glob
import pprint as pp

def plot(res, title='Winrate over time vs random player'):
	fontsize = 22

	games = [i[0] for i in res]
	winrates = [i[1] for i in res]
	fig, ax = plt.subplots()
	plt.rcParams.update({'font.size': fontsize})
	plt.ticklabel_format( style='sci', axis='x', scilimits=(0,0))
	plt.plot(np.asfarray(games), winrates)
	plt.ylabel('Winrate', fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlabel('Number of games', fontsize=fontsize)
	plt.xticks(fontsize=fontsize)
	ax.xaxis.offsetText.set_fontsize(fontsize)
	plt.minorticks_on()
	ax.grid(which='both')
	plt.ylim([0, 1])
	plt.xlim(xmin = 0)
	plt.title(title)
	plt.show()

def save(data):
	# data: type dictionary, should contain save_network_file_path and results

	network_folder = data['save_network_file_path']
	# if file was given (containing extention) then remove the filename from the path
	if '.' in network_folder:
		network_folder = '/'.join(network_folder.split('/')[:-1]) or '.'
	if not isdir(network_folder):
		print('Folder does not exist: ' + network_folder)
	else:
		# how many games were played in total
		nr_games = str(data["results"][-1][0])
		filename = time.strftime("./"+network_folder+"/_results_%Y-%m-%d_%H%M%S_"+nr_games+".json")
		with open(filename, 'w') as outfile:
			json.dump(data, outfile)

def load(file):
	# loads and returns results using json
	with open(file) as json_data:
		data = json.load(json_data)

	return data
	
def load_results(file, results_only = True):
	# selects most recent resultfile from folder of given file
	# returns: total number of games played, history of winrates
	paths = glob.glob('\\'.join(file.split('/')[:-1] + ['*.json']))
	paths.sort()
	data = load(paths[-1])
	if results_only:
		return data["results"][-1][0], data["results"]
	else:
		return data

def plot_last(directory = './networks/'):
	# replace /*/ with /**/ to make recursive
	paths = glob.glob(r'C:\Users\User\APH\1B 2017 2018\Advanced Machine Learning\Resit\Git\QLUT\networks\*\*.json', recursive=True)
	
	if (len(paths)>0):
		files = [(i.split('\\')[-1], i) for i in paths]
		# sort on filename, not path
		files = sorted(files, key = lambda x:x[0])
		files = files[::-1]
		filelist = []
		for file in files:
			p = re.compile('_results_\d{4}-\d{2}-\d{2}_\d{6}_\d+\.json')
			if p.search(file[0]) is not None:
				filelist.append(file)


		for i,file in enumerate(filelist):
			print(i, file[1])
		choice = input("Choose number of network you want to load: ")

		try:
			choice = int(choice)
			chosen_file = filelist[choice]
		except ValueError:
			print("ERROR: Please input integer")
			input("Press enter to exit")
		except IndexError:
			print("ERROR: Please choose integer from list")
			input("Press enter to exit")	

		print(choice)
		print(chosen_file)
		# loads the results from it
		data = load(chosen_file[1])
		results = data['results']
		del data['results']

		print('Loaded file: ' + chosen_file[1])
		pp.pprint(data)
		# plots the results
		plot(results)
	else:
		raise ValueError('No result files available')