import glob
from pprint import pprint
import json
from unittest import TestCase

from common.visualisation import plot_last, load_results

class TestVisualisation(TestCase):
	# def test_load_results(self):
		# load_results('networks/test/test.p')

	def test_list_directories(self):
		plot_last()