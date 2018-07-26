import glob
from pprint import pprint
import json
import unittest

from common.visualisation import plot_last, load_results

class TestVisualisation(unittest.TestCase):
	@unittest.skip
	def test_load_results(self):
		load_results('networks/test/test.p')

	def test_list_directories(self):
		plot_last()