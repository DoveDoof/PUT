import glob
from pprint import pprint
import json
from unittest import TestCase

from common.visualisation import plot_last

class TestVisualisation(TestCase):
	def test_list_directories(self):
		plot_last()
