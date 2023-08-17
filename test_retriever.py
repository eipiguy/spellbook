import unittest
from os import path

from spellbook import *

this_dir = path.dirname( path.realpath( __file__ ) )
ROOT_TEST_DIR = path.join(this_dir, 'assets', 'tests')

class TestEmbedding(unittest.TestCase):

	def setUp(self):
		self.spellbook_loader = SpellbookLoader(ROOT_TEST_DIR)

	def test_txt_file_1of2_exists(self):
		pass


if __name__ == '__main__':
	unittest.main()
