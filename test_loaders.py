import unittest
from os import path

from spellbook import *

this_dir = path.dirname( path.realpath( __file__ ) )

ROOT_TEST_DIR = path.join(this_dir, 'assets', 'tests')

class TestTxtLoader(unittest.TestCase):

	def setUp(self):
		self.spellbook_loader = SpellbookLoader(ROOT_TEST_DIR)

	def test_txt_file_1of2_exists(self):
		test_file_path_1of2 = path.join(ROOT_TEST_DIR, 'text_loader_test_doc_1of2.txt')
		self.assertTrue(path.isfile(test_file_path_1of2))

	def test_txt_file_2of2_exists(self):
		test_file_path_1of2 = path.join(ROOT_TEST_DIR, 'text_loader_test_doc_2of2.txt')
		self.assertTrue(path.isfile(test_file_path_1of2))

	def test_find_txt_files(self):
		test_docs = self.spellbook_loader.docs['txt']
		self.assertEqual( len(test_docs), 2 )

	def test_load_txt_as_langchain_docs(self):
		test_docs = self.spellbook_loader.docs['txt']
		self.assertEqual( str( type(test_docs[0]) ), "<class 'langchain.schema.Document'>" )


class TestMdLoader(unittest.TestCase):

	def setUp(self):
		self.spellbook_loader = SpellbookLoader(ROOT_TEST_DIR)

	def test_md_file_1of2_exists(self):
		test_file_path_1of2 = path.join(ROOT_TEST_DIR, 'markdown_loader_test_doc_1of2.md')
		self.assertTrue(path.isfile(test_file_path_1of2))

	def test_md_file_2of2_exists(self):
		test_file_path_2of2 = path.join(ROOT_TEST_DIR, 'markdown_loader_test_doc_2of2.md')
		self.assertTrue(path.isfile(test_file_path_2of2))

	def test_finds_md_files(self):
		test_docs = self.spellbook_loader.docs['md']
		self.assertEqual(len(test_docs), 2)

	def test_load_md_as_langchain_docs(self):
		test_docs = self.spellbook_loader.docs['md']
		self.assertEqual( str( type(test_docs[0]) ), "<class 'langchain.schema.Document'>" )

class TestPyLoader(unittest.TestCase):

	def setUp(self):
		self.spellbook_loader = SpellbookLoader(ROOT_TEST_DIR)

	def test_py_file_1of2_exists(self):
		test_file_path_1of2 = path.join(ROOT_TEST_DIR, 'python_loader_test_doc_1of2.py')
		self.assertTrue(path.isfile(test_file_path_1of2))

	def test_py_file_2of2_exists(self):
		test_file_path_2of2 = path.join(ROOT_TEST_DIR, 'python_loader_test_doc_2of2.py')
		self.assertTrue(path.isfile(test_file_path_2of2))

	def test_finds_py_files(self):
		test_docs = self.spellbook_loader.docs['py']
		self.assertEqual(len(test_docs), 2)

	def test_load_py_as_langchain_docs(self):
		test_docs = self.spellbook_loader.docs['py']
		self.assertEqual( str( type(test_docs[0]) ), "<class 'langchain.schema.Document'>" )


if __name__ == '__main__':
	unittest.main()
