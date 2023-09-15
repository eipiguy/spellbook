import unittest
from os import path

from loader import *

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
		test_docs = self.spellbook_loader.assets['txt']
		self.assertEqual( len(test_docs), 2 )

	def test_load_txt_as_langchain_docs(self):
		test_docs = self.spellbook_loader.assets['txt']
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
		test_docs = self.spellbook_loader.assets['md']
		self.assertEqual(len(test_docs), 2)

	def test_load_md_as_langchain_docs(self):
		test_docs = self.spellbook_loader.assets['md']
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
		test_docs = self.spellbook_loader.assets['py']
		self.assertEqual(len(test_docs), 2)

	def test_load_py_as_langchain_docs(self):
		test_docs = self.spellbook_loader.assets['py']
		self.assertEqual( str( type(test_docs[0]) ), "<class 'langchain.schema.Document'>" )

class TestMergeStrings(unittest.TestCase):

	def test_basic_functionality(self):
		self.assertEqual(merge_strings("abcdef", "defghi"), "abcdefghi")

	def test_no_overlap(self):
		self.assertEqual(merge_strings("abcdef", "ghijkl"), "abcdefghijkl")

	def test_single_char_overlap(self):
		self.assertEqual(merge_strings("abcdef", "fghijk"), "abcdefghijk")

	def test_case_sensitivity(self):
		self.assertEqual(merge_strings("abcdef", "DEFghi"), "abcdefDEFghi")

	def test_whitespace(self):
		self.assertEqual(merge_strings("abc def", " defghi"), "abc defghi")

	def test_empty_strings(self):
		self.assertEqual(merge_strings("", "abc"), "abc")

	def test_non_unique_overlap(self):
		self.assertEqual(merge_strings("abcabc", "abcabc"), "abcabcabc")

	def test_long_strings(self):
		s1 = "a" * int(1e6)
		s2 = "a" * int(1e6)
		self.assertEqual(merge_strings(s1, s2), s1 + s2[1:])

	def test_special_characters(self):
		self.assertEqual(merge_strings("abc$%#", "$%#def"), "abc$%#def")

	def test_unicode(self):
		self.assertEqual(merge_strings("abc😀", "😀def"), "abc😀def")

# Running the unittests
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestMergeStrings))


if __name__ == '__main__':
	unittest.main()
