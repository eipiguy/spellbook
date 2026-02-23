from header import *
from os.path import splitext
from pypdf import PdfReader

import string
from nltk.corpus import stopwords

# set this to True to print verbose output
DEBUG = ROOT_DEBUG or False

def read_text( file_path ):
	ext = splitext( file_path )[-1]
	if DEBUG: print( f"Reading text from {ext} file, {file_path}" )
	pages = [] # pages are lists of lines of text

	if ext == '.txt':
		with open( file_path, 'r' ) as file:
			lines = file.readlines()
			lines = [ line.replace( '\n', '' ) for line in lines ]
			pages.append( lines )
	if ext == '.pdf':
		with open( file_path, 'rb' ) as file:
			reader = PdfReader( file )
			for page in reader.pages:
				lines = page.extract_text().split('\n')
				pages.append( lines )
	else:
		print( f"No current {ext} support! Sorry =(" )

	return pages

def simplify( text_line ):
	stop_words = set( stopwords.words('english') )
	phrases = text_line.split( string.punctuation )
	simple_sentence = []
	for phrase in phrases:
		for word in phrase:
			word = word.lower()
			if not word in stop_words:
				simple_sentence.append( word )
