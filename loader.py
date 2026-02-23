from header import *
from tools import *
from os import walk
from os.path import splitext

# set this to True to print verbose output
DEBUG = ROOT_DEBUG or False

class SpellbookLoader:
	def __init__( self, path = ASSET_DIR ):
		if DEBUG: print( f"New loader at: {path}" )
		self.path = path
		# extensions to list of all files with said extension
		self.file_paths = {}
		self.find_files( True )

	def find_files( self, first_look = False ):
		for cur_dir, folders, files in walk( self.path ):
			if DEBUG: print( f"Current directory: {cur_dir}" )
			for file_name in files:
				name_ext = splitext( file_name )
				file_title = name_ext[0]
				file_ext = name_ext[-1]
				file_path = path.join( cur_dir, file_name )

				# if this is the first of this file type,
				# start a new list in the path dictionary
				if not file_ext in self.file_paths:
					if DEBUG: print( f"New type {file_ext}! Adding {file_path}" )
					self.file_paths[ file_ext ] = [ file_path ]
				elif first_look or not file_path in self.file_paths[ file_ext ]:
					if DEBUG: print( f"Adding {len( self.file_paths[ file_ext ] ) + 1 }th {file_ext} file: {file_path}" )
					self.file_paths[ file_ext ].append( file_path )
		return


if __name__ == '__main__':
	loader = SpellbookLoader()
	for ext in loader.file_paths:
		for file in loader.file_paths[ext]:
			pages = read_text( file )
			if DEBUG: print( pages )

