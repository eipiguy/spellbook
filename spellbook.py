from loader import SpellbookLoader
from interface import SpellbookInterface

from os import path
this_dir = path.dirname( path.realpath( __file__ ) )

ASSET_DIR = path.join(this_dir, 'assets')
MODEL_DIR = path.join(this_dir, 'models')
EMBED_DIR = path.join(this_dir, 'embeds')

class Spellbook:
	def __init__( self, directory ):
		self.directory = directory
		self.memory = {}

		self.loader = SpellbookLoader(directory)
		self.interface = SpellbookInterface()

		self.interface.chat()


if __name__ == '__main__':
	bob = Spellbook(path.join(this_dir,'..'))