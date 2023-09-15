from loader import SpellbookLoader
from interface import SpellbookInterface

from os import path
THIS_DIR = path.dirname( path.realpath( __file__ ) )

ASSET_DIR = path.join(THIS_DIR, 'assets')
MODEL_DIR = path.join(THIS_DIR, 'models')
EMBED_DIR = path.join(THIS_DIR, 'embeds')

class Spellbook:
	def __init__( self, directory ):
		self.directory = directory

		self.loader = SpellbookLoader( directory )
		self.interface = SpellbookInterface( self.loader )
		self.interface.chat()


if __name__ == '__main__':
	alice = Spellbook( ASSET_DIR )
	#bob = Spellbook( path.join(THIS_DIR,'..') )