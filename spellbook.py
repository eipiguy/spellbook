from loader import SpellbookLoader
from interface import SpellbookInterface

from header import *

# A Spellbook is a local prompt interface with a particular sense of identity.
# It is linked to a particular directory it has access to for taking notes and storing/retrieving references.
class Spellbook:

	def __init__( self, directory ):

		# The directory the spellbook has access to
		self.directory = directory

		# The loader is a way to store and retrieve data in the directory.
		self.loader = SpellbookLoader( directory )

		# The interface is what you talk to.
		# It communicates behind the scenes to formulate intelligent responses.
		self.interface = SpellbookInterface( self.loader )
		self.interface.chat()

if __name__ == '__main__':
	# When we start the spellbook, we spin up Alice as the default personality.
	alice = Spellbook( ASSET_DIR )
