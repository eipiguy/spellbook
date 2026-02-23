## This file sets the key directory locations for use by all the scripts

from os import path

# set this to True to enable global verbose output
ROOT_DEBUG = False

# Directory variables:
# Spellbook/    (the root directory for the project)
#    assets     (human readable files to learn/use as reference)
#    models     (the language models downloaded for local use)
#    embeds     (the databases containing the parsed/learned data)

THIS_DIR = path.dirname( path.realpath( __file__ ) )
ASSET_DIR = path.join(THIS_DIR, 'assets')
MODEL_DIR = path.join(THIS_DIR, 'models')
EMBED_DIR = path.join(THIS_DIR, 'embeds')
IDENTITIES_DIR = path.join(THIS_DIR, 'identities')
