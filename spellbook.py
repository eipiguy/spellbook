from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader, PythonLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, PythonCodeTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from os import path
this_dir = path.dirname( path.realpath( __file__ ) )

ASSET_DIR = path.join(this_dir, 'assets')
MODEL_DIR = path.join(this_dir, 'models')
EMBED_DIR = path.join(this_dir, 'embeds')

class Spellbook:
	def __init__( self, directory ):
		self.directory = directory

		self.loader = SpellbookLoader(directory)
		print('test')

class SpellbookLoader:
	def __init__( self, directory ):
		self.directory = directory

		# Load all recognized file types with their appropriate loaders
		self.supported_file_extensions = {
			'txt': TextLoader, 
			'md': UnstructuredMarkdownLoader,
			'py': PythonLoader }
		self.text_splitters = {
			'txt': RecursiveCharacterTextSplitter,
			'md': MarkdownTextSplitter,
			'py': PythonCodeTextSplitter }

		self.docs = {}
		for ext in self.supported_file_extensions:
			self.docs[ext] = DirectoryLoader(
				directory,
				glob = '*.' + ext,
				recursive = True,
				show_progress = True,
				use_multithreading = True,
				loader_cls = self.supported_file_extensions[ext]).load()

		tokenized_docs = {}
		sentences = {}
		for ext in self.text_splitters:
			splitter = self.text_splitters[ext](
				chunk_size = 300,
				chunk_overlap  = 20,
				length_function = len,
			)

			tokenized_docs[ext] = splitter.create_documents(
				[doc.page_content for doc in self.docs[ext]],
				metadatas = [doc.metadata for doc in self.docs[ext]])
			sentences[ext] = [ chunk.page_content for chunk in tokenized_docs[ext] ]

if __name__ == '__main__':
	bob = Spellbook(path.join(this_dir,'..'))