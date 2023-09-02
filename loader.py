from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader, PythonLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, PythonCodeTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from os import path
from datetime import datetime as date
import time
from chromadb import PersistentClient

THIS_DIR = path.dirname( path.realpath( __file__ ) )
ASSET_DIR = path.join(THIS_DIR, 'assets')
MODEL_DIR = path.join(THIS_DIR, 'models')
EMBED_DIR = path.join(THIS_DIR, 'embeds')

class SpellbookLoader:
	def __init__( self, directory ):
		self.directory = directory

		# Load all recognized file types with their appropriate loaders
		self.text_loaders = {
			'txt': TextLoader, 
			'md': UnstructuredMarkdownLoader,
			'pdf': PyPDFLoader,
			'py': PythonLoader }

		self.text_splitters = {
			'txt': RecursiveCharacterTextSplitter,
			'md': MarkdownTextSplitter,
			'pdf': RecursiveCharacterTextSplitter,
			'py': PythonCodeTextSplitter }

		self.embedding_model_name = "BAAI/bge-small-en"

		self.docs = self.load_docs( directory )
		self.embed_tokens( self.split_docs( self.docs ) )


	def load_docs( self, directory ):
		print(f'\nLoading docs from filesystem...') 
		st = time.time()

		docs = {}
		new_docs = 0
		for ext in self.text_loaders:
			docs[ext] = DirectoryLoader(
				directory,
				glob = '*.' + ext,
				recursive = True,
				show_progress = True,
				use_multithreading = True,
				loader_cls = self.text_loaders[ext]).load()
			num_docs = len(docs[ext])
			new_docs += num_docs
		
		dt = time.time() - st
		print(f"{new_docs} docs loaded in {dt} seconds.\n")
		return docs


	def split_docs( self, docs ):
		print(f'\nSplitting docs into tokens...') 
		st = time.time()

		tokenized_docs = {}
		new_docs = 0
		for ext in self.text_splitters:
			num_docs = len(docs[ext])
			new_docs += num_docs
			print(f"{num_docs} *.{ext} docs to split.")
			
			splitter = self.text_splitters[ext](
				chunk_size = 1000,
				chunk_overlap  = 500,
				length_function = len,
			)

			content = []
			metadata = []
			for doc in docs[ext]:
				doc_path = doc.metadata['source']
				cur_metadata = {
					'extension': ext,
					'source': doc_path,
					'created': f"{date.fromtimestamp(path.getctime(doc_path))}",
					'modified': f"{date.fromtimestamp(path.getmtime(doc_path))}"
				}

				content.append( doc.page_content )
				metadata.append( cur_metadata )

			tokenized_docs[ext] = splitter.create_documents(
				texts = content,
				metadatas = metadata
			)

		dt = time.time() - st
		print(f"{new_docs} docs split in {dt} seconds.\n")
		return tokenized_docs


	def embed_tokens( self, tokens, collection_name = 'assets' ):
		print(f"\nLoading embedding model `{self.embedding_model_name}`...") 
		st = time.time()

		self.embedding_model = HuggingFaceEmbeddings(
			model_name = self.embedding_model_name,
			cache_folder = MODEL_DIR,
			model_kwargs = {'device': 'cuda'},
			encode_kwargs = {'normalize_embeddings': True}
		)

		dt = time.time() - st
		print(f"Model loaded in {dt} seconds.")
		print(f"\nStarting database client...") 
		st = time.time()

		self.vectordb = PersistentClient( path = EMBED_DIR )
		self.collection = self.vectordb.get_or_create_collection(collection_name)
		for ext in tokens:
			print(f"\nAdding {ext} files to database...") 

			ext_tokens = tokens[ext]
			cur_source_split = 0
			self.token_ids = [ token.metadata['source'] for token in ext_tokens ]

			last_source = self.token_ids[0]
			last_i = 0

			for i in range(len(ext_tokens)):
				if( self.token_ids[i] != last_source ):
					num_splits = cur_source_split
					print(f"Embedding {num_splits} splits of {last_source}.")
					st = time.time()
					for j in range(num_splits):
						self.token_ids[last_i+j] += f"-{num_splits}"
					self.collection.add(
						ids = self.token_ids[last_i:i],
						documents = [ token.page_content for token in ext_tokens[last_i:i] ],
						metadatas = [ token.metadata for token in ext_tokens[last_i:i] ]
					)
					dt = time.time() - st
					print(f"Finished {last_source} in {dt} seconds.")
					cur_source_split = 0
					last_source = self.token_ids[i]
					last_i = i

				self.token_ids[i] = f"{self.token_ids[i]}_{ext_tokens[i].metadata['modified']}_{cur_source_split}"
				cur_source_split += 1

			print(f"All {ext} files embedded.\n")

		print(f"\nStarting local vectorstore from database...") 
		st = time.time()

		self.vectorstore = Chroma(
			client = self.vectordb,
			collection_name = collection_name,
			embedding_function = self.embedding_model
		)

		dt = time.time() - st
		print(f"Vectorstore started in {dt} seconds.\n")
		print("There are", self.vectorstore._collection.count(), "in the collection")

if __name__ == '__main__':
	SpellbookLoader(ASSET_DIR)