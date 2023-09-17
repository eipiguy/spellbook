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
		self.asset_dir = directory

		# Loaders/splitters used per file type
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
		self.chunk_size = 600
		self.chunk_overlap = 400
		self.length_function = len

		self.retrieval_chunks = 10

		data_length = ( self.chunk_size * self.retrieval_chunks ) \
			- ( self.chunk_overlap * (self.retrieval_chunks - 1) )
		print(f"Retrieval data length = {data_length}")

		# Process given directory

		# Raw documents are loaded and stored
		self.assets = self.load_docs( directory )

		# Raw documents are split and embedded in database
		self.embed_assets( self.split_docs( self.assets ) )


	def load_docs( self, directory ):
		print(f'\nLoading docs from {directory}...') 
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
			print(f"{num_docs} .{ext} docs to split.")
			
			splitter = self.text_splitters[ext](
				chunk_size = self.chunk_size,
				chunk_overlap  = self.chunk_overlap,
				length_function = self.length_function,
				add_start_index = True
			)

			content = []
			metadata = []
			for doc in docs[ext]:
				doc_path = doc.metadata['source']
				cur_metadata = doc.metadata
				cur_metadata['extension'] = ext
				cur_metadata['created'] = f"{date.fromtimestamp(path.getctime(doc_path))}"
				cur_metadata['modified'] = f"{date.fromtimestamp(path.getmtime(doc_path))}"

				content.append( doc.page_content )
				metadata.append( cur_metadata )

			tokenized_docs[ext] = splitter.create_documents(
				texts = content,
				metadatas = metadata
			)

		dt = time.time() - st
		print(f"{new_docs} docs split in {dt} seconds.\n")
		return tokenized_docs


	def load_model( self ):
		print(f"\nLoading embedding model `{self.embedding_model_name}`...") 
		st = time.time()

		self.embedding_model = HuggingFaceEmbeddings(
			model_name = self.embedding_model_name,
			cache_folder = MODEL_DIR,
			model_kwargs = {'device': 'cuda'},
			encode_kwargs = {'normalize_embeddings': True}
		)

		print(f"Model loaded in {time.time() - st} seconds.")


	def embed_tokens( self, tokens ):
		print(f"\nAdding tokens to database...")
		st = time.time()

		# Each file's tokens should be trackable;
		# This sets the token's file name as the id,
		# and the rest of this function adds the split information.
		self.token_ids = [ token.metadata['source'] for token in tokens ]

		# if the file has been split into pages, account for that in the id.
		self.token_ids = []
		for token in tokens:
			id = f"{token.metadata['source']} {token.metadata['modified']}"
			if 'page' in token.metadata:
				id += f" pg{token.metadata['page']}"
			self.token_ids.append(id)


		# Initial conditions
		last_source = self.token_ids[0]
		cur_split_num = 0
		last_i = 0

		for i in range( len(tokens) ):

			# If in a new document, we need to store the tagged tokens
			# from the last one, and then reset
			if( self.token_ids[i] != last_source ):
				
				# The last token number for the last document 
				num_splits = cur_split_num

				print(f"Embedding {num_splits+1} splits of {last_source}.")
				sst = time.time()

				# Add last token number to end of token ids
				for j in range(num_splits):
					self.token_ids[last_i+j] += f"-{num_splits}"

				# Add tokens to collection
				self.collection.add(
					ids = self.token_ids[last_i:i],
					documents = [ token.page_content for token in tokens[last_i:i] ],
					metadatas = [ token.metadata for token in tokens[last_i:i] ]
				)

				print(f"Finished {last_source} in {time.time() - sst} seconds.")
				
				cur_split_num = 0
				last_source = self.token_ids[i]
				last_i = i

			# Still in the middle of a document. Append current split number to id
			self.token_ids[i] += f" {cur_split_num}"
			cur_split_num += 1

		print(f"\nAll tokens added in {time.time() - st} seconds.\n")


	def embed_assets( self, tokens, collection_name = 'assets' ):

		self.load_model()

		print(f"\nStarting database client...") 
		st = time.time()

		# Set up database
		self.vectordb = PersistentClient( path = EMBED_DIR )
		self.collection = self.vectordb.get_or_create_collection(collection_name)

		# Embed documents found for each file extension
		for ext in tokens:
			print(f"\nAdding {ext} file tokens to database...") 
			self.embed_tokens(tokens[ext])

		print(f"\nStarting local vectorstore from database...") 
		st = time.time()

		self.vectorstore = Chroma(
			client = self.vectordb,
			collection_name = collection_name,
			embedding_function = self.embedding_model
		)

		print(f"Vectorstore started in {time.time() - st} seconds.\n")
		print(f"There are {self.vectorstore._collection.count()} tokens in the collection.")


	def format_token_id( self, source_path, page_num, time_modified, start_id, end_id ):
		return f"{source_path} {time_modified} pg{page_num} {start_id}-{end_id}"


	def parse_token_id( self, id ):
		id_values = id.split()
		id_meta = {}

		id_meta['source'] = id_values[0]
		id_meta['modified_date'] = id_values[1]
		id_meta['modified_time'] = id_values[2]

		# if there is a page value in the id
		if id_values[3] == id_values[-2]:
			id_meta['page'] = id_values[3].split('pg')[-1]

		token_split = id_values[-1]
		token_split = token_split.split('-')
		
		id_meta['start_index'] = token_split[0]
		id_meta['out_of'] = token_split[1]

		return id_meta

	def combine_token_content( self, tokens ):
		# Find relevant document paths
		source_ids = tokens['ids'][0]
		ids_metas = [ self.parse_token_id( id ) for id in source_ids ]
		assets = tokens['documents'][0]

		# find all content related to those documents: page, split
		paginated_contents = {}
		for i,meta in enumerate(ids_metas):
			doc_path = meta['source']
			page = meta['page'] if 'page' in meta else 0
			start_id = meta['start_index']
			out_of = meta['out_of']

			# Map relevant page and split index
			if not doc_path in paginated_contents:
				paginated_contents[ doc_path ] = []
			paginated_contents[ doc_path ].append( [ int(page), int(start_id), int(out_of), assets[i] ] )

		# now we have a list of (pg,split) values per source,
		# sort each of them and combine contents
		combined_contents = {}
		for doc in paginated_contents:
			combined_contents[doc] = combine_snippets( paginated_contents[doc] )
			# combined_contents = { doc_path: combined_content_string }

		return combined_contents


	def retrieve( self, query ):
		# Search using embedding model
		potentials = self.collection.query(
			query_texts = [query],
			n_results = self.retrieval_chunks
			#where={"metadata_field": "is_equal_to_this"},
			#where_document={"$contains":"search_string"}
		)

		# Combine tokens from common docs
		# [(doc,pgs,splits), data] -> [doc, combined_data]
		combined_retrieval_string = ""
		data_source_map = self.combine_token_content(potentials)
		for data_source in data_source_map:
			combined_retrieval_string += f"{data_source}\n{data_source_map[ data_source ]}\n\n"
		return combined_retrieval_string


def check_neighbors( page, split, last_page, last_split, last_out_of ):
	# Same page
	if page == last_page:
		# check split == last+1
		return split == last_split+1
	# Different pages
	# Page must be last+1
	elif page == last_page+1:
		# Split must be 0, last split must be at the end of its page (out_of)
		return split == 0 and last_split == last_out_of
	return False


def combine_snippets(doc_snippets):
	doc_snippets.sort()
	last_page = -1
	last_split = -1
	last_out_of = -1
	combined_content = ""
	for snippet in doc_snippets:
		page = snippet[0]
		split = snippet[1]
		out_of = snippet[2]
		content = snippet[3]
		if check_neighbors( page, split, last_page, last_split, last_out_of ):
			# combine
			combined_content = merge_strings( combined_content, content )
		else:
			# append with ellipses
			combined_content += f"\n...\n{content}"

		last_page = page
		last_split = split
		last_out_of = out_of
	return combined_content


def merge_strings(s1: str, s2: str) -> str:

	# Merges two strings s1 and s2
	# based on unique overlapping characters
	# at the end of s1 and the start of s2.
	for i in range(1, min(len(s1), len(s2)) + 1):
		if s1[-i:] == s2[:i]:
			return s1 + s2[i:]
	return s1 + s2


if __name__ == '__main__':
	SpellbookLoader(ASSET_DIR)