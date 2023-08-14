from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from os import path
import time

this_dir = path.dirname( path.realpath( __file__ ) )

ASSET_DIR = path.join(this_dir, 'assets')
MODEL_DIR = path.join(this_dir, 'models')
EMBED_DIR = path.join(this_dir, 'embeds')

dt = []

print('Loading documents ...')
st = time.time()

docs = []

txt_loader = DirectoryLoader(ASSET_DIR, glob="*.txt", recursive=True, show_progress=True, use_multithreading=True, loader_cls=TextLoader)
docs.extend( txt_loader.load() )

md_loader = DirectoryLoader(ASSET_DIR, glob="*.md", recursive=True, show_progress=True, use_multithreading=True, loader_cls=UnstructuredMarkdownLoader)
docs.extend( md_loader.load() )

pdf_loader = DirectoryLoader(ASSET_DIR, glob="*.pdf", recursive=True, show_progress=True, use_multithreading=True, loader_cls=PyPDFLoader)
docs.extend(pdf_loader.load())

dt.append(time.time() - st)
print(f'Docs loaded: {dt[-1]} seconds.') 
print(f'Tokenizing ...') 
st = time.time()

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = 300,
	chunk_overlap  = 20,
	length_function = len,
)

tokenized_txt_docs = text_splitter.create_documents(
	[doc.page_content for doc in docs],
	metadatas = [doc.metadata for doc in docs])
sentences = [ chunk.page_content for chunk in tokenized_txt_docs ]

dt.append(time.time() - st)
print(f'Tokenized: {dt[-1]} seconds.') 
print(f'Loading model ...') 
st = time.time()

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder=MODEL_DIR)

dt.append(time.time() - st)
print(f'Model Loaded: {dt[-1]} seconds.')
print(f'Computing embeddings ...') 
st = time.time()

vectordb = Chroma.from_documents(tokenized_txt_docs, embedding_model, persist_directory=EMBED_DIR)

dt.append(time.time() - st)
print(f'Embedded Docs: {dt[-1]} seconds.')
print(f'Writing embeddings database ...') 
st = time.time()

vectordb.persist()

dt.append(time.time() - st)
print(f'Wrote Database: {dt[-1]} seconds.')

continue_queries = True
while continue_queries:
	#query = "What did the president say about Ketanji Brown Jackson"
	query = input("Similarity Search: ")

	st = time.time()
	docs = vectordb.similarity_search(query)
	print(f"Closest:\n{docs[0].page_content}")

	dt.append(time.time() - st)
	print(f"Response time: {dt[-1]} seconds")