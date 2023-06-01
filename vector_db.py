from langchain.document_loaders import DirectoryLoader#, ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import time, pickle

ASSET_DIR='D:\\Dropbox\\projects\\code\\spellbook\\assets\\'
MODEL_DIR='D:\\Dropbox\\projects\\code\\spellbook\\models\\'
EMBED_DIR='D:\\Dropbox\\projects\\code\\spellbook\\embeds\\'

dt = []

print('Loading documents ...')
st = time.time()

loader = DirectoryLoader(ASSET_DIR, glob="*.txt", recursive=True, show_progress=True, use_multithreading=True)
docs = loader.load()

dt.append(time.time() - st)
print(f'Docs loaded: {dt[-1]} seconds.') 
print(f'Tokenizing ...') 
st = time.time()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,
    chunk_overlap  = 20,
    length_function = len,
)
tokenized_docs = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
sentences = [ chunk.page_content for chunk in tokenized_docs ]

dt.append(time.time() - st)
print(f'Tokenized: {dt[-1]} seconds.') 
print(f'Loading model ...') 
st = time.time()

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder=MODEL_DIR)

dt.append(time.time() - st)
print(f'Model Loaded: {dt[-1]} seconds.')
print(f'Computing embeddings ...') 
st = time.time()

vectordb = Chroma.from_documents(tokenized_docs, embedder, persist_directory=EMBED_DIR)

dt.append(time.time() - st)
print(f'Embedded Docs: {dt[-1]} seconds.')
print(f'Writing embeddings database ...') 
st = time.time()

vectordb.persist()

dt.append(time.time() - st)
print(f'Wrote Database: {dt[-1]} seconds.')
print(f'Query tests ...') 
st = time.time()

query = "What did the president say about Ketanji Brown Jackson"
print(query)
docs = vectordb.similarity_search(query)
print(docs[0].page_content)

dt.append(time.time() - st)
print(f'Queries Run: {dt[-1]} seconds.')