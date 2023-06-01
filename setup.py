from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain import HuggingFacePipeline

import os

loader = TextLoader('./assets/state_of_the_union.txt', encoding='utf8')
index = VectorstoreIndexCreator().from_loaders([loader])
query = "Give a summary of the contents of the speech please."
response = index.query_with_sources(query)

print(response)