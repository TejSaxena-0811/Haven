from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
import shutil

from loader import load_file

DB_LOCATION = "./chroma_langchain_db"

documents = load_file()

# creating the embeddings
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

# deleting old database, which makes sure that old data does not remain in chromadb.
if os.path.exists(DB_LOCATION):
    shutil.rmtree(DB_LOCATION)

# create chromadb
vector_store = Chroma(
    collection_name="documents",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

# unique id to every doc
ids = [str(i) for i in range(len(documents))]

# add docs to chromadb
vector_store.add_documents(
    documents=documents,
    ids=ids
)


# only the 5 most relevant docss will be retrieved
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
