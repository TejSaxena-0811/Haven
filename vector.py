from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
import shutil

from loader import load_file

DB_LOCATION = "./chroma_langchain_db"

# Load the file from the data folder
documents = load_file()

# Create embeddings
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

# Delete old database
# This ensures that when you replace the file,
# old data does not remain in ChromaDB.
if os.path.exists(DB_LOCATION):
    shutil.rmtree(DB_LOCATION)

# Create ChromaDB
vector_store = Chroma(
    collection_name="documents",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

# Give every document a unique ID
ids = [str(i) for i in range(len(documents))]

# Add documents to ChromaDB
vector_store.add_documents(
    documents=documents,
    ids=ids
)

# Create retriever
# Only the 5 most relevant documents will be retrieved
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)