# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# import os
# import pandas as pd

# df = pd.read_csv("realistic_restaurant_reviews.csv")
# embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

# db_location = "./chroma_langchain_db"

# add_documents = not os.path.exists(db_location)

# if(add_documents):
#     documents = []
#     ids = []

#     for i, row in df.iterrows():
#         document = Document(
#             page_content = row["Title"] + " " + row["Review"],
#             metadata = {"rating": row["Rating"], "date": row["Date"]},
#             id = str(i)
#         )

#         ids.append(str(i))
#         documents.append(document)

# vector_store = Chroma(
#     collection_name = "restaurant_reviews",
#     persist_directory = db_location,
#     embedding_function = embeddings
# )

# if add_documents:
#     vector_store.add_documents(documents = documents , ids = ids)

# retriever = vector_store.as_retriever(
#     search_kwargs = {"k": 5}
# )




















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