import os

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader
)


DATA_FOLDER = "./data"


def load_file():

    files = os.listdir(DATA_FOLDER)

    if not files:
        raise Exception("No file found in the data folder.")

    if len(files) > 1:
        raise Exception("Please keep only one file inside the data folder.")

    file_name = files[0]
    file_path = os.path.join(DATA_FOLDER, file_name)

    extension = os.path.splitext(file_name)[1].lower()

    if extension == ".csv":
        loader = CSVLoader(file_path)

    elif extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    else:
        raise Exception(
            "Unsupported file type. Use CSV, PDF, or DOCX."
        )

    documents = loader.load()

    return documents