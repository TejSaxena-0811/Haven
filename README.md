# Haven: RAG AI Agent

Haven is a local and offline RAG-based document assistant that allows users to ask questions about their own documents without sending their data to a cloud-based AI service. It uses locally running AI models to retrieve relevant information and generate answers while keeping the documents on the user's own machine.


# A chart explaining the complete flow:
<img width="421" height="1040" alt="image" src="https://github.com/user-attachments/assets/f4b4e66b-652e-4a5e-b42d-8e1a536ab0f7" />

---

## Features

- Ask questions about PDF, CSV, and DOCX files
- Local document processing using Ollama
- Offline RAG-based question answering
- Local vector storage using ChromaDB
- Supports CSV-based data analysis
- No external LLM API required
- Documents remain on the user's local machine
- Automatically detects the uploaded file type

---

## Tech Stack

### AI and RAG

- Ollama
- Llama 3.2
- mxbai-embed-large
- LangChain
- ChromaDB

### Document Processing

- CSVLoader
- PyPDFLoader
- Docx2txtLoader
- Pandas

### Language

- Python

---

## How It Works

```text
Document
   |
   v
File Loader
   |
   v
LangChain Documents
   |
   v
mxbai-embed-large
   |
   v
ChromaDB
   |
   v
Relevant Information
   |
   v
Llama 3.2
   |
   v
Answer
```


## Haven uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the local document before providing it to the language model.

The language model and embedding model run locally through Ollama, allowing the core application to work without an internet connection after the required models and dependencies have been installed.

## A safe (privacy) "haven" for the user's data, which runs without any wifi service.
