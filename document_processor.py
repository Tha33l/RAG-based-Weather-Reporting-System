import os
import fitz
import nltk
import chromadb
import re
import torch
import types

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from pathlib import Path

# Downloading chunking library
# nltk.download('punkt')
# nltk.download('punkt_tab')

if not hasattr(torch.distributed, 'is initialized'):
    torch.distributed.is_initialized = types.MethodType(lambda self = None: 0, torch.distributed)
if not hasattr(torch.distributed, 'get_rank'):
    torch.distributed.get_rank= types.MethodType(lambda self = None: 0, torch.distributed)

base_dir = Path(__file__).resolve().parent
folder_path = "weather_documents"
weatherInfo_db_path = "Info_db"
embedding_model = "all-MiniLM-L6-v2"

splitter = NLTKTextSplitter(chunk_size=600)
embedder = SentenceTransformer(embedding_model)  # Loading the embedding model

# Creating vector databases
# Weather Info Database
docs_db = chromadb.PersistentClient(
    path="backend/vector_databases/weatherInfo_db_path")
docs_collection = docs_db.get_or_create_collection("weather_chunks")


def info_doc_embeddings(chunks):
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [t[1] for t in batch]
        ids = [f"{t[0]}_{i+j}" for j, t in enumerate(batch)]
        metadatas = [{"source": t[0]} for t in batch]

        embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()
        docs_collection.add(
            documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)


# Reading PDF using Python Library


def pdf_loader(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


# Loading Documents from Folder


def document_loader(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            text = pdf_loader(filepath)
        else:
            print(f"Skipping unsupported file")
            continue

        docs.append({"name": filename, "text": text})
    return docs


def main():
    documents = document_loader(folder_path)
    # Chunking each Document
    for doc in documents:

        chunks = [(doc['name'], chunk)
                  for chunk in splitter.split_text(doc['text'])]
        print(f"Total number of chunks are {len(chunks)}\n")

      
    
    all_docs = docs_collection.get()
    print(all_docs["documents"][:5])
    info_doc_embeddings(chunks)


main()