from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
import json
import os
from dotenv import load_dotenv, find_dotenv

import asyncio
_ = load_dotenv(find_dotenv(),override=True)

FILE_PATH = os.getenv("URLs_JSON_FILE_PATH", "/ai-chatbot/data/energySavingUrls.json")
PDF_FOLDER_PATH = os.getenv("PDF_FOLDER_PATH", "/ai-chatbot/data/pdfs")
async def get_vectorstore():
    """
    Returns the vectorstore using ChromaDB
    """
    # Add web documents
    try:
        with open(FILE_PATH, "r") as file:
            urls_refs = json.load(file)
    except FileNotFoundError:
        raise Exception(f"File not found at: {FILE_PATH}")


    urls = urls_refs.get("energy_saving_resources", [])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            data = loader.load()
            
            split_docs = text_splitter.split_documents(data)
            documents.extend(split_docs)
        except Exception as e:
            print(f"Failed to load URL {url}: {e}")

    # add pdf documents
    for file in os.listdir(PDF_FOLDER_PATH):
        pdf_file_path = os.path.join(PDF_FOLDER_PATH, file)
        pdf_loader = PyPDFLoader(pdf_file_path)
        pages = []
        async for page in pdf_loader.alazy_load():
            pages.append(page)
        split_pdf_docs = text_splitter.split_documents(pages)
        documents.extend(split_pdf_docs)

    


    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = InMemoryVectorStore.from_documents(documents=documents, embedding=local_embeddings)
    return vectorstore


### Uncomment if you want to test the retriever ###
# vectorstore = asyncio.run(get_vectorstore())

# q = "how can I do energy saving?"

# docs = vectorstore.similarity_search(q, k=10)

# for doc in docs:
#     print(doc)