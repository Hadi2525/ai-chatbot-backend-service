from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import json
import os
from dotenv import load_dotenv, find_dotenv
from uuid import uuid4
from datetime import datetime
import time
_ = load_dotenv(find_dotenv(),override=True)

URL_JSON_FILE_PATH = os.getenv("URLs_JSON_FILE_PATH", "/ai-chatbot/data/energySavingUrls.json")
PDF_FOLDER_PATH = os.getenv("PDF_FOLDER_PATH", "/ai-chatbot/data/pdfs")
CONN_STRING = os.getenv("CONN_STRING2")
OAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = MongoClient(CONN_STRING)
db = client["ai_chatbot"]
collection = db["data"]

model = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=model, api_key=OAI_API_KEY) #OllamaEmbeddings(model="nomic-embed-text") # 

def check_mongodb_connection(client: MongoClient):
    """
    Checks if the MongoDB connection is successful
    """

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

def index_pdf_contents(pdf_folder_path):
    """
    Returns the text chunks from the pdf files in the folder
    """
    try:
        for file in os.listdir(pdf_folder_path):
            pdf_file_path = os.path.join(pdf_folder_path, file)
            pdf_loader = PyPDFLoader(pdf_file_path)
            pdf_documents = []
            for i, page in enumerate(pdf_loader.lazy_load()):
                if len(page.page_content) == 0:
                    continue
                document = {
                    "id": str(uuid4()),
                    "chunk_number": i,
                    "timestamp": datetime.now(),
                    "text": page.page_content,
                    "source": str(file),
                    "vector_embeddings": embeddings.embed_documents(page.page_content)[0]
                }
                # pdf_documents.append(document)
                collection.insert_one(document)
        return True
    except Exception as e:
        print(f"Failed to index PDF contents: {e}")
        return False

def index_web_contents(urls_json_file_path):
    """
    Returns the text chunks from the web documents
    """
    try:
        with open(urls_json_file_path, "r") as file:
            urls_refs = json.load(file)
    except FileNotFoundError:
        raise Exception(f"File not found at: {urls_json_file_path}")

    urls = urls_refs.get("energy_saving_resources", [])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            data = loader.load()
            split_docs = text_splitter.split_documents(data)
            for i, doc in enumerate(split_docs):
                document = {
                    "id": str(uuid4()),
                    "text": doc.page_content,
                    "chunk_number": i,
                    "source": str(url),
                    "timestamp": datetime.now(),
                    "vector_embeddings": embeddings.embed_documents(doc.page_content)[0]
                }
                collection.insert_one(document)
            return True
        except Exception as e:
            print(f"Failed to load URL {url}: {e}")
            return False


# Define a function to run vector search queries
def get_query_results(query):
    """Gets results from a vector search query."""

    query_embedding = embeddings.embed_documents(query)[0]
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "vector_embeddings",
                "exact": True,
                "limit": 10
            }
        },
        {
            "$project": {
                "_id": 0,
                "vector_embeddings": 0,
                "timestamp": 0
            }
        }
    ]

    results = collection.aggregate(pipeline)

    array_of_results = []
    for doc in results:
        array_of_results.append(doc)
    return array_of_results

def setup_mongodb_vector_search():
    """
    Sets up the MongoDB collection for the retrieval system
    """

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "vector_embeddings",
                    "numDimensions": 1536,
                    "similarity": "cosine"
                }
            ]
        },
        name="vector_index",
        type="vectorSearch"
        )

    result = collection.create_search_index(model=search_index_model)

    print("New search index named " + result + " is building.")
    # Wait for initial sync to complete
    print("Polling to check if the index is ready. This may take up to a minute.")
    predicate=None
    if predicate is None:
        predicate = lambda index: index.get("queryable") is True
    start_time = time.time()
    while True:
        indices = list(collection.list_search_indexes(result))
        if len(indices) and predicate(indices[0]):
            break
        if time.time() - start_time > 70:
            print("Building search index process failed: Timeout after 1 minute.")
            return False
        time.sleep(5)
    print(result + " is ready for querying.")
    return True