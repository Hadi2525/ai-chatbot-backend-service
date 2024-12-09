from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(find_dotenv(),override=True)

CONN_STRING = os.getenv("CONN_STRING")

client = MongoClient(CONN_STRING, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Connected to MongoDB")
except Exception as e:
    print(e)

database = client['ai-chatbot']
collection = database['data']

