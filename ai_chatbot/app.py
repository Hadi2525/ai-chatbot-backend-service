from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from uuid import uuid4
import json

from pymongo import MongoClient
from schema import Message, SummaryRequest, SaveRequest
from ai_chatbot.chain_config import get_graph
from ai_chatbot.collection_config import get_query_results

from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv(),override=True)
CONN_STRING = os.getenv("CONN_STRING2")

app = FastAPI()
client = MongoClient(CONN_STRING)
db = client["ai_chatbot"]
chat_collection = db["chat_history"]

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME)


VALID_API_KEYS = {"full-stack-ai-lab",
                  "secret-key",
                  "admin-key"}

def validate_api_key(api_key: str = Depends(api_key_header)):
    """Validate the API key."""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


def save_to_database(session_id: str, data: dict):
    """Save the session data to the database."""

    if not chat_collection.find_one({"session_id": session_id}):
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        chat_collection.update_one({"session_id": session_id}, {"$set": data})
    except:
        raise HTTPException(status_code=500, detail="An error occured while saving to database")


# Endpoints
@app.post("/get_session_id")
def get_session_id():
    """Generate a new session ID."""
    session_id = str(uuid4())
    try:
        chat_collection.insert_one({"session_id": session_id,
                                    "message_history": []})
    except:
        raise HTTPException(status_code=500, detail="An error occured while saving to database")
    return {"session_id": session_id}

@app.post("/ask")
def ask(session_id: str, message: Message):
    """Handle user questions."""
    if not chat_collection.find_one({"session_id": session_id}):
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        chat_history = chat_collection.find_one({"session_id": session_id})["message_history"]
        chat_history.append(message.message)
        chat_collection.update_one({"session_id": session_id}, {"$set": {"message_history": chat_history}})
    except:
        raise HTTPException(status_code=500, detail="An error occured while saving to database")
    return {"message": "Message received", "session_id": session_id}

@app.post("/retrieve_contexts")
async def retrieve_contexts(message: str):
    """Retrieve contexts from the vector store."""
    
    retrieved_contexts = await get_query_results(message)
    
    return { "contexts": retrieved_contexts, "message_history": message}

@app.post("/generate_summary")
async def generate_summary(request: SummaryRequest,
                           api_key: str = Depends(validate_api_key)):
    """Generate a summary based on retrieved contexts and message history."""
    # Simulate calling OpenAI API or another language model
    if not chat_collection.find_one({"session_id": request.session_id}):
        raise HTTPException(status_code=404, detail="Session not found")
    
    if len(request.message_history) == 0:
        raise HTTPException(status_code=400, detail="Message history is empty")
    
    question = request.message_history[0]
    graph = get_graph()
    response = await graph.ainvoke({"question":question})
    contexts_dict = [doc.dict() for doc in response.get("context")]
    
    return {"session_id": request.session_id, 
            "summary": json.dumps(response.get('answer')), 
            "retrieved_contexts": contexts_dict,
            "question": question
            }

@app.post("/save_records")
def save_records(request: SaveRequest):
    """Save session summary in the database."""
    if not chat_collection.find_one({"session_id": request.session_id}):
        raise HTTPException(status_code=404, detail="Session not found")
    
    message_history = chat_collection.find_one({"session_id": request.session_id})["message_history"]    
    save_to_database(request.session_id, {
        "messages": message_history,
        "summary": request.summary
    })
    return {"message": "Session data saved", "session_id": request.session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  
        host="0.0.0.0",  
        port=8000,
    )