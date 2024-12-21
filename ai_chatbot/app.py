import json
import os
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pymongo import MongoClient
from ai_chatbot.schema import Message, SaveRequest, SummaryRequest

from ai_chatbot.chain_config import get_graph
from ai_chatbot.collection_config import get_query_results

# Load environment variables
load_dotenv(find_dotenv(), override=True)
CONN_STRING = os.getenv("CONN_STRING2")
if not CONN_STRING:
    raise RuntimeError("Database connection string (CONN_STRING2) is not set.")

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection
try:
    client = MongoClient(CONN_STRING)
    db = client["ai_chatbot"]
    chat_collection = db["chat_history"]
except Exception as e:
    raise RuntimeError(f"Failed to connect to MongoDB: {e}")

# API Key authentication
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# Set of valid API keys
VALID_API_KEYS = {"full-stack-ai-lab", "secret-key", "admin-key"}


def validate_api_key(api_key: str = Depends(api_key_header)):
    """Validate the API key."""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


def save_to_database(session_id: str, data: dict):
    """Save the session data to the database."""
    try:
        result = chat_collection.update_one({"session_id": session_id}, {"$set": data})
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


# Endpoints
@app.post("/get_session_id")
def get_session_id():
    """Generate a new session ID."""
    session_id = str(uuid4())
    try:
        chat_collection.insert_one({"session_id": session_id, "message_history": []})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    return {"session_id": session_id}


@app.post("/ask")
def ask(session_id: str, message: Message):
    """Handle user questions."""
    try:
        chat = chat_collection.find_one({"session_id": session_id})
        if not chat:
            raise HTTPException(status_code=404, detail="Session not found")

        chat_history = chat["message_history"]
        chat_history.append({"message": message.message, "role": "user"})

        chat_collection.update_one(
            {"session_id": session_id}, {"$set": {"message_history": chat_history}}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    return {"message": "Message received", "session_id": session_id}


@app.post("/retrieve_contexts")
def retrieve_contexts(message: str):
    """Retrieve contexts from the vector store."""
    try:
        retrieved_contexts = get_query_results(message)
        return {"contexts": retrieved_contexts, "message_history": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context retrieval error: {e}")


@app.post("/generate_summary")
async def generate_summary(
    request: SummaryRequest, api_key: str = Depends(validate_api_key)
):
    """Generate a summary based on retrieved contexts and message history."""
    try:
        chat = chat_collection.find_one({"session_id": request.session_id})
        if not chat:
            raise HTTPException(status_code=404, detail="Session not found")

        if not request.message_history:
            raise HTTPException(status_code=400, detail="Message history is empty")

        question = request.message_history[0]
        graph = get_graph()
        response = await graph.ainvoke({"question": question})
        contexts_dict = [doc.dict() for doc in response.get("context")]

        return {
            "session_id": request.session_id,
            "summary": json.dumps(response.get("answer")),
            "retrieved_contexts": contexts_dict,
            "question": question,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation error: {e}")


@app.post("/get_session_history")
def get_session_history(session_id: str):
    """Retrieve the session history."""
    try:
        chat = chat_collection.find_one({"session_id": session_id})
        if not chat:
            raise HTTPException(status_code=404, detail="Session not found")

        message_history = chat["message_history"]
        return {"message_history": message_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.post("/save_record")
def save_record(request: SaveRequest):
    """Save session summary in the database."""
    try:
        chat = chat_collection.find_one({"session_id": request.session_id})
        if not chat:
            raise HTTPException(status_code=404, detail="Session not found")

        message_history = chat["message_history"]
        save_to_database(
            request.session_id, {"messages": message_history, "summary": request.summary}
        )
        return {"message": "Session data saved", "session_id": request.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
    )
