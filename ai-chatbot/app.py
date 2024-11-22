from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from uuid import uuid4
from typing import List, Dict, Optional
import json

app = FastAPI()

sessions = {}
database = {}

# Models
class Message(BaseModel):
    message: str

class SessionData(BaseModel):
    session_id: str
    messages: List[str]

class SummaryRequest(BaseModel):
    session_id: str
    contexts: List[str]
    message_history: List[str]

class SaveRequest(BaseModel):
    session_id: str
    summary: str

# Helper function to simulate a database operation
def save_to_database(session_id: str, data: dict):
    database[session_id] = data

# Endpoints
@app.post("/get_session_id")
def get_session_id():
    """Generate a new session ID."""
    session_id = str(uuid4())
    sessions[session_id] = {"messages": []}
    return {"session_id": session_id}

@app.post("/ask")
def ask(session_id: str, message: Message):
    """Handle user questions."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["messages"].append(message.message)
    return {"message": "Message received", "session_id": session_id}

@app.post("/retrieve_contexts")
def retrieve_contexts(session_id: str):
    """Retrieve contexts from the vector store."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    message_history = sessions[session_id]["messages"]
    
    # Simulate retrieving contexts (replace with actual vector store logic)
    retrieved_contexts = ["Context 1", "Context 2"]
    
    return {"session_id": session_id, "contexts": retrieved_contexts, "message_history": message_history}

@app.post("/generate_summary")
def generate_summary(request: SummaryRequest):
    """Generate a summary based on retrieved contexts and message history."""
    # Simulate calling OpenAI API or another language model
    summary = f"Summary based on: {json.dumps(request.contexts)} and {json.dumps(request.message_history)}"
    
    return {"session_id": request.session_id, "summary": summary}

@app.post("/save_records")
def save_records(request: SaveRequest):
    """Save session summary in the database."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save the session's data to a mock database
    save_to_database(request.session_id, {
        "messages": sessions[request.session_id]["messages"],
        "summary": request.summary
    })
    return {"message": "Session data saved", "session_id": request.session_id}