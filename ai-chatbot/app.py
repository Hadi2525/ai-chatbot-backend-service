from fastapi import FastAPI, HTTPException
from uuid import uuid4
import json

from schema import Message, SummaryRequest, SaveRequest
from vector_data_store import lookup_contexts
from chain_config import get_graph
app = FastAPI()

sessions = {
    "000-000": {"messages": ["tell me something about energy saving"]}
}
database = {}


def save_to_database(session_id: str, data: dict):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        database[session_id] = data
    except:
        raise HTTPException(status_code=500, detail="An error occured while saving to database")


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
async def retrieve_contexts(session_id: str):
    """Retrieve contexts from the vector store."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    message_history = sessions[session_id]["messages"][0]
    
    retrieved_contexts = await lookup_contexts(message_history)
    
    return {"session_id": session_id, "contexts": retrieved_contexts, "message_history": message_history}

@app.post("/generate_summary")
async def generate_summary(request: SummaryRequest):
    """Generate a summary based on retrieved contexts and message history."""
    # Simulate calling OpenAI API or another language model
    if request.session_id not in sessions:
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
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save the session's data to a mock database
    save_to_database(request.session_id, {
        "messages": sessions[request.session_id]["messages"],
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