"""

"""

from typing import Dict, List, Optional

from pydantic import BaseModel


# Models
class Message(BaseModel):
    message: str


class SessionData(BaseModel):
    session_id: str
    messages: List[str]


class SummaryRequest(BaseModel):
    session_id: str
    message_history: List[str]


class SaveRequest(BaseModel):
    session_id: str
    message_history: List[str]
    contexts: List[str]
    summary: str
