from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict

class TicketClassification(BaseModel):
    topic_tags: List[str] = Field(default_factory=list)
    topic_tag_confidence: Dict[str, float] = Field(default_factory=dict)  
    core_problem: str = ""
    priority: str = ""
    sentiment: str = ""
