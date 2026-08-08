from pydantic import BaseModel
from typing import List


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float