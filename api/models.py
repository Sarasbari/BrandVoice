from pydantic import BaseModel
from typing import Literal

class GenerateRequest(BaseModel):
    content_type: Literal["tweet", "linkedin", "blog_intro", "changelog", "how_to"]
    topic: str
    audience: str
    tone: str

class GenerateResponse(BaseModel):
    generated_text: str
    model_id: str
    content_type: str
