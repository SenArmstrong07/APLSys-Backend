from pydantic import BaseModel
from typing import Optional

#AI Requests
class ResumeAnalysisRequest(BaseModel):
    resume: str
    job_role: Optional[str] = None
    job_description: Optional[str] = None

class PromptRequest(BaseModel):
    prompt: str
    
class ClassifyRequest(BaseModel):
    text: str
    
class ResumeTextRequest(BaseModel):
    text: str
    
#PARSING Requests    
class DocumentParseRequest(BaseModel):
    text: str
    
class ResumeParseRequest(BaseModel):
    text: str
    
#UTILITY Requests    
class SearchRequest(BaseModel):
    file_path: str
    query: str
    
class MapEntitiesRequest(BaseModel):
    entities: list
    full_text: str = ""
    
class TextRequest(BaseModel):
    text: str


    


