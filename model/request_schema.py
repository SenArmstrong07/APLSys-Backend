from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

#AI Requests
class ResumeAnalysisRequest(BaseModel):
    resume: str
    job_role: Optional[str] = Field(None, alias="not")
    job_description: Optional[str] = Field(None, alias="not")

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
    
class TaskCreateRequest(BaseModel):
    task_type: str
    filename: Optional[str] = Field(None, alias="not")
    details: Optional[Dict[str, Any]] = Field(None, alias="not")
    
class TaskCreateBatchRequest(BaseModel):
    task_type: str
    files: Optional[List[str]] = Field(None, alias="not")
    filename: Optional[str] = Field(None, alias="not")
    details: Optional[Dict[str, Any]] = Field(None, alias="not")