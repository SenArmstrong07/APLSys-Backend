from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter()

class ResumeParseRequest(BaseModel):
    text: str

# @router.post("/label-tokens")
# async def label_tokens_resume(request: ResumeParseRequest, fastapi_req: Request):
#     ner_resume_pipeline = fastapi_req.app.state.ner_resume_pipeline
#     return lbl_resume_text(request.text, ner_resume_pipeline)
