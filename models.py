from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
from datetime import datetime

class ProcessedDoc(BaseModel):
    name: str
    url: str
    md5: str
    phash_hex: str
    was_relevant: bool
    low_confidence: bool = False

class RegistrationDetails(BaseModel):
    doc_id: Optional[str] = Field(None, description="Source filename / document identifier")
    registration_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    promised_completion_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    classification_rounds: Optional[int] = None
    classification_unanimous: Optional[bool] = None
    extraction_rounds: Optional[int] = None
    extraction_unanimous: Optional[bool] = None

class ExtensionDetail(BaseModel):
    doc_id: Optional[str] = Field(None, description="Source filename / document identifier")
    extension_approval_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    revised_promised_completion_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    classification_rounds: Optional[int] = None
    classification_unanimous: Optional[bool] = None
    extraction_rounds: Optional[int] = None
    extraction_unanimous: Optional[bool] = None

class OCCCDetail(BaseModel):
    doc_id: str
    issue_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    type: Optional[str] = Field(None, description="Occupation / Completion")
    details: Optional[Union[str, List[str]]] = Field(None, description="Scope/Tower details")
    classification_rounds: Optional[int] = None
    classification_unanimous: Optional[bool] = None
    extraction_rounds: Optional[int] = None
    extraction_unanimous: Optional[bool] = None

    @field_validator('details', mode='before')
    @classmethod
    def ensure_string_details(cls, v):
        if isinstance(v, list):
            return ", ".join(map(str, v))
        return v

class RunMetrics(BaseModel):
    elapsed_seconds: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    generation_tps: Optional[float] = None
    pipeline_tps: Optional[float] = None
    gemini_input_tokens: Optional[int] = None
    gemini_output_tokens: Optional[int] = None

class ProjectData(BaseModel):
    rera_id: str
    registration_details: Optional[RegistrationDetails] = None
    extension_details: List[ExtensionDetail] = []
    oc_cc_documents: List[OCCCDetail] = []
    processed_docs: List[ProcessedDoc] = []
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    metrics: Optional[RunMetrics] = None

