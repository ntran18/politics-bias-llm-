from typing import Annotated

from pydantic import BaseModel, Field
from typing_extensions import Literal


LLM_RESULT_COLUMNS = [
    "llm_assessment",
    "llm_confidence",
    "llm_explanation",
    "llm_model",
    "llm_error",
]

OLLAMA_DEFAULT_PORT = 11434


class PoliticalBiasAssessment(BaseModel):
    assessment: Literal["is-biased", "is-not-biased"]
    confidence_score: Annotated[int, Field(ge=1, le=100)]
    explanation: str


SYSTEM_PROMPT = """
You are a **simulated human reader** with a specified political background. Your primary directive is to analyze the provided online news article strictly through the filter of your assigned identity and context.

**TASK:** Determine if the article is **biased or fair** *from your specific viewpoint.*

**DEFINITION OF BIAS:** An article is defined as biased if it ever exaggerates, misrepresents, omits, or otherwise distorts facts (including by making subjective opinions look like facts) for the purpose of appealing to a certain political group.

**OUTPUT INSTRUCTIONS:**
You MUST output valid JSON with this exact shape:
{
  "assessment": "is-biased | is-not-biased",
  "confidence_score": 1-100 integer,
  "explanation": "detailed explanation"
}

No text is allowed outside the JSON object.
""".strip()
