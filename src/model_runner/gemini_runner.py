import asyncio
import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tenacity import retry, stop_after_attempt, wait_random_exponential

from base import BaseBiasRunner
from models import PoliticalBiasAssessment

load_dotenv()
class GeminiBiasRunner(BaseBiasRunner):
    def __init__(self, gemini_api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required")

        # Initialize the new Client
        self._client = genai.Client(api_key=api_key)

    @retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(10))
    def _sync_generate(self, prompt: str):
        schema = PoliticalBiasAssessment.model_json_schema()

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_json_schema=schema,
        )
        
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response

    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        llm_data = None
        llm_error = None
        try:
            response = await asyncio.to_thread(self._sync_generate, prompt)
            
            if response.text:
                llm_data = PoliticalBiasAssessment.model_validate_json(response.text)
            
        except Exception as exc:
            llm_error = str(exc)
            print(f"\n[Error] Gemini row {index}: {exc}", file=sys.stderr)

        return {
            "index": index,
            "row_data": row_data,
            "llm_data": llm_data,
            "llm_model": self.model_name,
            "llm_error": llm_error,
        }
