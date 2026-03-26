import asyncio
import os
import sys

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import BaseBiasRunner
from .models import PoliticalBiasAssessment


class GeminiBiasRunner(BaseBiasRunner):
    def __init__(self, gemini_api_key: str | None = None, **kwargs):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package is required for GeminiBiasRunner"
            ) from exc

        super().__init__(**kwargs)

        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini")

        self._model = genai.Client()

    @retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(10))
    def _sync_generate(self, prompt: str) -> str:
        config = {
            "temperature": self.temperature,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
            "response_schema": PoliticalBiasAssessment,
        }
        try:
            response = self._model.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        except TypeError:
            config.pop("response_mime_type", None)
            response = self._model.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

        text = getattr(response, "text", "")
        return text.strip()

    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        llm_data = None
        llm_error = None
        try:
            response_text = await asyncio.to_thread(self._sync_generate, prompt)
            llm_data = response_text.parsed
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
