import asyncio
import os
import sys

from openai import OpenAI
from pydantic import ValidationError

from base import BaseBiasRunner


class OpenAIDirectBiasRunner(BaseBiasRunner):
    def __init__(
        self,
        openai_api_key: str | None = None,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        **kwargs,
    ):
        super().__init__(**kwargs)

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI direct mode")

        self._client = OpenAI(api_key=api_key)
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary

    def _extract_native_cot(self, response) -> str | None:
        summaries: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "reasoning":
                continue
            for s in (getattr(item, "summary", None) or []):
                if getattr(s, "type", None) == "summary_text" and getattr(s, "text", None):
                    summaries.append(s.text)
        return "\n\n".join(summaries) if summaries else None

    def _sync_infer(self, prompt: str):
        request_kwargs = {
            "model": self.model_name,
            "input": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "text_format": self.assessment_model,
        }

        # Native CoT is optional and only requested when enabled.
        if self.include_native_cot:
            request_kwargs["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary,
            }

        # GPT-5 family may ignore/forbid temperature in some configs.
        if "gpt-5" not in self.model_name.lower():
            request_kwargs["temperature"] = self.temperature

        return self._client.responses.parse(**request_kwargs)

    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        llm_data = None
        llm_error = None
        llm_native_cot = None

        try:
            response = await asyncio.to_thread(self._sync_infer, prompt)
            llm_data = response.output_parsed
            if self.include_native_cot:
                llm_native_cot = self._extract_native_cot(response)

        except ValidationError as exc:
            llm_error = str(exc)
            print(f"\n[Warning] Pydantic validation failed for row {index}: {exc}", file=sys.stderr)
        except Exception as exc:
            llm_error = str(exc)
            print(f"\n[Error] OpenAI direct inference error for row {index}: {exc}", file=sys.stderr)

        return {
            "index": index,
            "row_data": row_data,
            "llm_data": llm_data,
            "llm_model": self.model_name,
            "llm_error": llm_error,
            "llm_native_cot": llm_native_cot,
        }