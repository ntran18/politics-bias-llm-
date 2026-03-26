import sys

from pydantic import ValidationError

from base import BaseBiasRunner
from models import OLLAMA_DEFAULT_PORT, PoliticalBiasAssessment

try:
    from ollama import AsyncClient as OllamaAsyncClient
    from ollama import ResponseError as OllamaResponseError
except ImportError:
    OllamaAsyncClient = None
    OllamaResponseError = Exception


class OllamaBiasRunner(BaseBiasRunner):
    def __init__(self, ollama_port: int = OLLAMA_DEFAULT_PORT, **kwargs):
        if OllamaAsyncClient is None:
            raise ImportError("ollama package is required for OllamaBiasRunner")
        super().__init__(**kwargs)
        self.ollama_port = ollama_port
        self._client = OllamaAsyncClient(host=f"http://localhost:{self.ollama_port}")

    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        llm_data = None
        llm_error = None
        try:
            response = await self._client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": self.temperature, "num_ctx": self.context_length},
                format=PoliticalBiasAssessment.model_json_schema(),
                keep_alive="5m",
            )

            response_text = response["message"]["content"].strip()
            llm_data = PoliticalBiasAssessment.model_validate_json(response_text)

        except ValidationError as exc:
            llm_error = str(exc)
            print(
                f"\n[Warning] Pydantic validation failed for row {index}. Error details: {exc}",
                file=sys.stderr,
            )
        except OllamaResponseError as exc:
            llm_error = str(exc)
            print(f"\n[Error] Ollama Response Error for row {index}: {exc}", file=sys.stderr)
        except Exception as exc:
            llm_error = str(exc)
            print(f"\n[Error] General Ollama inference error for row {index}: {exc}", file=sys.stderr)

        return {
            "index": index,
            "row_data": row_data,
            "llm_data": llm_data,
            "llm_model": self.model_name,
            "llm_error": llm_error,
        }
