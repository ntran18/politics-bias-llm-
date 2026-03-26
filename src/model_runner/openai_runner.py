import json
import os
import sys
import tempfile
import time

from openai import BadRequestError, OpenAI

from .base import BaseBiasRunner
from .models import LLM_RESULT_COLUMNS, PoliticalBiasAssessment
from .utils import extract_text_content


class OpenAIBatchBiasRunner(BaseBiasRunner):
    MAX_REQUESTS_PER_BATCH = 500

    def __init__(
        self,
        batch_poll_interval: int = 30,
        openai_api_key: str | None = None,
        **kwargs,
    ):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIBatchBiasRunner")
        super().__init__(workers=1, **kwargs)

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI batch mode")

        self._client = OpenAI(api_key=api_key)
        self.batch_poll_interval = batch_poll_interval

    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        raise NotImplementedError("OpenAI batch runner does not use per-row async requests")

    def _build_batch_jsonl(self, df_to_process, jsonl_path: str) -> dict[str, dict]:
        custom_id_to_row: dict[str, dict] = {}

        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for sequence_number, (_, row) in enumerate(df_to_process.iterrows()):
                row_data = row.drop("prompt").to_dict()
                custom_id = f"{row_data['article_id']}::{row_data['index']}::{sequence_number}"
                custom_id_to_row[custom_id] = row_data

                body = {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "political_bias_assessment",
                            "schema": PoliticalBiasAssessment.model_json_schema(),
                        },
                    },
                }
                request_line = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                handle.write(json.dumps(request_line, ensure_ascii=False) + "\n")

        return custom_id_to_row

    def _poll_batch(self, batch_id: str):
        running_states = {"validating", "in_progress", "finalizing"}
        while True:
            batch = self._client.batches.retrieve(batch_id)
            print(f"Batch {batch.id} status: {batch.status}")
            if batch.status in running_states:
                time.sleep(self.batch_poll_interval)
                continue
            return batch

    def _download_file_text(self, file_id: str) -> str:
        content = self._client.files.content(file_id)
        if hasattr(content, "text"):
            return content.text
        if hasattr(content, "read"):
            return content.read().decode("utf-8")
        return str(content)

    def _parse_batch_output_line(
        self,
        line: str,
        custom_id_to_row: dict[str, dict],
    ) -> dict | None:
        if not line.strip():
            return None

        parsed_line = json.loads(line)
        custom_id = parsed_line.get("custom_id")
        row_data = custom_id_to_row.get(custom_id)
        if row_data is None:
            return None

        llm_data = None
        llm_error = None
        try:
            response = parsed_line.get("response", {})
            if response.get("status_code") != 200:
                raise ValueError(f"status_code={response.get('status_code')}")

            body = response.get("body", {})
            choices = body.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            content = extract_text_content(message.get("content", ""))
            llm_data = PoliticalBiasAssessment.model_validate_json(content)
        except Exception as exc:
            llm_error = str(exc)
            print(f"[Warning] Failed to parse batch item {custom_id}: {exc}", file=sys.stderr)

        return {
            "index": row_data.get("index"),
            "row_data": row_data,
            "llm_data": llm_data,
            "llm_model": self.model_name,
            "llm_error": llm_error,
        }

    def _process_single_file(self, input_file_path: str) -> None:
        print(f"\n--- Processing File (OpenAI Batch): {os.path.basename(input_file_path)} ---")

        df = self._load_data(input_file_path)
        if df is None:
            print(f"Skipping {os.path.basename(input_file_path)}.")
            return

        output_path = self._setup_output_file(input_file_path)
        original_columns = [column for column in df.columns.tolist() if column != "prompt"]
        all_columns = original_columns + LLM_RESULT_COLUMNS

        processed_keys = self._initialize_output_file(output_path, all_columns)
        mask = df.apply(
            lambda row: (row["article_id"], row["index"]) not in processed_keys,
            axis=1,
        )
        df_to_process = df[mask]

        print(f"Total prompts in file: {len(df)}")
        print(f"Already processed: {len(processed_keys)}")
        print(f"Remaining to process: {len(df_to_process)}")

        if len(df_to_process) == 0:
            print("All prompts already processed. Skipping file.")
            return

        total_remaining = len(df_to_process)
        chunk_size = self.MAX_REQUESTS_PER_BATCH
        total_chunks = (total_remaining + chunk_size - 1) // chunk_size

        for chunk_index in range(total_chunks):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, total_remaining)
            chunk_df = df_to_process.iloc[start:end]
            print(
                f"Submitting OpenAI batch chunk {chunk_index + 1}/{total_chunks} "
                f"with {len(chunk_df)} prompts"
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_jsonl:
                jsonl_path = temp_jsonl.name

            custom_id_to_row = self._build_batch_jsonl(chunk_df, jsonl_path)

            with open(jsonl_path, "rb") as handle:
                input_file = self._client.files.create(file=handle, purpose="batch")

            try:
                batch = self._client.batches.create(
                    input_file_id=input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "model": self.model_name,
                        "version": self.version,
                        "source_file": os.path.basename(input_file_path),
                        "chunk_index": str(chunk_index + 1),
                        "chunk_size": str(len(chunk_df)),
                    },
                )
            except BadRequestError as exc:
                error_text = str(exc)
                if "billing_hard_limit_reached" in error_text:
                    print(
                        "[Error] OpenAI billing hard limit reached for this API key/account. "
                        "Increase billing limit, add funds, or use a different key/account.",
                        file=sys.stderr,
                    )
                elif "Enqueued token limit reached" in error_text:
                    print(
                        "[Error] OpenAI Batch enqueued token limit reached for the organization. "
                        "Wait for in-progress batches to complete, cancel stale in-progress batches, "
                        "or use another organization/key with available batch capacity.",
                        file=sys.stderr,
                    )
                else:
                    print(f"[Error] OpenAI batch creation failed: {exc}", file=sys.stderr)
                return
            except Exception as exc:
                print(f"[Error] Unexpected error creating OpenAI batch: {exc}", file=sys.stderr)
                return

            print(f"Created batch job: {batch.id}")
            batch = self._poll_batch(batch.id)

            if batch.status != "completed" or not getattr(batch, "output_file_id", None):
                print(
                    f"[Error] Batch did not complete successfully. "
                    f"status={batch.status}, error_file_id={getattr(batch, 'error_file_id', None)}"
                )
                return

            output_text = self._download_file_text(batch.output_file_id)

            results_buffer = []
            for line in output_text.splitlines():
                parsed_result = self._parse_batch_output_line(line, custom_id_to_row)
                if parsed_result is None:
                    continue

                results_buffer.append(parsed_result)

                if len(results_buffer) >= self.checkpoint_size:
                    self._save_results_buffer(results_buffer, all_columns, output_path)
                    results_buffer = []

            if results_buffer:
                self._save_results_buffer(results_buffer, all_columns, output_path)

        print(
            f"--- Finished processing {os.path.basename(input_file_path)}. "
            f"Results written to: {output_path} ---"
        )
