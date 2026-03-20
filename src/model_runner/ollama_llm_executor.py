from typing_extensions import Literal
import pandas as pd
import argparse
import asyncio
import os
from pydantic import BaseModel, ValidationError, conint
from tqdm import tqdm
from ollama import AsyncClient, ResponseError as OllamaResponseError
from prompt_generation.constants import Constants
import sys

# LLM columns to be added to the output file
LLM_RESULT_COLUMNS = [
    'llm_assessment',
    'llm_confidence',
    'llm_explanation',
    'llm_model'
]

OLLAMA_DEFAULT_PORT = 11434

class PoliticalBiasAssessment(BaseModel):
    """Defines the structure for the LLM's output."""
    assessment: Literal["is-biased", "is-not-biased"]
    confidence_score: conint(ge=1, le=100)
    explanation: str


SYSTEM_PROMPT = """
    You are a **simulated human reader** with a specified political background. Your primary directive is to analyze the provided online news article strictly through the filter of your assigned identity and context.

    **TASK:** Determine if the article is **biased or fair** *from your specific viewpoint.*

    **DEFINITION OF BIAS:** An article is defined as biased if it ever exaggerates, misrepresents, omits, or otherwise distorts facts (including by making subjective opinions look like facts) for the purpose of appealing to a certain political group.

    **OUTPUT INSTRUCTIONS:**
    You MUST output using the following format EXACTLY:

    <json>
    {
        "assessment": "...",
        "confidence_score": ...,
        "explanation": "..."
    }
    </json>

    No text is allowed outside the <json> block.

    **JSON SCHEMA:** The JSON object MUST contain the following three keys:
    1. **'assessment'**: (Value must be one of two strings: 'is-biased' or 'is-not-biased').
    2. **'confidence_score'**: (Value must be an integer from 1 to 100). 1 indicates not confident at all; 100 indicates absolute certainty.
    3. **'explanation'**: (Value must be a detailed string). Explain *how* the article's tone, content, or framing impacts your assessment, referencing specific parts of the article. Structure the explanation with bullet points.
"""


class OllamaBiasRunner:
    """
    Manages the batch execution of LLM inference using the Ollama API for political bias assessment.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        version: str,
        workers: int,
        checkpoint_size: int,
        temperature: float,
        ollama_port: int = OLLAMA_DEFAULT_PORT,
        context_length: int = 2048,
    ):
        """
        Initializes the runner with configuration parameters.
        """
        self.model_name = model_name
        self.output_dir = os.path.join(
            output_dir,
            version,
            model_name.replace('/', '_'),
            Constants.DEFAULT_LLM_OUTPUT_FOLDER,
        )
        self.workers = workers
        self.checkpoint_size = checkpoint_size
        self.temperature = temperature
        self.version = version
        self.system_prompt = SYSTEM_PROMPT
        self.json_schema = PoliticalBiasAssessment.model_json_schema()
        self.ollama_port = ollama_port
        self.context_length = context_length

    def _format_messages(self, system_prompt: str, user_query: str) -> list[dict]:
        """Formats the system and user input into the Ollama message list."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

    def _load_data(self, input_file: str) -> pd.DataFrame | None:
        """Loads the input CSV file containing prompts."""
        if not os.path.exists(input_file):
            print(f"Error: Input file not found at '{input_file}'.")
            return None

        data = pd.read_csv(input_file)

        if data.empty:
            print(f"Error: Input file '{input_file}' is empty.")
            return None

        if 'prompt' not in data.columns:
            print(f"Error: Input CSV must contain a 'prompt' column.")
            return None

        # Sanity check: we expect article_id and index columns for resume logic
        missing_required = [col for col in ('article_id', 'index') if col not in data.columns]
        if missing_required:
            print(
                f"Error: Input CSV must contain the columns {missing_required} "
                f"for resume functionality."
            )
            return None

        return data

    def _save_results_buffer(self, results_buffer, all_columns, output_path: str) -> None:
        """Saves the results buffer to the output CSV file."""
        final_rows = []
        for result in results_buffer:
            if result['llm_data'] is None:
                # Skip rows where LLM failed / validation failed.
                continue

            # row_data is a per-row dict created from .to_dict(), safe to mutate here.
            final_row = dict(result['row_data'])
            final_row.update({
                'llm_assessment': result['llm_data'].assessment,
                'llm_confidence': result['llm_data'].confidence_score,
                'llm_explanation': result['llm_data'].explanation,
                'llm_model': result['llm_model'],
            })
            final_rows.append(final_row)

        if not final_rows:
            return  # nothing to append

        pd.DataFrame(final_rows, columns=all_columns).to_csv(
            output_path,
            index=False,
            mode='a',
            header=False,
        )

    def _setup_output_file(self, input_file_path: str) -> str:
        """Prepares the output directory and file path for incremental saving."""
        input_filename = os.path.basename(input_file_path).replace('.csv', '')
        output_filename = f"{input_filename}.csv"
        output_path = os.path.join(self.output_dir, output_filename)

        os.makedirs(self.output_dir, exist_ok=True)
        return output_path

    def _initialize_output_file(self, output_path: str, columns: list[str]) -> set[tuple]:
        """
        Initialize output CSV (if needed) and return a set of processed keys.

        Keys are (article_id, index) pairs, taken from the existing output file.
        """
        write_header = not os.path.exists(output_path)
        processed_keys: set[tuple] = set()

        if write_header:
            pd.DataFrame(columns=columns).to_csv(output_path, index=False, mode='w')
            print(f"Initialized output file: {output_path}")
        else:
            try:
                processed_df = pd.read_csv(output_path, usecols=['article_id', 'index'])
                processed_keys = set(
                    zip(processed_df['article_id'], processed_df['index'])
                )
                print(
                    f"Appending to existing file: {output_path} "
                    f"({len(processed_keys)} rows already processed)"
                )
            except Exception as e:
                print(
                    f"[Warning] Could not load existing output ({e}). "
                    f"Starting from scratch for {output_path}"
                )
                processed_keys = set()

        return processed_keys

    async def _fetch_llm_response(self, client: AsyncClient, messages, row_data, index):
        """Sends messages to Ollama asynchronously and returns results."""
        llm_data = None

        try:
            response = await client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.context_length,
                },
                format=self.json_schema,
                keep_alive='5m',
            )

            response_text = response['message']['content'].strip()
            llm_data = PoliticalBiasAssessment.model_validate_json(response_text)

        except ValidationError as e:
            print(
                f"\n[Warning] Pydantic validation failed for row {index}. "
                f"Error details: {e}",
                file=sys.stderr,
            )
        except OllamaResponseError as e:
            print(
                f"\n[Error] Ollama Response Error for row {index}: {e}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"\n[Error] General inference error for row {index}: {e}",
                file=sys.stderr,
            )

        return {
            'index': index,
            'row_data': row_data,
            'llm_data': llm_data,
            'llm_model': self.model_name,
        }

    async def _process_one_experiment(
        self,
        df_to_process: pd.DataFrame,
        all_columns: list[str],
        output_path: str,
        initial_count: int = 0,
    ) -> None:
        """Manages the concurrent asynchronous processing loop."""
        client = AsyncClient(host=f'http://localhost:{self.ollama_port}')
        semaphore = asyncio.Semaphore(self.workers)

        tasks = []
        for index, row in df_to_process.iterrows():
            user_query = row['prompt']
            messages = self._format_messages(self.system_prompt, user_query)
            row_data = row.drop('prompt').to_dict()

            async def limited_fetch(idx, data, messages=messages):
                async with semaphore:
                    return await self._fetch_llm_response(client, messages, data, idx)

            tasks.append(limited_fetch(index, row_data))

        print(
            f"Starting async inference with {self.workers} workers and "
            f"checkpoint size: {self.checkpoint_size}..."
        )

        results_buffer = []
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(df_to_process),
            initial=initial_count,
            desc="Async Inference",
        ):
            try:
                result_data = await future
                results_buffer.append(result_data)

                if len(results_buffer) >= self.checkpoint_size:
                    print(
                        f"\n[Checkpoint] Saving {len(results_buffer)} "
                        f"results to disk..."
                    )
                    self._save_results_buffer(
                        results_buffer,
                        all_columns,
                        output_path,
                    )
                    results_buffer = []
            except Exception as e:
                print(
                    f"\n[Fatal Checkpoint Error] Could not process or save result: {e}"
                )
                raise

        if results_buffer:
            print(
                f"\n[Final Save] Saving remaining {len(results_buffer)} results to disk..."
            )
            self._save_results_buffer(results_buffer, all_columns, output_path)
            print("Final save complete.")

    def _process_single_file(self, input_file_path: str) -> None:
        """Handles the end-to-end processing for a single input CSV file."""

        print(f"\n--- Processing File: {os.path.basename(input_file_path)} ---")

        df = self._load_data(input_file_path)
        if df is None:
            print(f"Skipping {os.path.basename(input_file_path)}.")
            return

        output_path = self._setup_output_file(input_file_path)

        original_columns = df.columns.tolist()
        if 'prompt' in original_columns:
            original_columns.remove('prompt')

        all_columns = original_columns + LLM_RESULT_COLUMNS

        processed_keys = self._initialize_output_file(output_path, all_columns)

        def not_processed(row):
            return (row['article_id'], row['index']) not in processed_keys

        mask = df.apply(not_processed, axis=1)
        df_to_process = df[mask]

        total_prompts = len(df)
        print(f"Total prompts in file: {total_prompts}")
        print(f"Already processed: {len(processed_keys)}")
        print(f"Remaining to process: {len(df_to_process)}")

        if len(df_to_process) == 0:
            print("All prompts already processed. Skipping file.")
            return

        try:
            asyncio.run(
                self._process_one_experiment(
                    df_to_process=df_to_process,
                    all_columns=all_columns,
                    output_path=output_path,
                    initial_count=len(processed_keys),
                )
            )
        except Exception as e:
            print(f"\n[FATAL ERROR] Async processing failed: {e}")
            return

        print(
            f"--- Finished processing {os.path.basename(input_file_path)}. "
            f"Results written to: {output_path} ---"
        )

    def run_experiment(self, file_type: str, prompt_dir: str) -> None:
        """
        Main execution logic for running all experiments based on command-line arguments.
        """

        if file_type == 'all':
            files_to_run = list(Constants.PROMPT_FILE_MAP.values())
        else:
            files_to_run = [Constants.PROMPT_FILE_MAP[file_type]]

        print(f"Targeting prompt files in: {prompt_dir}")
        print(
            f"Using model: {self.model_name} "
            f"(Temperature: {self.temperature})"
            f" on port {self.ollama_port}"
            f" with {self.workers} workers"
            f" and context length {self.context_length}."
        )
        print(f"Output directory: {self.output_dir}")
        print(f"Files to process: {files_to_run}")
        
        prompt_version = self.version
        
        if '.' in prompt_version:
            prompt_version = prompt_version.split('.')[0]

        for file_name in files_to_run:
            input_file_path = os.path.join(prompt_dir, prompt_version, file_name)

            if not os.path.exists(input_file_path):
                print(f"\n[Skipping] Input file not found: {input_file_path}")
                continue

            self._process_single_file(input_file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch LLM inference over generated prompts using Ollama."
    )

    file_type_choices = ['all'] + list(Constants.PROMPT_FILE_MAP.keys())

    parser.add_argument(
        '--file-type',
        type=str,
        default='all',
        choices=file_type_choices,
        help=(
            "The type of prompt file(s) to process. "
            f"Choices: {', '.join(file_type_choices)}. Default: 'all'"
        ),
    )
    parser.add_argument(
        '--model',
        type=str,
        default=Constants.MODEL_NAME,
        help=(
            "Ollama model name to use for inference "
            "(e.g., 'mistral', 'llama3'). "
            f"Default: {Constants.MODEL_NAME}"
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=Constants.DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to save the resulting CSV file. "
            f"Default: {Constants.DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        '--version',
        type=str,
        default=Constants.VERSION,
        help=(
            "Version subdirectory within the prompt directory to use "
            "(e.g., 'v1', 'v2'). "
            f"Default: {Constants.VERSION}"
        ),
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help=(
            "Number of concurrent API calls to Ollama "
            "(Ollama's --parallel setting). Default: 8"
        ),
    )
    parser.add_argument(
        '--checkpoint-size',
        type=int,
        default=100,
        help=(
            "Number of completed prompts to collect before saving a checkpoint "
            "to the output file. Default: 100"
        ),
    )
    parser.add_argument(
        '--model-temperature',
        type=float,
        default=0.0,
        help=(
            "Temperature setting for the LLM model "
            "(default: 0.0 for deterministic outputs)."
        ),
    )
    parser.add_argument(
        '--prompt-dir',
        type=str,
        default=Constants.DEFAULT_PROMPT_DIR,
        help=(
            "Directory where prompt CSV files are located "
            f"(default: {Constants.DEFAULT_PROMPT_DIR})."
        ),
    )
    parser.add_argument(
        '--ollama-server-port',
        type=int,
        default=OLLAMA_DEFAULT_PORT,
        help=(
            "Port number where the Ollama server is running "
            f"(default: {OLLAMA_DEFAULT_PORT})."
        ),
    )
    
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help=(
            "Maximum context length for LLM prompts (default: 2048)."
        ),
    )

    args = parser.parse_args()

    runner = OllamaBiasRunner(
        model_name=args.model,
        output_dir=args.output_dir,
        version=args.version,
        workers=args.workers,
        checkpoint_size=args.checkpoint_size,
        temperature=args.model_temperature,
        ollama_port=args.ollama_server_port,
        context_length=args.context_length,
    )

    runner.run_experiment(
        file_type=args.file_type,
        prompt_dir=args.prompt_dir,
    )


if __name__ == '__main__':
    main()
