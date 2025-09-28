import pandas as pd
import json
import argparse
import os
import ollama
from pydantic import BaseModel, ValidationError 
from tqdm import tqdm

from constants import Constants

# --- Configuration ---
# Define the base directory for input prompts and the mapping of file types
PROMPT_DIR = 'data/prompts-v5'


DEFAULT_OUTPUT_DIR = 'data/llm_results/'
MODEL_NAME = "llama3" # Ollama model name (e.g., llama3, mistral, phi3)

# LLM columns to be added to the output file
LLM_RESULT_COLUMNS = [
    'llm_assessment',
    'llm_confidence',
    'llm_explanation',
    'llm_model'
]

# Define the required output structure using Pydantic
class PoliticalBiasAssessment(BaseModel):
    """Defines the structure for the LLM's output."""
    assessment: str
    confidence_score: int
    explanation: str

# The SYSTEM_PROMPT is still necessary to guide the LLM's behavior,
# even though the format is enforced by the schema.
SYSTEM_PROMPT = (
    "You are a specialized political bias detection agent. "
    "Your response MUST be a single JSON object. "
    "The object MUST contain three keys: 'assessment' (value must be 'is-biased' or 'is-not-biased'), "
    "'confidence_score' (value must be an integer from 1 to 100 representing your confidence), "
    "and 'explanation' (value must be a string detailing your full reasoning). "
    "Provide no other text, commentary, or markdown blocks, only the complete JSON object."
)

def format_messages(system_prompt, user_query):
    """
    Formats the system and user input into the standardized list of message dictionaries
    required by the Ollama /api/chat endpoint.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    

def _load_data(input_file):
    """
    Loads the input CSV file containing prompts.
    """
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
    return data

def _setup_output_file(output_dir, input_file_path, model_name):
    """
    Prepares the output directory and file path for incremental saving.
    """
    input_filename = os.path.basename(input_file_path).replace('.csv', '')
    
    # Create output path using the input file name and model name
    output_filename = f"llm_output_{input_filename}_{model_name.split('/')[-1]}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)

    return output_path
    
def _initialize_output_file(output_path, columns):
    """
    Initializes the output CSV file with the appropriate header, or determines 
    how many rows have already been processed if the file exists (for resuming).
    """
    # Check if file exists to decide whether to write header
    write_header = not os.path.exists(output_path)
    processed_count = 0
    
    if write_header:
        # Write the header row using the definitive column list
        pd.DataFrame(columns=columns).to_csv(output_path, index=False, mode='w')
        print(f"Initialized output file: {output_path}")
    else:
        # File exists, try to read it to determine processed count
        try:
            # We read only the necessary columns (non-prompt columns)
            processed_df = pd.read_csv(output_path, usecols=[col for col in columns if col not in LLM_RESULT_COLUMNS])
            processed_count = processed_df.shape[0]
        except Exception:
            processed_count = 0
            
        print(f"Appending to existing file: {output_path} ({processed_count} rows already processed)")
        
    return processed_count


def _process_single_file(input_file_path, output_dir, model_name):
    """
    Loads prompts from a single file, runs batch inference, and saves the results incrementally.
    """
    print(f"\n--- Processing File: {os.path.basename(input_file_path)} ---")
    
    # 1. Load Data
    df = _load_data(input_file_path)
    if df is None:
        print(f"Skipping {os.path.basename(input_file_path)}.")
        return

    # 2. Setup Output File and Header
    output_path = _setup_output_file(output_dir, input_file_path, model_name)

    original_columns = df.columns.tolist()
    
    # Remove the 'prompt' column from the list of columns to be saved
    if 'prompt' in original_columns:
        original_columns.remove('prompt')
    
    all_columns = original_columns + LLM_RESULT_COLUMNS

    processed_count = _initialize_output_file(output_path, all_columns)

    # Get the JSON schema from the Pydantic model
    json_schema = PoliticalBiasAssessment.model_json_schema()

    # 3. Process Prompts and Save Incrementally
    total_prompts = len(df)
    
    print(f"Total prompts in file: {total_prompts}")
    
    # Skip rows that have already been processed
    df_to_process = df.iloc[processed_count:]
    
    # Use tqdm to show progress only for the remaining rows
    # The total in tqdm must be the remaining length, and initial is the starting index in the full DF
    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), initial=processed_count, desc=f"Inference ({os.path.basename(input_file_path)})"):
        user_query = row['prompt']
        messages = format_messages(SYSTEM_PROMPT, user_query)

        assessment = "INFERENCE_FAIL"
        confidence = None
        explanation = "Inference failed due to an unknown error."

        try:
            # 3.1. Make the Ollama API request using the client
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": 0.0,
                },
                format=json_schema,
            )
            
            response_text = response['message']['content'].strip()
            
            # 3.2. Validate and parse the JSON output using Pydantic
            llm_data = PoliticalBiasAssessment.model_validate_json(response_text)
            
            # 3.3. Extract data from the validated Pydantic model
            assessment = llm_data.assessment
            confidence = llm_data.confidence_score
            explanation = llm_data.explanation
                
        except ValidationError as e:
            # Catches errors if the model outputs JSON that doesn't match the schema
            print(f"\n[Warning] Pydantic validation failed for row {index}.")
            print(f"Error details: {e}")
            assessment = "VALIDATION_FAIL"
            confidence = None
            # Store the raw, invalid response for debugging
            explanation = response_text if 'response_text' in locals() else "Validation failed before response was retrieved."

        except ollama.ResponseError as e:
            # Catches errors from the Ollama API (e.g., model not found, internal server error)
            print(f"\n[Error] Ollama Response Error for row {index}: {e}")
            assessment = "OLLAMA_RESPONSE_FAIL"
            confidence = None
            explanation = f"Ollama response error: {e}"

        except Exception as e:
            # Catch all other exceptions (e.g., connection errors, other system issues)
            print(f"\n[Error] General inference error for row {index}: {e}")
            assessment = "INFERENCE_FAIL"
            confidence = None
            explanation = f"General system error: {e}"

        # 4. Collect results and write incrementally
        result = row.to_dict()
        
        # Add LLM results
        result.update({
            'llm_assessment': assessment,
            'llm_confidence': confidence,
            'llm_explanation': explanation,
            'llm_model': model_name
        })

        # Convert the result dictionary to a single-row DataFrame, 
        # specifying all_columns ensures only the desired columns (excluding 'prompt') are written
        result_df_row = pd.DataFrame([result], columns=all_columns)
        
        # Append the row to the CSV file (mode='a', header=False)
        result_df_row.to_csv(output_path, index=False, mode='a', header=False)


    print(f"--- Finished processing {os.path.basename(input_file_path)}. Results written to: {output_path} ---")


def main():
    parser = argparse.ArgumentParser(description="Run batch LLM inference over generated prompts using Ollama.")
    
    # Available file types for selection
    file_type_choices = ['all'] + list(Constants.PROMPT_FILE_MAP.keys())

    parser.add_argument(
        '--file-type', 
        type=str, 
        default='all', 
        choices=file_type_choices,
        help=f"The type of prompt file(s) to process. Choices: {', '.join(file_type_choices)}. Default: 'all'"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default=MODEL_NAME, 
        help=f"Ollama model name to use for inference (e.g., 'mistral', 'llama3'). Default: {MODEL_NAME}"
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=Constants.DEFAULT_OUTPUT_DIR, 
        help=f"Directory to save the resulting CSV file. Default: {Constants.DEFAULT_OUTPUT_DIR}"
    )
    
    args = parser.parse_args()
    
    # Determine which files to run
    if args.file_type == 'all':
        files_to_run = list(Constants.PROMPT_FILE_MAP.values())
    else:
        # Look up the specific file name based on the type provided
        files_to_run = [Constants.PROMPT_FILE_MAP[args.file_type]]

    print(f"Targeting prompt files in: {Constants.DEFAULT_PROMPT_DIR}")
    print(f"Using model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Files to process: {files_to_run}")

    # Execute batch processing for all selected files
    for file_name in files_to_run:
        input_file_path = os.path.join(Constants.DEFAULT_PROMPT_DIR, args.version, file_name)
        
        if not os.path.exists(input_file_path):
            print(f"\n[Skipping] Input file not found: {input_file_path}")
            continue
            
        _process_single_file(input_file_path, args.output_dir, args.model)


if __name__ == '__main__':
    main()
