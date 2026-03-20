"""
Combine datasets for each model across different versions and prompts.
This script creates a comprehensive dataset for each model by combining data
from all versions and all prompt types.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
import re
from utils import MODELS


def get_all_versions(results_dir: str) -> List[str]:
    """Get all version folders (v5, v6, v7.1, etc.)"""
    versions = []
    for item in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, item)) and item.startswith('v'):
            # Filter out non-version folders like 'results' and 'test'
            if re.match(r'^v\d+(\.\d+)?$', item):
                versions.append(item)
    return sorted(versions)


def get_all_models(results_dir: str, versions: List[str]) -> List[str]:
    """Get all unique model names across all versions"""
    models = set()
    for version in versions:
        version_path = os.path.join(results_dir, version)
        if os.path.exists(version_path):
            for item in os.listdir(version_path):
                item_path = os.path.join(version_path, item)
                if os.path.isdir(item_path) and item not in ['llm_outputs', 'analysis_reports']:
                    models.add(item)
    return sorted(list(models))


def get_prompt_files(model_dir: str) -> List[str]:
    """Get all CSV files in the llm_outputs directory"""
    llm_outputs_dir = os.path.join(model_dir, 'llm_outputs')
    if not os.path.exists(llm_outputs_dir):
        return []
    
    csv_files = []
    for file in os.listdir(llm_outputs_dir):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(llm_outputs_dir, file))
    return csv_files


def combine_model_data(results_dir: str, model_name: str, versions: List[str], original_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Combine all data for a specific model across all versions and prompts.
    Optionally merge with original data to get comprehensive information.
    
    Args:
        results_dir: Path to results directory
        model_name: Name of the model (e.g., 'llama3.2:3b')
        versions: List of version folders to process
        original_df: Optional DataFrame with original article information
    
    Returns:
        Combined DataFrame with version, prompt_type, and original data columns added
    """
    all_data = []
    
    for version in versions:
        model_dir = os.path.join(results_dir, version, model_name)
        
        if not os.path.exists(model_dir):
            print(f"  Skipping {version}/{model_name} - directory not found")
            continue
        
        csv_files = get_prompt_files(model_dir)
        
        if not csv_files:
            print(f"  No CSV files found in {version}/{model_name}")
            continue
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Add metadata columns
                df['version'] = version
                df['prompt_type'] = Path(csv_file).stem
                df['llm_model'] = model_name
                
                all_data.append(df)
                print(f"  ✓ Loaded {version}/{model_name}/{Path(csv_file).name} ({len(df)} rows)")
                
            except Exception as e:
                print(f"  ✗ Error loading {csv_file}: {e}")
    
    if not all_data:
        print(f"  No data found for {model_name}")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    llm_cols_to_keep = ['index', 'llm_assessment', 'llm_confidence', 'llm_explanation', 'version', 'prompt_type', 'llm_model']
    combined_df = combined_df[llm_cols_to_keep]
    combined_df = combined_df.merge(original_df, on='index', how='left')
    
    return combined_df


def main():
    # Configuration
    results_dir = './results'
    output_dir = os.path.join(results_dir, 'combined_datasets')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original data once
    print("Loading original data...")
    original_df = None
    data_dir = './data'
    original_data_path = os.path.join(data_dir, 'clean_original_data.csv')
    if os.path.exists(original_data_path):
        try:
            original_df = pd.read_csv(original_data_path)
            print(f"Loaded original data: {len(original_df)} records, {len(original_df.columns)} columns\n")
        except Exception as e:
            print(f"Warning: Could not load original data: {e}\n")
    else:
        print(f"Original data file not found: {original_data_path}\n")
    
    # Get all versions and models
    print("Scanning results directory...")
    versions = ["v7.1", "v7.2", "v7.3"]
    print(f"Found versions: {', '.join(versions)}")
    
    
    print(f"Found models: {', '.join(MODELS)}\n")
    
    # Process each model
    summary = []
    
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Processing model: {model}")
        print(f"{'='*60}")
        
        combined_df = combine_model_data(results_dir, model, versions, original_df)
        
        if not combined_df.empty:
            # Save combined dataset
            # Replace special characters in model name for filename
            safe_model_name = model.replace(':', '_').replace('/', '_')
            output_file = os.path.join(output_dir, f'{safe_model_name}_combined.csv')
            
            combined_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved combined dataset: {output_file}")
            print(f"  Total rows: {len(combined_df)}")
            print(f"  Columns: {len(combined_df.columns)}")
            print(f"  Versions: {combined_df['version'].nunique()}")
            print(f"  Prompt types: {combined_df['prompt_type'].nunique()}")
            
            # Add to summary
            summary.append({
                'model': model,
                'total_rows': len(combined_df),
                'total_columns': len(combined_df.columns),
                'versions': combined_df['version'].nunique(),
                'prompt_types': combined_df['prompt_type'].nunique(),
                'output_file': output_file
            })

        else:
            print(f"\n✗ No data found for {model}")
    
    # Save summary
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_file = os.path.join(output_dir, '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo data was combined.")


if __name__ == '__main__':
    main()
