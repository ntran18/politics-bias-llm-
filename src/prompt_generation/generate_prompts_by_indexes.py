#!/usr/bin/env python3
"""
Generate v8-cot prompts for the top-20 high-conflict articles.

This script:
1. Loads the clean data with article info
2. Filters to the top-20 conflict article indexes
3. Generates all 7 prompt variants (article_only, +source, +politics, +all, etc.)
4. Exports to data/prompts/v8-cot/
"""

import os
import sys
import pandas as pd

from prompt_generator import ArticlePromptGenerator
from constants import Constants

def load_indexes(sample_file=Constants.SAMPLE_ROWS_DATA_FILE):
    """Load conflict article indexes from analysis output."""
    if not os.path.exists(sample_file):
        print(f"ERROR: Conflict file not found: {sample_file}")
        sys.exit(1)
    
    conflict_df = pd.read_csv(sample_file)
    indexes = conflict_df['index'].tolist()
    print(f"✓ Loaded {len(indexes)} article indexes from {sample_file}")
    return indexes

def load_file(indexes=None, sample_file=None):
    """Load and filter clean data to top conflict articles."""
    print("Loading clean data with article info...")
    
    if not os.path.exists(Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO):
        print(f"ERROR: File not found: {Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO}")
        sys.exit(1)
    
    # Load conflict indexes if not provided
    if indexes is None:
        if sample_file is None:
            sample_file = Constants.SAMPLE_ROWS_DATA_FILE
        indexes = load_indexes(sample_file)
    
    df = pd.read_csv(Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO)
    print(f"Loaded {len(df)} total rows")
    
    # Filter to conflict indexes
    filtered_df = df[df['index'].isin(indexes)].copy()
    
    # Sort by index to maintain order
    filtered_df = filtered_df.sort_values('index')
    
    print(f"Filtered to {len(filtered_df)} rows for {len(indexes)} conflict articles")
    print(f"Article indexes: {sorted(filtered_df['index'].unique())}")
    
    return filtered_df

def generate_sample_prompts(data, output_dir='../../data/prompts/v8cot'):
    """Generate all 7 prompt variants for v8-cot."""
    print(f"\nGenerating v8-cot prompts to {output_dir}...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create generator
    generator = ArticlePromptGenerator(output_dir=output_dir, context_length=-1)
    
    # Generate all prompt types
    print("\n--- Generating Article Info Prompts ---")
    generator.generate_article_info_prompts(data)
    
    print("\n--- Generating Politics Variant Prompts ---")
    generator.generate_politics_prompts(data)
    
    print("\n--- Generating Source Variant Prompts ---")
    generator.generate_source_prompts(data)
    
    print("\n--- Generating PII Combined All Prompts ---")
    generator.generate_pii_combined_all_prompts(data)
    
    print("\n--- Generating Source + Politics Variant Prompts ---")
    generator.generate_source_politics_prompts(data)
    
    print("\n--- Generating Source + PII Variant Prompts ---")
    generator.generate_source_pii_prompts(data)
    
    print("\n--- Generating Politics + PII Variant Prompts ---")
    generator.generate_politics_pii_prompts(data)
    
    print(f"\n✓ All v8-cot prompt files generated successfully!")
    
    # Verify generated files
    print("\nVerifying generated files:")
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            print(f"  {filename}: {len(df)} rows")

if __name__ == '__main__':
    print("=" * 70)
    print("Prompt generation by indexes")
    print("=" * 70)
    
    # Load conflict data from analysis results
    data = load_file()
    
    # Generate v8-cot prompts
    generate_sample_prompts(data)
    
    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)
