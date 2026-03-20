import argparse
from article_fetcher import ArticleFetcher
from prompt_generation.constants import Constants 
import os
import pandas as pd

class ArticlePromptGenerator:
    """
    Generates various LLM prompts based on processed article data.
    Modified to prioritize row-by-row generation from the original dataset.
    """

    # --- Configuration Constants ---

    # COMMON STRING TEMPLATES
    ARTICLE_INFO_TEMPLATE = "Article Title: {title}\nArticle Content: {content}\n"
    
    VIEWPOINT_PROMPT = "Assume the viewpoint of an individual who is **{politics}**."
    PII_PROMPT = "**Demographics:** A {age}-year-old {gender} {language} speaker from {country}."
    SOURCE_AWARENESS_PROMPT = "**Source Awareness:** Aware this article is **from the publication or organization {source}**."
    CONTEXT_START = "Analyze the following article based on this reader context."
    CONTEXT_START_BLANK = "Assume the viewpoint of an **unspecified reader**. Analyze the following article."
    
    def __init__(self, output_dir='data', context_length=2048):
        """
        Initializes the generator with output path.
        """
        self.output_dir = output_dir
        self.context_length = context_length

    def _export_data(self, data, file_path):
        """Internal helper to export a DataFrame to a CSV file."""
        print(f"Exporting data to {file_path}")
        data.to_csv(file_path, index=False)

    def _create_article_info_text(self, row):
        """Helper to create the Article Title/Content block."""
        return self.ARTICLE_INFO_TEMPLATE.format(
            title=row['article_title'],
            content=row['article_content']
        )
        
    def _create_pii_context_text(self, row):
        """Helper to create the Reader PII info block."""
        return self.PII_PROMPT.format(
            age=row['age'],
            gender=row['gender'],
            language=row['language'],
            country=row['country']
        )
    
    def _build_user_prompt_context(self, row, include_source=False, include_politics=False, include_pii=False):
        """
        Helper to build the introductory context part of the User Prompt.
        """
        
        if not (include_source or include_politics or include_pii):
            return self.CONTEXT_START_BLANK
        
        context_parts = []
        
        if include_politics:
            politics_context = self.VIEWPOINT_PROMPT.format(politics=row['politics'])
            context_parts.append(politics_context)
        else:
            context_parts.append("Assume the viewpoint of a reader with **unspecified politics**.")
        
        if include_source:
            source_context = self.SOURCE_AWARENESS_PROMPT.format(source=row['source'])
            context_parts.append(source_context)
            
        if include_pii:
            pii_context = self._create_pii_context_text(row)
            context_parts.append(pii_context)
            
        context_parts.append(self.CONTEXT_START)
            
        return "\n".join(context_parts).strip()

    def _generate_prompts_with_context(self, data, file_name, context_config):
        """
        Internal helper to generate row-by-row prompts based on a configuration.

        :param data: Input DataFrame.
        :param file_name: Output file name (including assumed path).
        :param context_config: Dictionary defining which contexts to include and 
                               which columns to include in the output CSV.
               Example: 
               {
                   'include_source': True, 
                   'include_politics': False, 
                   'include_pii': False,
                   'output_cols': ['source_variant']
               }
        """
        print(f"Generating {file_name} with context: {list(k for k, v in context_config.items() if v and k != 'output_cols')}...")
        prompt_data = []

        include_source = context_config.get('include_source', False)
        include_politics = context_config.get('include_politics', False)
        include_pii = context_config.get('include_pii', False)
        output_cols = context_config.get('output_cols', [])

        for index, row in data.iterrows():
            user_context = self._build_user_prompt_context(
                row, 
                include_source=include_source, 
                include_politics=include_politics, 
                include_pii=include_pii
            )
            article_info_text = self._create_article_info_text(row)

            prompt = user_context + "\n" + article_info_text

            if self.context_length > 0 and len(prompt) > self.context_length:
                prompt = prompt[:self.context_length]
            
            output_row = {
                'article_id': row['article_id'],
                'index': index,
                'prompt': prompt,
            }
            
            for col in output_cols:
                output_row[col] = row[col]

            prompt_data.append(output_row)

        df = pd.DataFrame(prompt_data)
        output_path = os.path.join(self.output_dir, file_name)
        self._export_data(df, output_path)

    def generate_article_info_prompts(self, data):
        FILE_NAME = Constants.DEFAULT_PROMPT_ARTICLE_INFO_FILE
        config = {
            'include_source': False, 
            'include_politics': False, 
            'include_pii': False,
            'output_cols': []
        }
        self._generate_prompts_with_context(data, FILE_NAME, config)

    def generate_source_prompts(self, data):
        FILE_NAME = Constants.DEFAULT_PROMPT_SOURCE_FILE 
        config = {
            'include_source': True, 
            'include_politics': False, 
            'include_pii': False,
            'output_cols': ['source']
        }
        self._generate_prompts_with_context(data, FILE_NAME, config)
    
    def generate_politics_prompts(self, data):
        FILE_NAME = Constants.DEFAULT_PROMPT_POLITICS_FILE
        config = {
            'include_source': False, 
            'include_politics': True, 
            'include_pii': False,
            'output_cols': ['politics']
        }
        self._generate_prompts_with_context(data, FILE_NAME, config)
        
    def generate_pii_combined_all_prompts(self, data):
        FILE_NAME = Constants.DEFAULT_PROMPT_PII_COMBINED_ALL_FILE
        config = {
            'include_source': True, 
            'include_politics': True, 
            'include_pii': True,
            'output_cols': ['age', 'gender', 'source', 'politics']
        }
        self._generate_prompts_with_context(data, FILE_NAME, config)

    def generate_all_prompts(self, data):
        """
        Generates all four required variations of LLM prompts.
        """
        print("Starting prompt generation and segmentation into files...")
        
        self.generate_article_info_prompts(data)
        self.generate_politics_prompts(data)
        self.generate_source_prompts(data)
        self.generate_pii_combined_all_prompts(data)
        
        print("All prompt files generated successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate LLM prompts from news bias data with selective execution.")
    
    parser.add_argument('--clean', action='store_true', help="Run the data cleaning step (requires original input CSV).")
    parser.add_argument('--fetch', action='store_true', help="Run the article fetching/scraping step (requires clean data or runs after --clean).")
    parser.add_argument('--input-file', type=str, default=Constants.DEFAULT_INPUT_FILE, help="Path to the original input CSV file.")
    parser.add_argument('--output-dir', type=str, default=Constants.DEFAULT_PROMPT_DIR, help="Directory to save intermediate and final CSV files.")
    parser.add_argument('--version', type=str, default='v5', help="Version label for output directory (e.g., v1, v2, v3, etc.).")
    parser.add_argument('--context-length', type=int, default=-1, help="Maximum context length for LLM prompts (default: 2048).")
    
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument('--all-prompts', action='store_true', help="Generate all four simplified prompt types.")
    prompt_group.add_argument('--prompts', nargs='+', choices=[
        'articles_info', 'politics', 'sources', 'pii_combined_all'
    ], help="Specify a list of simplified prompt types to generate.")
    
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.version)
    fetcher = ArticleFetcher(input_file=args.input_file, output_dir=output_dir)
    generator = ArticlePromptGenerator(output_dir=output_dir, context_length=args.context_length)

    if args.clean:
        fetcher.clean_data()
        
    if args.fetch:
        data = fetcher.fetch_article_info()
    
    data = fetcher.get_data()
    
    if (args.prompts or args.all_prompts):
        print("Articles info path:", Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO)
        if data is None:
            print("Data is not ready. Cannot generate prompts. Please run --clean and --fetch first.")
            exit(1)

    prompt_methods_map = {
        'articles_info': generator.generate_article_info_prompts,
        'politics': generator.generate_politics_prompts,
        'sources': generator.generate_source_prompts,
        'pii_combined_all': generator.generate_pii_combined_all_prompts,
    }

    if args.all_prompts:
        generator.generate_all_prompts(data)
    elif args.prompts:
        print("Starting prompt generation based on original arguments...")
        for method_key in args.prompts:
            method = prompt_methods_map[method_key]
            method(data)
        print("Selected prompt files generated successfully.")