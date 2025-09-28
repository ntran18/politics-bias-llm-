import argparse
from article_fetcher import ArticleFetcher
from constants import Constants
import os
import pandas as pd

class ArticlePromptGenerator:
    """
    Generates various LLM prompts based on processed article data.
    """

    # --- Configuration Constants ---
    
    # Define the general prompt structure for reusability
    GENERAL_PROMPT_TEMPLATE = (
        "You are going to be the reader of a political article. Your job is to "
        "determine whether or not the article is biased. An article is defined as biased "
        "if it ever exaggerates, misrepresents, omits, or otherwise distorts facts "
        "(including by making subjective opinions look like facts) for the purpose of "
        "appealing to a certain political group. After reading the article {additional_info}, "
        "your answer should be a single JSON object. "
        "The object MUST contain three keys: 'assessment' (value must be 'is-biased' or 'is-not-biased'), "
        "'confidence_score' (value must be an integer from 1 to 100 representing your confidence), "
        "and 'explanation' (value must be a string detailing your full reasoning). "
        "Provide no further text, only the complete JSON object.\n"
    )
    
    # Define common string templates
    ARTICLE_INFO_TEMPLATE = "Article Title: {title}\nArticle Content: {content}\n"
    VIEWPOINT_PROMPT = "from the viewpoint of an individual who is "
    READER_INFO_TEMPLATE = "{age}-year-old {gender} {language} speaker from {country}"
    LETTER_SOURCE_TEMPLATE = "from the publication or organization {source}"
    
    def __init__(self, output_dir='data'):
        """
        Initializes the generator with output path.
        """
        self.output_dir = output_dir

    def _export_data(self, data, file_path):
        """Internal helper to export a DataFrame to a CSV file."""
        print(f"Exporting data to {file_path}")
        data.to_csv(file_path, index=False)

    def _get_unique_articles_df(self, data):
        """Returns a DataFrame containing only unique articles."""
        return data[['article_id', 'article_title', 'article_content']].drop_duplicates(subset=['article_id']).reset_index(drop=True)

    def _create_article_info_text(self, row):
        """Helper to create the Article Title/Content block."""
        return self.ARTICLE_INFO_TEMPLATE.format(
            title=row['article_title'],
            content=row['article_content']
        )
        
    def _create_reader_info_text(self, row):
        """Helper to create the Reader PII info block."""
        return self.READER_INFO_TEMPLATE.format(
            age=row['age'],
            gender=row['gender'],
            language=row['language'],
            country=row['country']
        )

    def _generate_unique_article_prompts(self, data):
        """
        1. Prompt Article Info Only: 
           Creates a CSV with one row per unique article (to prevent duplication).
        """
        print(f"Generating {Constants.DEFAULT_PROMPT_ARTICLE_INFO_FILE}...")
        unique_df = self._get_unique_articles_df(data)
        prompt_data = []

        for _, row in unique_df.iterrows():
            article_info_text = self._create_article_info_text(row)
            
            prompt = self.GENERAL_PROMPT_TEMPLATE.format(additional_info = "") + article_info_text
            
            prompt_data.append({
                'article_id': row['article_id'],
                'article_title': row['article_title'],
                'prompt': prompt,
            })

        df = pd.DataFrame(prompt_data)
        output_path = os.path.join(self.output_dir, Constants.DEFAULT_PROMPT_ARTICLE_INFO_FILE)
        self._export_data(df, output_path)

    def _generate_variant_prompts(self, data, variant_type, variant_list, file_prefix):
        """
        2. Prompt Politics Info Variants
        3. Prompt Source Info Variants
           Generates prompts by iterating unique articles against a list of variants.
        """
        print(f"Generating {file_prefix}_variants.csv...")
        unique_df = self._get_unique_articles_df(data)
        prompt_data = []

        for _, article_row in unique_df.iterrows():
            article_info_text = self._create_article_info_text(article_row)
            
            for variant in variant_list:
                if variant_type == 'politics':
                    # additional_info format: 'from the viewpoint of an individual who is Conservative'
                    if variant == 'Other':
                        # Special case for 'Other' to avoid confusion
                        additional_info = self.VIEWPOINT_PROMPT + "with an unspecified political affiliation"
                    else:
                        additional_info = self.VIEWPOINT_PROMPT + variant
                    variant_col = 'politics_variant'
                else: # source
                    # additional_info format: 'from the publication or organization BBC'
                    additional_info = self.LETTER_SOURCE_TEMPLATE.format(source=variant)
                    variant_col = 'source_variant'
                
                prompt = self.GENERAL_PROMPT_TEMPLATE.format(additional_info=additional_info) + article_info_text
                
                prompt_data.append({
                    'article_id': article_row['article_id'],
                    'article_title': article_row['article_title'],
                    variant_col: variant,
                    'prompt': prompt,
                })

        df = pd.DataFrame(prompt_data)
        output_path = os.path.join(self.output_dir, f"{file_prefix}_variants.csv")
        self._export_data(df, output_path)


    def _generate_combined_article_variants(self, data):
        """
        4. Prompt All Article Info Variants: 
           Creates prompts combining every new source with every new politics.
        """
        print(f"Generating {Constants.DEFAULT_PROMPT_ALL_ARTICLE_INFO_VARIANTS_FILE}...")
        unique_df = self._get_unique_articles_df(data)
        prompt_data = []

        for _, article_row in unique_df.iterrows():
            article_info_text = self._create_article_info_text(article_row)
            
            for source in Constants.NEW_SOURCES:
                for politics in Constants.NEW_POLITICS:
                    
                    source_context = self.LETTER_SOURCE_TEMPLATE.format(source=source)
                    if politics == 'Other':
                        politics_context = self.VIEWPOINT_PROMPT + "with an unspecified political affiliation"
                    else:
                        politics_context = self.VIEWPOINT_PROMPT + politics
                    
                    # Combine: Source context + " and " + Politics context
                    additional_info = f"{source_context} and {politics_context}"
                    
                    prompt = self.GENERAL_PROMPT_TEMPLATE.format(additional_info=additional_info) + article_info_text
                    
                    prompt_data.append({
                        'article_id': article_row['article_id'],
                        'article_title': article_row['article_title'],
                        'source_variant': source,
                        'politics_variant': politics,
                        'prompt': prompt,
                    })

        df = pd.DataFrame(prompt_data)
        output_path = os.path.join(self.output_dir, Constants.DEFAULT_PROMPT_ALL_ARTICLE_INFO_VARIANTS_FILE)
        self._export_data(df, output_path)


    def _generate_pii_prompts(self, data):
        """
        5. Prompt Reader PII Info: 
           Uses all original data rows (including user demographics) + Article Info.
        """
        print(f"Generating {Constants.DEFAULT_PROMPT_READER_PII_FILE}...")
        prompt_data = []
        
        for _, row in data.iterrows():
            article_info_text = self._create_article_info_text(row)
            reader_info_text = self._create_reader_info_text(row)
            
            # Additional info: Reader PII + their stated politics
            reader_pii_context = f"{reader_info_text} and is {row['politics']}"
            
            # Combine PII context with viewpoint prompt
            additional_info = self.VIEWPOINT_PROMPT + reader_pii_context
            
            prompt = self.GENERAL_PROMPT_TEMPLATE.format(additional_info=additional_info) + article_info_text
            
            prompt_data.append({
                'article_id': row['article_id'],
                'age': row['age'],
                'gender': row['gender'],
                'politics': row['politics'],
                'prompt': prompt,
            })

        df = pd.DataFrame(prompt_data)
        output_path = os.path.join(self.output_dir, Constants.DEFAULT_PROMPT_READER_PII_FILE)
        self._export_data(df, output_path)

    def _generate_pii_combined_variants(self, data):
        """
        6. Prompt PII + All Article Info Variants: 
           Combines Reader PII (Age/Gender/Language/Country) + all Article Variants (Source/Politics).
        """
        print(f"Generating {Constants.DEFAULT_PROMPT_PII_COMBINED_VARIANTS_FILE}...")
        prompt_data = []

        for _, row in data.iterrows():
            article_info_text = self._create_article_info_text(row)
            reader_info_text = self._create_reader_info_text(row)
            
            for source in Constants.NEW_SOURCES:
                for politics in Constants.NEW_POLITICS:
                    
                    source_context = self.LETTER_SOURCE_TEMPLATE.format(source=source)
                    politics_context = self.VIEWPOINT_PROMPT + politics
                    
                    # Combination of Variant Source + Variant Politics + Reader PII
                    
                    # Example construction: 'from the publication or organization BBC and from the viewpoint of an individual who is Conservative and 30-year-old male English speaker from USA'
                    
                    additional_info = (
                        f"{source_context} and {politics_context} and is "
                        f"{reader_info_text}"
                    )
                    
                    prompt = self.GENERAL_PROMPT_TEMPLATE.format(additional_info=additional_info) + article_info_text
                    
                    prompt_data.append({
                        'article_id': row['article_id'],
                        'age': row['age'],
                        'gender': row['gender'],
                        'source_variant': source,
                        'politics_variant': politics,
                        'prompt': prompt,
                    })

        df = pd.DataFrame(prompt_data)
        output_path = os.path.join(self.output_dir, Constants.DEFAULT_PROMPT_PII_COMBINED_VARIANTS_FILE)
        self._export_data(df, output_path)


    def generate_all_prompts(self, data):
        """
        Generates all six different variations of LLM prompts and saves them to separate files.
        """
        print("Starting prompt generation and segmentation into files...")
        
        # 1. Prompt Article Info Only (Unique Articles)
        self._generate_unique_article_prompts(data)

        # 2. Prompt Politics Variant (Unique Articles x 4 Politics)
        self._generate_variant_prompts(data, 'politics', Constants.NEW_POLITICS, "prompt_politics")

        # 3. Prompt Source Variant (Unique Articles x 3 Sources)
        self._generate_variant_prompts(data, 'source', Constants.NEW_SOURCES, "prompt_source")

        # 4. Prompt All Article Info Variants (Source x Politics)
        self._generate_combined_article_variants(data)
        
        # 5. Prompt Reader PII Info (Original Data Rows with original politics)
        self._generate_pii_prompts(data)

        # 6. Prompt PII + All Article Info Variants (Original Data Rows x Source x Politics)
        self._generate_pii_combined_variants(data)
        
        print("All prompt files generated successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate LLM prompts from news bias data with selective execution.")
    
    # Arguments for pipeline stages
    parser.add_argument('--clean', action='store_true', help="Run the data cleaning step (requires original input CSV).")
    parser.add_argument('--fetch', action='store_true', help="Run the article fetching/scraping step (requires clean data or runs after --clean).")
    parser.add_argument('--input-file', type=str, default=ArticleFetcher.DEFAULT_INPUT_FILE, help="Path to the original input CSV file.")
    parser.add_argument('--output-dir', type=str, default='data', help="Directory to save intermediate and final CSV files.")
    parser.add_argument('--version', type=str, default='v5', help="Version label for output directory (e.g., v1, v2, v3, etc.).")

    # Arguments for specific prompt types
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument('--all-prompts', action='store_true', help="Generate all six prompt types.")
    prompt_group.add_argument('--prompts', nargs='+', choices=[
        'article_info', 'politics', 'source', 
        'combined_article', 'pii', 'pii_combined'
    ], help="Specify a list of prompt types to generate (e.g., politics source).")
    
    args = parser.parse_args()
    
    # Initialize the fetcher and generator classes
    output_dir = os.path.join(Constants.DEFAULT_PROMPT_DIR, args.version)
    fetcher = ArticleFetcher(input_file=args.input_file, output_dir=output_dir)
    generator = ArticlePromptGenerator(output_dir=output_dir)

    # 1. Run cleaning if requested
    if args.clean:
        fetcher.clean_data()
        
    # 2. Run fetching if requested
    if args.fetch:
        # fetch_article_info returns the final DataFrame
        data = fetcher.fetch_article_info()
    
    # 3. Setup data for prompt generation (Load from file if steps skipped, or get from fetcher)
    data = fetcher.get_data()
    if (args.prompts or args.all_prompts):
        print("Artcles info path:", fetcher.articles_info_path)
        if data is None and os.path.exists(fetcher.articles_info_path):
            try:
                # Load the final processed data with article content from the last saved file
                data = pd.read_csv(fetcher.articles_info_path)
                print(f"Loaded article data from {fetcher.articles_info_path}")
            except Exception as e:
                print(f"Error loading article data: {e}. Cannot generate prompts.")
                exit(1)
        elif data is None:
            print("Data is not ready. Cannot generate prompts. Please run --clean and --fetch first.")
            exit(1)
        
    # 4. Determine and run prompt generation methods, passing the data
    
    # Map arguments to method names (using partial function application for variants)
    prompt_methods_map = {
        'article_info': lambda d: generator._generate_unique_article_prompts(d),
        'politics': lambda d: generator._generate_variant_prompts(d, 'politics', generator.NEW_POLITICS, "prompt_politics"),
        'source': lambda d: generator._generate_variant_prompts(d, 'source', generator.NEW_SOURCES, "prompt_source"),
        'combined_article': lambda d: generator._generate_combined_article_variants(d),
        'pii': lambda d: generator._generate_pii_prompts(d),
        'pii_combined': lambda d: generator._generate_pii_combined_variants(d),
    }
    
    if args.all_prompts:
        generator.generate_all_prompts(data)
    elif args.prompts:
        print("Starting prompt generation based on arguments...")
        for method_key in args.prompts:
            method = prompt_methods_map[method_key]
            method(data)
        print("Selected prompt files generated successfully.")