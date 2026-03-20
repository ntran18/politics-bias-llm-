class Constants:
    NEW_SOURCES = ['BBC', 'Fox News', 'CNN']
    NEW_POLITICS = ['Conservative', 'Liberal', 'Independent', 'Other']

    # Default file paths
    DEFAULT_INPUT_FILE = '../data/news_bias_full_data.csv'
    CLEAN_DATA_FILE = '../data/clean_original_data.csv'
    CLEAN_DATA_FILE_WITH_ARTICLE_INFO = '../data/clean_data_with_article_info.csv'
    ARTICLES_INFO_FILE = '../data/data_articles_info.csv'
    DEFAULT_PROMPT_DIR = '../data/prompts/'
    DEFAULT_OUTPUT_DIR = '../results/'
    DEFAULT_LLM_OUTPUT_FOLDER = "llm_outputs/"
    DEFAULT_PROMPT_ARTICLE_INFO_FILE = 'prompt_article_info.csv'
    DEFAULT_PROMPT_POLITICS_FILE = 'prompt_politics_variants.csv'
    DEFAULT_PROMPT_SOURCE_FILE = 'prompt_source_variants.csv'
    DEFAULT_PROMPT_PII_COMBINED_ALL_FILE = 'prompt_pii_combined_variants.csv'
    
    DEFAULT_ANALYSIS_FOLDER = 'analysis_reports/'
    
    LLM_RESULT_FILE_PREFIX = 'llm_output'
    PROMPT_FILE_MAP = {
        'articles_info': DEFAULT_PROMPT_ARTICLE_INFO_FILE,
        'politics_variants': DEFAULT_PROMPT_POLITICS_FILE,
        'sources': DEFAULT_PROMPT_SOURCE_FILE,
        'pii_combined_all': DEFAULT_PROMPT_PII_COMBINED_ALL_FILE
    }
    
    MODEL_NAME = "llama3.1:8b"
    VERSION = 'v5'
    BATCH_SIZE = 4
    TOKEN_LIMIT = 4096
    CONTEXT_LENGTH = 4096