class Constants:
    NEW_SOURCES = ['BBC', 'Fox News', 'CNN']
    NEW_POLITICS = ['Conservative', 'Liberal', 'Independent', 'Other']

    # Default file paths
    DEFAULT_INPUT_FILE = '../data/news_bias_full_data.csv'
    CLEAN_DATA_FILE = '../data/clean_original_data.csv'
    ARTICLES_INFO_FILE = '../data/data_articles_info.csv'
    DEFAULT_PROMPT_DIR = '../data/prompts/'
    DEFAULT_OUTPUT_DIR = '../results/'
    DEFAULT_PROMPT_ARTICLE_INFO_FILE = 'prompt_article_info.csv'
    DEFAULT_PROMPT_POLITICS_VARIANTS_FILE = 'prompt_politics_variants.csv'
    DEFAULT_PROMPT_SOURCE_VARIANTS_FILE = 'prompt_source_variants.csv'
    DEFAULT_PROMPT_ALL_ARTICLE_INFO_VARIANTS_FILE = 'prompt_all_article_info_variants.csv'
    DEFAULT_PROMPT_READER_PII_FILE = 'prompt_reader_pii.csv'
    DEFAULT_PROMPT_PII_COMBINED_VARIANTS_FILE = 'prompt_pii_combined_variants.csv'
    
    PROMPT_FILE_MAP = {
        'article_info': DEFAULT_PROMPT_ARTICLE_INFO_FILE,
        'politics_variants': DEFAULT_PROMPT_POLITICS_VARIANTS_FILE,
        'source_variants': DEFAULT_PROMPT_SOURCE_VARIANTS_FILE,
        'all_article_info_variants': DEFAULT_PROMPT_ALL_ARTICLE_INFO_VARIANTS_FILE,
        'reader_pii': DEFAULT_PROMPT_READER_PII_FILE,
        'pii_combined_variants': DEFAULT_PROMPT_PII_COMBINED_VARIANTS_FILE
    }