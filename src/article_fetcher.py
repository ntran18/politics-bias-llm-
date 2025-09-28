from constants import Constants
from bs4 import BeautifulSoup
import chardet
import os
import pandas as pd
import requests

class ArticleFetcher:
    """
    Handles data cleaning, article scraping, and managing the intermediate data files.
    """
    
    # Define default file paths for this class
    DEFAULT_INPUT_FILE = 'news_bias_full_data.csv'
    CLEAN_DATA_FILE = 'data/clean_original_data.csv'
    ARTICLES_INFO_FILE = 'data/data_articles_info.csv'
    
    def __init__(self, input_file=Constants.DEFAULT_INPUT_FILE, output_dir='data'):
        """
        Initializes the fetcher with input/output paths.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.clean_data_path = Constants.CLEAN_DATA_FILE
        self.articles_info_path = Constants.ARTICLES_INFO_FILE
        self._ensure_output_dir()
        self.data = None

    def _ensure_output_dir(self):
        """Creates the output directory if it does not exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _export_data(self, data, file_path):
        """Internal helper to export a DataFrame to a CSV file."""
        print(f"Exporting data to {file_path}")
        data.to_csv(file_path, index=False)

    def clean_data(self):
        """
        Loads the original dataset and preprocesses it by selecting and renaming columns.
        Saves the cleaned data and updates the internal DataFrame.
        """
        print(f"Starting data cleaning from {self.input_file}...")
        try:
            data = pd.read_csv(self.input_file)
        except FileNotFoundError:
            print(f"Error: Input file {self.input_file} not found.")
            return

        # Select columns for the clean data
        clean_data = data[['Answer.age', 'Answer.articleNumber', 'Answer.batch', 'Answer.bias-question', 'Answer.country', 'Answer.gender', 'Answer.language1', 'Answer.newsOutlet', 'Answer.politics', 'Answer.url']]
        
        # Define renaming map
        rename_map = {
            'Answer.age': 'age', 
            'Answer.articleNumber': 'articleNumber',
            'Answer.batch': 'batch', 
            'Answer.bias-question': 'bias-question', 
            'Answer.country': 'country', 
            'Answer.gender': 'gender', 
            'Answer.language1': 'language', 
            'Answer.newsOutlet': 'source', 
            'Answer.politics': 'politics', 
            'Answer.url': 'url'
        }
        
        clean_data.rename(columns=rename_map, inplace=True)
        self.data = clean_data
        self._export_data(self.data, self.clean_data_path)
        print("Data cleaning complete.")

    def _get_article_details(self, url):
        """
        Internal method to scrape the title and content from a given URL.
        Includes robust error handling and encoding detection.
        """
        try:
            response = requests.get(url, timeout=10)

            # Detect encoding and attempt to use it
            detected_encoding = chardet.detect(response.content)['encoding']
            
            if detected_encoding:
                response.encoding = detected_encoding
            else:
                response.encoding = 'utf-8'

            if response.status_code != 200:
                print(
                    f"Failed to fetch {url}. Status code: {response.status_code}")
                return None, None

            soup = BeautifulSoup(response.text, "html.parser")

            # --- NEW EXCLUSION LOGIC ---
            # Remove unwanted structural elements before scraping the content
            EXCLUSION_CLASSES = ['article-meta', 'article-footer']
            for class_name in EXCLUSION_CLASSES:
                for unwanted_element in soup.find_all(class_=class_name):
                    unwanted_element.decompose() # Remove the element and its content
            # ---------------------------

            # Attempt to find title (h1 is a common target)
            title_tag = soup.find("h1")
            title = title_tag.text.strip() if title_tag else "No Title Found"
            
            # Attempt to find article body (using <article> tag)
            content = []
            body = soup.find("article")
            if body:
                for paragraph in body.find_all("p"):
                    # Exclude paragraphs explicitly marked with the 'footnote' class
                    if 'footnote' not in paragraph.get('class', []):
                        content.append(paragraph.text.strip())
            
            # Fallback if <article> is not found, checking common body containers
            if not body:
                 # Look for paragraphs in the main document body, often used for simple blogs
                for paragraph in soup.find_all("p", limit=10):
                    # Exclude paragraphs explicitly marked with the 'footnote' class
                    if 'footnote' not in paragraph.get('class', []):
                        content.append(paragraph.text.strip())
                    
            if not content:
                content = ["No Content Found"]

            return title, " ".join(content)
        
        except requests.exceptions.Timeout:
             print(f"Error: Timeout fetching URL: {url}")
             return None, None
        except Exception as e:
            print(f"Error occurred during scraping {url}: {e}")
            return None, None

    def fetch_article_info(self):
        """
        Loads the clean data, generates unique article IDs, scrapes details for
        each unique URL, and merges the data back into the DataFrame.
        """
        # Attempt to load data if not already loaded (e.g., if clean_data() was skipped)
        if self.data is None and os.path.exists(self.clean_data_path):
            self.data = pd.read_csv(self.clean_data_path)
        elif self.data is None:
            print("Clean data not found. Please run clean_data() first.")
            return

        print("Starting article detail fetching...")

        # 1. Prepare data for fetching
        data = self.data.copy()
        
        # Drop rows where URL is missing before factorization
        data.dropna(subset=['url'], inplace=True)

        # Create unique IDs for each unique URL
        data['article_id'] = data['url'].factorize()[0]
        
        # Get unique IDs and URLs to scrape only once per article
        unique_articles = data[['article_id', 'url']].drop_duplicates().sort_values(by='article_id')
        
        # Initialize dictionary to store results
        id_mappings = {}
        total_unique = len(unique_articles)

        # 2. Scrape details
        for index, row in unique_articles.iterrows():
            url = row['url']
            article_id = row['article_id']
            
            print(f"Fetching article {article_id + 1}/{total_unique}: {url}")
            title, content = self._get_article_details(url)
            
            id_mappings[article_id] = {
                'title': title, 
                'content': content
            }
        
        # 3. Merge results back into the main DataFrame
        title_map = {id: info['title'] for id, info in id_mappings.items()}
        content_map = {id: info['content'] for id, info in id_mappings.items()}

        data['article_title'] = data['article_id'].map(title_map)
        data['article_content'] = data['article_id'].map(content_map)
        
        self.data = data
        self._export_data(self.data, self.articles_info_path)
        print("Article detail fetching complete.")
        
        return self.data

    def get_data(self):
        """Returns the current internal DataFrame."""
        return self.data
