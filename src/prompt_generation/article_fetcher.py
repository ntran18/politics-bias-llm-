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
    
    def __init__(self, input_file=Constants.DEFAULT_INPUT_FILE, output_dir='data'):
        """
        Initializes the fetcher with input/output paths.
        """
        self.input_file = input_file
        self.output_dir = output_dir
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
        clean_data.reset_index(inplace=True)
        
        clean_data.rename(columns=rename_map, inplace=True)
        self.data = clean_data
        self._export_data(self.data, Constants.CLEAN_DATA_FILE)
        print("Data cleaning complete.")
        
    def _filter_necessary_text(self, paragraph):
        text = ""
        if 'footnote' not in paragraph.get('class', []):
            text = paragraph.text.strip()
            if text in ["CLICK HERE TO GET THE FOX NEWS APP", "Stay tuned for all the latest details."]:
                text = ""
        return text

    def _get_article_details(self, url) -> tuple[str, str]:
        """
        Internal method to scrape the title and content from a given URL.
        Includes robust error handling and encoding detection.
        """
        try:
            response = requests.get(url, timeout=10)
            response.encoding = chardet.detect(response.content)['encoding'] or 'utf-8'

            if response.status_code != 200:
                print(
                    f"Failed to fetch {url}. Status code: {response.status_code}")
                return None, None

            soup = BeautifulSoup(response.text, "html.parser")
            
            for unwanted in soup.find_all("div", attrs={"data-component": "links-block"}):
                unwanted.decompose()

            EXCLUSION_CLASSES = ['article-footer', 'site-footer', 'sidebar', 'advertisement', 'ad-banner', 'related-articles', 'footnote', 'caption', 'author-bio']
            for class_name in EXCLUSION_CLASSES:
                for unwanted_element in soup.find_all(class_=class_name):
                    unwanted_element.decompose()

            title_tag = soup.find("h1")
            title = title_tag.text.strip() if title_tag else "No Title Found"
            
            content = []
            body = soup.find("article")
            if body:
                for paragraph in body.find_all("p"):
                    text = self._filter_necessary_text(paragraph)
                    content.append(text)

            if not body:
                for paragraph in soup.find_all("p", limit=10):
                    text = self._filter_necessary_text(paragraph)
                    content.append(text)
                    
            if not content:
                content = ["No Content Found"]

            return title, " ".join(content).strip()
        
        except requests.exceptions.Timeout:
             print(f"Error: Timeout fetching URL: {url}")
             return "", ""
        except Exception as e:
            print(f"Error occurred during scraping {url}: {e}")
            return "", ""

    def fetch_article_info(self):
        """
        Loads the clean data, generates unique article IDs, scrapes details for
        each unique URL, and merges the data back into the DataFrame.
        """
        if self.data is None and os.path.exists(Constants.CLEAN_DATA_FILE):
            self.data = pd.read_csv(Constants.CLEAN_DATA_FILE)
        elif self.data is None:
            print("Clean data not found. Please run clean_data() first.")
            return

        print("Starting article detail fetching...")

        data = self.data.copy()
        data.dropna(subset=['url'], inplace=True)
        
        data['article_id'] = data.groupby('url').ngroup()
        unique_articles = data[['url', 'article_id']].drop_duplicates().reset_index(drop=True)
        total_unique = len(unique_articles)
        
        article_ids = []
        titles = []
        contents = []
        urls = []
        for _, row in unique_articles.iterrows():
            url = row['url']
            article_id = row['article_id']
            
            print(f"Fetching article {article_id + 1}/{total_unique}: {url}")
            title, content = self._get_article_details(url)
            
            article_ids.append(article_id)
            titles.append(title)
            contents.append(content)
            urls.append(url)

        article_info_df = pd.DataFrame({
            'article_id': article_ids,
            'article_title': titles,
            'article_content': contents,
            'url': urls
        })

        data = pd.merge(data, article_info_df, on=['article_id', 'url'], how='left')
        
        self.data = data
        
        self._export_data(article_info_df, Constants.ARTICLES_INFO_FILE)
        self._export_data(self.data, Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO)
        print("Article detail fetching complete.")
        
        return self.data

    def get_data(self):
        """Returns the current internal DataFrame."""
        if self.data is None:
            if os.path.exists(Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO):
                self.data = pd.read_csv(Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO)
        return self.data
