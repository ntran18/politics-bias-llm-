import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import os
import sys
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)

from src.article_fetcher import ArticleFetcher

# Mock response structure for requests.get
class MockResponse:
    """Mock class for the requests.Response object."""
    def __init__(self, content, status_code=200, encoding='utf-8'):
        self.content = content
        self.status_code = status_code
        self.encoding = encoding
        self.text = content.decode(self.encoding, errors='ignore')

    @property
    def content(self):
        # chardet.detect is usually called on self.content
        return self._content

    @content.setter
    def content(self, value):
        self._content = value
    
    @property
    def url(self):
        return "http://test.com/article"

class ArticleFetcherTest(unittest.TestCase):
    
    # Define a clean test setup before each test run
    def setUp(self):
        # Initialize the fetcher with dummy paths
        self.test_output_dir = 'test_data_output'
        self.fetcher = ArticleFetcher(input_file='dummy_input.csv', output_dir=self.test_output_dir)
    
    def tearDown(self):
        # Clean up any created test directories or files
        if os.path.exists(self.test_output_dir):
            for file in os.listdir(self.test_output_dir):
                os.remove(os.path.join(self.test_output_dir, file))
            os.rmdir(self.test_output_dir)
        
    @patch('os.makedirs')
    @patch('os.path.exists', return_value=False)
    def test_init_creates_output_dir(self, mock_exists, mock_makedirs):
        """Test that the constructor ensures the output directory exists."""
        _ = ArticleFetcher(output_dir=self.test_output_dir)
        mock_makedirs.assert_called_once_with(self.test_output_dir)

    # --- Test _get_article_details (Scraping Logic) ---

    @patch('requests.get')
    @patch('chardet.detect', return_value={'encoding': 'utf-8', 'confidence': 0.99})
    def test_get_article_details_success(self, mock_chardet, mock_get):
        """Test successful fetching and parsing of title and content."""
        test_url = "http://example.com/test-article"
        
        # Mock HTML response with h1 and article tags
        html_content = b"""
        <!DOCTYPE html>
        <html>
        <body>
            <header class="article-meta">Ignore this header</header>
            <h1>The Great Test Article Title</h1>
            <article>
                <p>This is the first paragraph of the article body.</p>
                <p>This is the second paragraph with important info.</p>
            </article>
        </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)
        
        title, content = self.fetcher._get_article_details(test_url)
        
        self.assertEqual(title, "The Great Test Article Title")
        # Check that the two main article paragraphs are included and concatenated
        expected_content = "This is the first paragraph of the article body. This is the second paragraph with important info."
        self.assertEqual(content, expected_content)
        mock_get.assert_called_once_with(test_url, timeout=10)
        
    @patch('requests.get')
    @patch('chardet.detect', return_value={'encoding': 'utf-8', 'confidence': 0.99})
    def test_get_article_details_success_without_paragraph_with_footnote_class(self, mock_chardet, mock_get):
        """Test successful fetching and parsing of title and content."""
        test_url = "http://example.com/test-article"
        
        # Mock HTML response with h1 and article tags
        html_content = b"""
        <!DOCTYPE html>
        <html>
        <body>
            <header class="article-meta">Ignore this header</header>
            <h1>The Great Test Article Title</h1>
            <article>
                <p>This is the first paragraph of the article body.</p>
                <p>This is the second paragraph with important info.</p>
                <p class="footnote">This should be removed before scraping.</p>
            </article>
        </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)
        
        title, content = self.fetcher._get_article_details(test_url)
        
        self.assertEqual(title, "The Great Test Article Title")
        # Check that the two main article paragraphs are included and concatenated
        expected_content = "This is the first paragraph of the article body. This is the second paragraph with important info."
        self.assertEqual(content, expected_content)
        mock_get.assert_called_once_with(test_url, timeout=10)
        
    @patch('requests.get')
    @patch('chardet.detect', return_value={'encoding': 'utf-8', 'confidence': 0.99})
    def test_get_article_details_success_without_footer_div(self, mock_chardet, mock_get):
        """Test successful fetching and parsing of title and content."""
        test_url = "http://example.com/test-article"
        
        # Mock HTML response with h1 and article tags
        html_content = b"""
        <!DOCTYPE html>
        <html>
        <body>
            <header class="article-meta">Ignore this header</header>
            <h1>The Great Test Article Title</h1>
            <article>
                <p>This is the first paragraph of the article body.</p>
                <p>This is the second paragraph with important info.</p>
                <p class="article-footer">This should be removed before scraping.</p>
            </article>
            <div class="article-meta">
                <div class="author-bio">
                    <p>This author bio should be excluded.</p>
                </div>
            </div>
        </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)
        
        title, content = self.fetcher._get_article_details(test_url)
        
        self.assertEqual(title, "The Great Test Article Title")
        # Check that the two main article paragraphs are included and concatenated
        expected_content = "This is the first paragraph of the article body. This is the second paragraph with important info."
        self.assertEqual(content, expected_content)
        mock_get.assert_called_once_with(test_url, timeout=10)

    @patch('requests.get')
    def test_get_article_details_404_failure(self, mock_get):
        """Test handling of non-200 HTTP status code."""
        test_url = "http://example.com/404"
        mock_get.return_value = MockResponse(b"Not Found", status_code=404)
        
        # Suppress the print output for clean testing
        with patch('builtins.print'):
            title, content = self.fetcher._get_article_details(test_url)
        
        self.assertIsNone(title)
        self.assertIsNone(content)

    @patch('requests.get', side_effect=requests.exceptions.Timeout)
    def test_get_article_details_timeout(self, mock_get):
        """Test handling of request timeout exception."""
        test_url = "http://example.com/timeout"
        
        with patch('builtins.print'):
            title, content = self.fetcher._get_article_details(test_url)
            
        self.assertIsNone(title)
        self.assertIsNone(content)

    @patch('requests.get')
    @patch('chardet.detect', return_value={'encoding': 'utf-8', 'confidence': 0.99})
    def test_get_article_details_fallback_content(self, mock_chardet, mock_get):
        """Test content extraction when only P tags are present (no <article> tag)."""
        test_url = "http://example.com/blog-post"
        
        # Mock HTML response with h1 and p tags in the body (fallback logic)
        html_content = b"""
        <!DOCTYPE html>
        <html>
        <body>
            <h1>Simple Blog Post</h1>
            <p>This is paragraph one.</p>
            <p>This is paragraph two.</p>
            <p class="footnote">This should be excluded by the 'footnote' check.</p>
            <p>This is paragraph three.</p>
        </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)
        
        title, content = self.fetcher._get_article_details(test_url)
        
        self.assertEqual(title, "Simple Blog Post")
        # Checks that the logic limits to 10 paragraphs and excludes footer class
        expected_content = "This is paragraph one. This is paragraph two. This is paragraph three."
        self.assertEqual(content, expected_content)


    # --- Test clean_data ---

    @patch('os.path.exists', return_value=True)
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.makedirs')
    def test_clean_data_processing(self, mock_makedirs, mock_to_csv, mock_read_csv, mock_exists):
        """Test that clean_data loads, renames, and exports the data."""
        
        # 1. Setup mock data that simulates the input CSV structure
        mock_input_data = pd.DataFrame({
            'Answer.age': [30, 45],
            'Answer.articleNumber': [1, 2],
            'Answer.batch': [101, 101],
            'Answer.bias-question': ['A', 'B'],
            'Answer.country': ['USA', 'CAN'],
            'Answer.gender': ['M', 'F'],
            'Answer.language1': ['en', 'fr'],
            'Answer.newsOutlet': ['FOX', 'BBC'],
            'Answer.politics': ['Cons', 'Lib'],
            'Answer.url': ['url1', 'url2'],
            'Other.Col': ['ignore', 'ignore']
        })
        mock_read_csv.return_value = mock_input_data
        
        # 2. Run the method under test
        with patch('builtins.print'):
             self.fetcher.clean_data()
        
        # 3. Assertions
        mock_read_csv.assert_called_once_with('dummy_input.csv')
        mock_to_csv.assert_called_once()
        
        # Verify the columns of the final self.data DataFrame
        expected_cols = ['age', 'articleNumber', 'batch', 'bias-question', 'country', 
                         'gender', 'language', 'source', 'politics', 'url']
        
        self.assertIsInstance(self.fetcher.data, pd.DataFrame)
        self.assertListEqual(list(self.fetcher.data.columns), expected_cols)

    @patch('pandas.read_csv', side_effect=FileNotFoundError)
    def test_clean_data_file_not_found(self, mock_read_csv):
        """Test handling of missing input file."""
        with patch('builtins.print') as mock_print:
            self.fetcher.clean_data()
            mock_print.assert_called_with("Error: Input file dummy_input.csv not found.")

    # --- Test fetch_article_info (Integration) ---

    @patch.object(ArticleFetcher, '_get_article_details')
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.exists', return_value=True)
    def test_fetch_article_info_integration(self, mock_exists, mock_to_csv, mock_read_csv, mock_get_details):
        """Test the unique URL processing and merging in fetch_article_info."""
        
        # 1. Setup mock clean data (2 unique URLs, 3 total rows)
        mock_clean_data = pd.DataFrame({
            'url': ['url_a', 'url_b', 'url_a'],  # url_a is repeated
            'data_col': [1, 2, 3]
        })
        
        # 2. Set up the mocks for fetching
        # Mock _get_article_details to return a title/content pair for each unique URL
        mock_get_details.side_effect = [
            ("Title A", "Content A"),  # For 'url_a' (article_id 0)
            ("Title B", "Content B"),  # For 'url_b' (article_id 1)
        ]
        
        # Set the mock data directly on the fetcher instance
        self.fetcher.data = mock_clean_data
        
        # 3. Run the method under test
        with patch('builtins.print'):
            result_df = self.fetcher.fetch_article_info()
            
        # 4. Assertions
        
        # Check that scraping was only called for unique URLs
        self.assertEqual(mock_get_details.call_count, 2)
        
        # Verify the shape and content of the result DataFrame
        self.assertIsNotNone(result_df)
        self.assertEqual(len(result_df), 3) # Should retain all original rows
        
        # Check that the article details are correctly mapped back
        self.assertEqual(result_df.loc[0, 'article_title'], 'Title A')
        self.assertEqual(result_df.loc[0, 'article_content'], 'Content A')
        self.assertEqual(result_df.loc[1, 'article_title'], 'Title B')
        self.assertEqual(result_df.loc[2, 'article_title'], 'Title A') # Check repeated merge

        # Verify that the final DataFrame was exported
        mock_to_csv.assert_called_once_with(self.fetcher.articles_info_path, index=False)


if __name__ == '__main__':
    # Since we cannot easily import the ArticleFetcher, we skip main execution 
    # to prevent errors if the user runs this file directly without the other one.
    # In a proper setup, the user would run 'python -m unittest test_article_fetcher.py'
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
