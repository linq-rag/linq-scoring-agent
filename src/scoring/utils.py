"""
LINQ Scoring Agent - Utility Functions Module

This module provides essential utility functions for text processing, data manipulation,
and API interactions used throughout the LINQ scoring pipeline. It includes functions
for transcript segmentation, company information retrieval, and ticker set operations.

Key Features:
- Text preprocessing and sentence tokenization
- Transcript and list chunking for parallel processing
- Company name resolution via Financial Modeling Prep API
- Historical ticker data filtering and intersection
- Retry decorators for robust API calls
"""

import logging
import os
from datetime import datetime
from typing import Any, List, Set

import dotenv
import pandas as pd
import requests
from nltk import sent_tokenize
from tenacity import retry, stop_after_attempt, wait_fixed

from src.scoring._default import DEFAULT_EMPTY_PARSED_COMPLETION

logger = logging.getLogger(__name__)

# Load environment variables for API keys
dotenv.load_dotenv()


def handle_max_retries(retry_state):
    """
    Handle retry exhaustion by logging the final error.
    
    This callback function is triggered when maximum retry attempts are reached.
    It logs only the final error message to avoid spam while providing visibility
    into persistent failures.
    
    Args:
        retry_state: Tenacity retry state object containing error information
        
    Returns:
        Default empty parsed completion object for graceful degradation
    """
    last_exception = retry_state.outcome.exception()
    if last_exception:
        logging.error(f"Task {retry_state.args[0]} failed after max retries: {str(last_exception)[:20]}")
    return DEFAULT_EMPTY_PARSED_COMPLETION


def retry_fetch(wait_seconds: int, max_retries: int):
    """
    Create a reusable retry decorator for asynchronous API functions.
    
    This decorator factory provides consistent retry behavior across different
    API calls with configurable wait time and maximum attempts.
    
    Args:
        wait_seconds: Number of seconds to wait between retry attempts
        max_retries: Maximum number of retry attempts before giving up
        
    Returns:
        Tenacity retry decorator configured with specified parameters
    """
    return retry(
        stop=stop_after_attempt(max_retries),  # Retry any exception up to max_retries times
        wait=wait_fixed(wait_seconds),  # Wait specified seconds between retries
        retry_error_callback=handle_max_retries  # Log only the last error
    )


def get_sentences(text: str) -> List[str]:
    """
    Tokenize text into individual sentences using NLTK.
    
    This function processes multi-line text by first splitting on newlines,
    then applying NLTK's sentence tokenizer to each line. This approach
    preserves document structure while ensuring proper sentence boundaries.
    
    Args:
        text: Input text to be tokenized into sentences
        
    Returns:
        List of sentences extracted from the input text
        
    Note:
        Requires NLTK punkt tokenizer data to be downloaded
    """
    text_sentences = text.split("\n")  # Split text into lines first
    sentences = []
    for sentence in text_sentences:
        # Apply sentence tokenization to each line
        sentences.extend(sent_tokenize(sentence))

    return sentences


def split_transcript_into_n(text: str, n: int) -> List[str]:
    """
    Split earnings call transcript into n roughly equal parts.
    
    This function divides a transcript by line breaks into n parts for
    parallel processing. It ensures roughly equal distribution of content
    while maintaining line boundaries.
    
    Args:
        text: Full transcript text to be split
        n: Number of parts to split into (must be >= 1)
        
    Returns:
        List of n text chunks, each containing roughly equal content
        
    Raises:
        ValueError: If n < 1
        
    Note:
        Any remaining lines after equal division are appended to the last part
    """
    if n < 2:
        if n == 1:
            return [text]
        else:
            raise ValueError("The number of parts (n) must be 1 or greater.")

    sections = text.split("\n")
    total_length = len(sections)
    split_size = total_length // n

    # Create n equal parts
    parts = [sections[i * split_size:(i + 1) * split_size] for i in range(n)]
    
    # Add any remaining sections to the last part
    remaining_sections = sections[n * split_size:]
    if remaining_sections:
        parts[-1].extend(remaining_sections)

    return ["\n".join(part) for part in parts]


def split_list_into_n(lst: List[Any], n: int) -> List[List[Any]]:
    """
    Split a list into n roughly equal sublists.
    
    This utility function divides any list into n parts with roughly equal
    element counts. It filters out empty sublists that may result from
    splitting small lists.
    
    Args:
        lst: The list to be split into parts
        n: Number of sublists to create
        
    Returns:
        List of sublists, each containing roughly equal elements from input
        
    Note:
        Returns fewer than n sublists if the input list is very small
    """
    len_list = len(lst)  # Total length of the input list
    return [lst[i * len_list // n: (i + 1) * len_list // n] for i in range(n) if lst[i * len_list // n: (i + 1) * len_list // n]]


def get_company_name(ticker: str) -> str:
    """
    Retrieve company name from ticker symbol using Financial Modeling Prep API.
    
    This function fetches the official company name for a given stock ticker
    symbol. It provides fallback behavior by returning the ticker itself
    if the API call fails or returns no data.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        Company name if successfully retrieved, otherwise the ticker symbol
        
    Note:
        Requires FMP_API_KEY environment variable to be set
    """
    api_key = os.getenv("FMP_API_KEY")
    url = f'https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0].get('companyName', 'Company name not found')
        else:
            return 'No data found for the given ticker symbol'
    else:
        logger.info(f'Error while retrieving company name for ticker {ticker}: {response.status_code}')
        return ticker


def get_ticker_set(
        start_date: datetime,
        end_date: datetime,
        filename: str = "./data/historical_component.csv"
) -> Set:
    """
    Extract common ticker subset from historical S&P 500 component data.
    
    This function finds tickers that were consistently part of the S&P 500
    throughout the specified date range. It's useful for ensuring analysis
    uses only stable index components during the period of interest.
    
    Args:
        start_date: Beginning of the date range for filtering
        end_date: End of the date range for filtering
        filename: Path to CSV file with historical component data
        
    Returns:
        Set of ticker symbols common across all dates in the range
        
    Raises:
        FileNotFoundError: If the specified CSV file doesn't exist
        ValueError: If no data found in range or required columns missing
        
    Note:
        CSV file must have 'date' as index and 'tickers' column with
        comma-separated ticker symbols
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} not found.")

    # Load historical component data with date parsing
    dataframe = pd.read_csv(filename, index_col='date', parse_dates=True)

    # Filter data to specified date range
    filtered_df = dataframe[(dataframe.index >= start_date) & (dataframe.index <= end_date)]

    print(filtered_df)

    if filtered_df.empty:
        raise ValueError("No data found in the specified date range.")

    if 'tickers' not in filtered_df.columns:
        raise ValueError("The required 'tickers' column is missing in the CSV file.")

    # Find intersection of all ticker sets in the date range
    tickers_sets = [set(row.split(',')) for row in filtered_df['tickers']]
    common_tickers = set.intersection(*tickers_sets)
    print(common_tickers)
    print(len(common_tickers))
    return {t.strip() for t in common_tickers}