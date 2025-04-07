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

dotenv.load_dotenv()


def handle_max_retries(retry_state):
    """Logs only the last error message after max retries are exhausted."""
    last_exception = retry_state.outcome.exception()
    if last_exception:
        logging.error(f"Task {retry_state.args[0]} failed after max retries: {str(last_exception)[:20]}")
    return DEFAULT_EMPTY_PARSED_COMPLETION


# Create a reusable decorator for retrying async functions
def retry_fetch(wait_seconds: int, max_retries: int):
    return retry(
        stop=stop_after_attempt(max_retries),  # Retry any exception up to 3 times
        wait=wait_fixed(wait_seconds),  # Wait 1 second between retries
        retry_error_callback=handle_max_retries  # Log only the last error
    )


def get_sentences(text: str):
    """
    Tokenizes the given text into sentences using NLTK's `sent_tokenize`.

    Args:
        text (`str`): The text to be tokenized into sentences.

    Returns:
        `List[str]`: A list of sentences extracted from the text.
    """
    text_sentences = text.split("\n")  # Split text into lines
    sentences = []
    for sentence in text_sentences:
        # Tokenize each line into sentences
        sentences.extend(sent_tokenize(sentence))

    return sentences


def split_transcript_into_n(text: str, n: int) -> List[str]:
    if n < 2:
        if n == 1:
            return [text]
        else:
            raise ValueError("The number of parts (n) must be 1 or greater.")

    sections = text.split("\n")
    total_length = len(sections)
    split_size = total_length // n

    parts = [sections[i * split_size:(i + 1) * split_size] for i in range(n)]
    remaining_sections = sections[n * split_size:]
    if remaining_sections:
        parts[-1].extend(remaining_sections)

    return ["\n".join(part) for part in parts]


def split_list_into_n(lst: List[Any], n: int) -> List[List[Any]]:
    """
    Splits a list into `n` roughly equal parts.

    Args:
        lst (`List[Any]`): The list to be split.
        n (`int`): The number of parts to split the list into.

    Returns:
        `List[List[Any]]`: A list of `n` sublists, each containing roughly equal elements from the input list.
    """
    len_list = len(lst)  # Total length of the input list
    return [lst[i * len_list // n: (i + 1) * len_list // n] for i in range(n) if lst[i * len_list // n: (i + 1) * len_list // n]]


def get_company_name(ticker):
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
    Extract the common subset of tickers that appear in all rows of a given date range.

    Parameters
    ----------
    start_date : datetime
        The start date for filtering the data.
    end_date : datetime
        The end date for filtering the data.
    filename : str, optional
        The path to the CSV file containing historical data (default is "./data/historical_component.csv").

    Returns
    -------
    set
        A set of tickers that are common across all rows within the specified date range.

    Raises
    ------
    FileNotFoundError
        If the specified file is not found.
    ValueError
        If the 'tickers' column is missing or the filtered DataFrame is empty.

    Notes
    -----
    - The CSV file should have a 'date' column as its index and a 'tickers' column containing comma-separated tickers.
    - The 'date' column in the CSV must be in a format that can be parsed as datetime.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} not found.")

    # Read the CSV file
    dataframe = pd.read_csv(filename, index_col='date', parse_dates=True)

    # Filter the DataFrame by date range
    filtered_df = dataframe[(dataframe.index >= start_date) & (dataframe.index <= end_date)]

    print(filtered_df)

    if filtered_df.empty:
        raise ValueError("No data found in the specified date range.")

    if 'tickers' not in filtered_df.columns:
        raise ValueError("The required 'tickers' column is missing in the CSV file.")

    # Split tickers in each row and find the common subset
    tickers_sets = [set(row.split(',')) for row in filtered_df['tickers']]
    common_tickers = set.intersection(*tickers_sets)
    print(common_tickers)
    print(len(common_tickers))
    return {t.strip() for t in common_tickers}