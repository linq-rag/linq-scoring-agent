"""
LINQ Scoring Agent - Dataset Loading Module

This module handles loading and filtering of earnings call transcripts from the
FinancialANN_merged dataset. It provides functions for date-based filtering,
specific transcript retrieval, and dataset preparation for analysis.

Key Features:
- Integration with Hugging Face datasets
- Date-based filtering for quarterly analysis
- Specific transcript retrieval by ticker and date
- Environment-based authentication handling
"""

import os
from datetime import datetime, timedelta
from typing import Dict

from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file for API authentication
load_dotenv()


def get_dataset(start_date: str):
    """
    Load and filter earnings call dataset for a specific date range.
    
    This function retrieves earnings call transcripts from the FinancialANN_merged
    dataset and filters them to a 60-day window starting from the specified date.
    This is useful for quarterly analysis where earnings calls cluster around
    specific periods.
    
    Args:
        start_date: Start date in "YYYY-MM-DD" format for filtering transcripts
        
    Returns:
        Filtered dataset containing only earnings calls within the date range
        
    Note:
        Requires HF_TOKEN environment variable for Hugging Face authentication
    """
    dataset = load_dataset(
        "Linq-AI-Research/FinancialANN_merged",
        token=os.getenv("HF_TOKEN"),
        split="train"
    )
    _start_date = datetime.strptime(start_date, "%Y-%m-%d")
    _end_date = _start_date + timedelta(days=60)  # 60-day window for quarterly coverage

    def filter_by_date_and_ticker(dict_example: Dict):
        """Filter function to select earnings calls within date range."""
        event_date = datetime.strptime(dict_example['event_start_at_et'], "%Y-%m-%d %H:%M:%S.%f")
        return (
                _start_date <= event_date <= _end_date
                and dict_example['type'] == "earnings_call"
        )

    filtered_dataset = dataset.filter(filter_by_date_and_ticker)

    return filtered_dataset


def get_transcript(ticker: str, target_date: datetime) -> str:
    """
    Retrieve a specific earnings call transcript by ticker and date.
    
    This function searches the dataset for a specific company's earnings call
    transcript on a particular date. It's useful for targeted analysis of
    individual company events.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        target_date: Exact date of the earnings call
        
    Returns:
        Text content of the matching earnings call transcript
        
    Raises:
        IndexError: If no matching transcript is found
        
    Note:
        Currently filters for transcripts from 2022 onwards
    """
    # Load the full dataset
    dataset = load_dataset(
        "Linq-AI-Research/FinancialANN_merged",
        token=os.getenv("HF_TOKEN"),
        split="train"
    )

    # Convert target_date to date object for comparison
    target_date = target_date.date()

    def filter_by_date_and_ticker(dict_example: Dict):
        """Filter function to select earnings calls from 2022 onwards."""
        event_date = datetime.strptime(dict_example['event_start_at_et'], "%Y-%m-%d %H:%M:%S.%f")
        return (
                datetime.strptime("2022-01-01", "%Y-%m-%d") <= event_date
                and dict_example['type'] == "earnings_call"
        )

    dataset = dataset.filter(filter_by_date_and_ticker)

    # Find matching transcript using efficient list comprehension
    texts = [
        entry["text"]
        for entry in dataset
        if entry["ticker"] == ticker
           and entry["type"] == "earnings_call"
           and datetime.strptime(entry["event_start_at_et"], "%Y-%m-%d %H:%M:%S.%f").date() == target_date
    ]

    return texts[0]


if __name__ == "__main__":
    """
    Generate comprehensive ticker list from multiple quarters.
    
    This script processes quarterly datasets from 2022-2024 to extract
    all unique tickers for analysis planning and coverage assessment.
    """
    from datasets import concatenate_datasets

    # Define quarterly start dates for comprehensive coverage
    start_dates = [
        "2022-01-01",
        "2022-04-01",
        "2022-07-01",
        "2022-10-01",
        "2023-01-01",
        "2023-04-01",
        "2023-07-01",
        "2023-10-01",
        "2024-01-01",
        "2024-04-01",
        "2024-07-01",
        "2024-10-01",
    ]
    
    # Concatenate all quarterly datasets and extract unique tickers
    for start_date in start_dates:
        dataset = concatenate_datasets([get_dataset(start_date) for start_date in start_dates])
        ticker_list = dataset.unique('ticker')
        for ticker in ticker_list:
            print(ticker, file=open(f"ticker_list.txt", "a"))
