"""
LINQ Scoring Agent - Original Data Analysis Module

This module analyzes original theme analysis data to extract sentiment statistics
by ticker symbol. It processes filtered theme output data to calculate comprehensive
sentiment metrics including averages, distributions, and counts.

Key Features:
- Ticker-based sentiment score aggregation
- Comprehensive statistical analysis (mean, std, counts)
- Sentiment distribution analysis (positive/negative/neutral)
- Percentage-based sentiment breakdown
- Support for multiple target tickers

The analysis helps evaluate the original sentiment extraction performance
and provides baseline metrics for comparison with bias testing results.
"""

import json
from collections import defaultdict
from typing import Dict, List

import numpy as np

# Anonymous company names for testing
COMPANY_NAMES = [
    "Company A",
    "Company B",
    "Company C",
    "Company D",
    "Company E",
    "Company F",
    "Company G",
    "Company H",
    "Company I",
    "Company J"
]

# Mapping of target tickers to their real company names
TICKER_TO_COMPANY = {
    "SYPR": "Sypris Solutions",
    "BWMN": "Bowman Consulting",
    "OTIS": "Otis Worldwide",
    "ADSK": "Autodesk"
}


def analyze_original_data(file_path: str) -> Dict:
    """
    Analyze original theme data to extract sentiment statistics by ticker.
    
    This function processes a JSONL file containing theme analysis results,
    extracts sentiment scores for each ticker, and calculates comprehensive
    statistics including averages, standard deviations, and sentiment distributions.
    
    Args:
        file_path: Path to JSONL file containing original theme analysis data
        
    Returns:
        Dictionary mapping ticker symbols to their sentiment statistics:
        - avg_sentiment: Average sentiment score across all quotes
        - std_sentiment: Standard deviation of sentiment scores
        - total_quotes: Total number of filtered quotes
        - positive_count: Number of positive sentiment quotes
        - negative_count: Number of negative sentiment quotes  
        - neutral_count: Number of neutral sentiment quotes
        
    The function focuses on target tickers defined in TICKER_TO_COMPANY
    and processes filtered_theme_output data from each record.
    """
    # Dictionary to store sentiment scores by ticker
    ticker_sentiments = defaultdict(list)
    
    # Read and process the data file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            
            # Extract ticker symbol from custom_id
            if custom_id.startswith("task-"):
                ticker = custom_id.split("-")[1]
                if ticker in ["SYPR", "BWMN", "OTIS", "ADSK"]:
                    filtered_output = data.get("filtered_theme_output", {})
                    sentiment_scores = filtered_output.get("sentiment_scores", [])
                    
                    if sentiment_scores:
                        ticker_sentiments[ticker].extend(sentiment_scores)
    
    # Calculate comprehensive statistics
    results = {}
    for ticker, sentiments in ticker_sentiments.items():
        sentiments = np.array(sentiments)
        results[ticker] = {
            'avg_sentiment': np.mean(sentiments),
            'std_sentiment': np.std(sentiments),
            'total_quotes': len(sentiments),
            'positive_count': np.sum(sentiments > 0),
            'negative_count': np.sum(sentiments < 0),
            'neutral_count': np.sum(sentiments == 0)
        }
    
    return results


def main():
    """
    Main function for executing original data sentiment analysis.
    
    This function analyzes the original theme data file to extract and display
    comprehensive sentiment statistics for each target ticker. It provides
    detailed breakdowns of sentiment distributions and percentages.
    """
    file_path = "/Users/junekwon/Desktop/Projects/linq-scoring-agent/data/final/data/2022_1Q/22_1q_theme_oil_and_gas.jsonl"
    results = analyze_original_data(file_path)
    
    print("Original Data Analysis Results:")
    print("-" * 50)
    for ticker, stats in results.items():
        print(f"\nTicker: {ticker} ({TICKER_TO_COMPANY[ticker]})")
        print(f"Average Sentiment Score: {stats['avg_sentiment']:.3f}")
        print(f"Sentiment Score Standard Deviation: {stats['std_sentiment']:.3f}")
        print(f"Total Number of Quotes: {stats['total_quotes']}")
        print(f"Positive Quotes Count: {stats['positive_count']}")
        print(f"Negative Quotes Count: {stats['negative_count']}")
        print(f"Neutral Quotes Count: {stats['neutral_count']}")
        print(f"Positive Percentage: {(stats['positive_count'] / stats['total_quotes'] * 100):.1f}%")
        print(f"Negative Percentage: {(stats['negative_count'] / stats['total_quotes'] * 100):.1f}%")
        print(f"Neutral Percentage: {(stats['neutral_count'] / stats['total_quotes'] * 100):.1f}%")


if __name__ == "__main__":
    main() 