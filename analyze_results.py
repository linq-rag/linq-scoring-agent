"""
LINQ Scoring Agent - Results Analysis Module

This module provides comprehensive analysis tools for evaluating the performance
and consistency of the LINQ scoring pipeline. It analyzes sentiment scores,
quote similarity, and common patterns across different company analyses.

Key Features:
- Cross-company sentiment comparison
- Quote similarity analysis using Jaccard similarity
- Common quote pattern detection
- Statistical analysis of scoring consistency
- Anonymous vs. real company name comparison
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

# Mapping of ticker symbols to company names for analysis
TICKER_TO_COMPANY = {
    "SYPR": "Sypris Solutions",
    "BWMN": "Bowman Consulting",
    "OTIS": "Otis Worldwide",
    "ADSK": "Autodesk"
}


def calculate_quote_similarity(quotes_list: List[List[str]]) -> float:
    """
    Calculate average Jaccard similarity between quote sets across companies.
    
    This function measures how similar the extracted quotes are across different
    company analyses for the same theme. High similarity indicates consistent
    extraction behavior, while low similarity may indicate company-specific
    content extraction.
    
    Args:
        quotes_list: List of quote lists, one per company analysis
        
    Returns:
        Average Jaccard similarity coefficient (0.0 to 1.0)
        
    Note:
        Jaccard similarity = |intersection| / |union|
        Returns 0.0 if no valid comparisons can be made
    """
    if not quotes_list:
        return 0.0
    
    total_similarity = 0
    count = 0
    
    # Compare each pair of quote lists using Jaccard similarity
    for i in range(len(quotes_list)):
        for j in range(i + 1, len(quotes_list)):
            set1 = set(quotes_list[i])
            set2 = set(quotes_list[j])
            
            # Calculate Jaccard similarity coefficient
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            if union > 0:
                similarity = intersection / union
                total_similarity += similarity
                count += 1
    
    return total_similarity / count if count > 0 else 0.0


def analyze_common_quotes(quotes_list: List[List[str]], sentiment_scores_list: List[List[int]]) -> List[Dict]:
    """
    Analyze sentiment patterns for quotes appearing across all companies.
    
    This function identifies quotes that appear in analyses of all companies
    and examines how sentiment scores vary for these common quotes. This helps
    identify potential biases or inconsistencies in sentiment scoring.
    
    Args:
        quotes_list: List of quote lists, one per company
        sentiment_scores_list: List of sentiment score lists, parallel to quotes
        
    Returns:
        List of dictionaries containing quote text, average sentiment,
        and sentiment standard deviation for each common quote
        
    Note:
        Only quotes appearing in ALL company analyses are considered
    """
    if not quotes_list or not sentiment_scores_list:
        return []
    
    # Find quotes that appear in all company analyses
    common_quotes = set(quotes_list[0])
    for quotes in quotes_list[1:]:
        common_quotes.intersection_update(quotes)
    
    if not common_quotes:
        return []
    
    # Collect sentiment scores for each common quote across companies
    quote_analysis = []
    for quote in common_quotes:
        quote_sentiments = []
        for company_quotes, company_sentiments in zip(quotes_list, sentiment_scores_list):
            for q, s in zip(company_quotes, company_sentiments):
                if q == quote:
                    quote_sentiments.append(s)
                    break
        
        quote_sentiments = np.array(quote_sentiments)
        quote_analysis.append({
            'quote': quote,
            'avg_sentiment': np.mean(quote_sentiments),
            'std_sentiment': np.std(quote_sentiments)
        })
    
    return quote_analysis


def analyze_ticker_results(ticker: str, results_dir: str) -> Dict:
    """
    Comprehensive analysis of all results for a specific ticker symbol.
    
    This function aggregates and analyzes all available results for a given
    ticker, separating anonymous company analyses from real company name
    analyses to detect potential naming bias effects.
    
    Args:
        ticker: Stock ticker symbol to analyze
        results_dir: Directory containing JSON result files
        
    Returns:
        Dictionary containing:
        - anonymous: Statistics for anonymous company analyses
        - real: Statistics for real company name analysis
        - quote_similarity: Overall quote similarity metric
        - common_quotes_analysis: Analysis of quotes common to all results
        
    Note:
        Expects files named with pattern: {ticker}_{company_name}.json
        for real names and other patterns for anonymous analyses
    """
    anonymous_sentiment_means = []
    real_sentiment = None
    quotes_list = []
    sentiment_scores_list = []
    
    # Collect all result files for the ticker
    for filename in os.listdir(results_dir):
        if filename.startswith(ticker):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)
                sentiment_mean = np.mean(data['sentiment_scores'])
                
                # Categorize as real company name vs. anonymous analysis
                if filename.endswith(f"{TICKER_TO_COMPANY[ticker].replace(' ', '_')}.json"):
                    real_sentiment = sentiment_mean
                else:
                    anonymous_sentiment_means.append(sentiment_mean)
                
                quotes_list.append(data['quotes'])
                sentiment_scores_list.append(data['sentiment_scores'])
    
    # Calculate comprehensive statistics
    anonymous_sentiment_means = np.array(anonymous_sentiment_means)
    
    # Measure quote extraction consistency
    quote_similarity = calculate_quote_similarity(quotes_list)
    
    # Analyze patterns in commonly extracted quotes
    common_quotes_analysis = analyze_common_quotes(quotes_list, sentiment_scores_list)
    
    return {
        'anonymous': {
            'avg_sentiment': np.mean(anonymous_sentiment_means),
            'std_sentiment': np.std(anonymous_sentiment_means),
            'sentiment_means': anonymous_sentiment_means.tolist()
        },
        'real': {
            'sentiment': real_sentiment
        },
        'quote_similarity': quote_similarity,
        'common_quotes_analysis': common_quotes_analysis
    }


def main():
    """
    Main analysis function for evaluating LINQ scoring consistency.
    
    This function performs comprehensive analysis across all test tickers
    to evaluate the consistency and reliability of the scoring pipeline.
    It provides insights into potential biases and extraction patterns.
    """
    results_dir = 'test'
    tickers = ['ADSK', 'BWMN', 'OTIS', 'SYPR']
    
    # Perform analysis for each ticker
    results = {}
    for ticker in tickers:
        results[ticker] = analyze_ticker_results(ticker, results_dir)
    
    # Display comprehensive analysis results
    print("Ticker Analysis Results:")
    print("-" * 50)
    for ticker, stats in results.items():
        print(f"\nTicker: {ticker} ({TICKER_TO_COMPANY[ticker]})")
        
        print("\nAnonymous Company Analysis:")
        print(f"Average Sentiment Score: {stats['anonymous']['avg_sentiment']:.3f}")
        print(f"Sentiment Standard Deviation: {stats['anonymous']['std_sentiment']:.3f}")
        print(f"Individual Company Averages: {[f'{x:.3f}' for x in stats['anonymous']['sentiment_means']]}")
        
        print("\nReal Company Name Analysis:")
        print(f"Sentiment Score: {stats['real']['sentiment']:.3f}")
        
        print(f"\nQuote Similarity Index: {stats['quote_similarity']:.3f}")
        
        # Note: Common quotes analysis output is commented out for brevity
        # Uncomment the following lines to display detailed quote analysis
        # print("\nCommon Quotes Analysis:")
        # for i, quote_analysis in enumerate(stats['common_quotes_analysis'], 1):
        #     print(f"\nQuote {i}:")
        #     print(f"Content: {quote_analysis['quote']}")
        #     print(f"Average Sentiment: {quote_analysis['avg_sentiment']:.3f}")
        #     print(f"Sentiment Std Dev: {quote_analysis['std_sentiment']:.3f}")


if __name__ == "__main__":
    main() 