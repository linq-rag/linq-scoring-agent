"""
LINQ Scoring Agent - Analysis Reproduction Module

This module reproduces sentiment analysis experiments using different company names
to test for potential biases in the LLM-based scoring system. It processes the same
quotes with various company identifiers to evaluate scoring consistency.

Key Features:
- Batch processing of target tickers with anonymized company names
- Systematic bias testing using real vs. anonymous company names
- Comparative analysis output generation
- Asynchronous processing for efficiency

The experiment helps identify whether company name recognition affects sentiment
scoring, which is crucial for ensuring objective financial analysis.
"""

import asyncio
import json
import os
from typing import Dict, List

from src.scoring.fetch import fetch_filtered_output
from src.scoring.utils import get_company_name

# Anonymous company names for bias testing
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

# Target tickers for bias analysis
TARGET_TICKERS = [
    "SYPR", 
    "BWMN", 
    "OTIS", 
    "ADSK"
]


async def process_ticker(ticker: str, company_name: str, quotes: List[Dict], theme: str, output_dir: str):
    """
    Process a single ticker with a specific company name for sentiment analysis.
    
    This function takes extracted quotes from a ticker and processes them through
    the filtering pipeline using a given company name. This allows testing whether
    different company names affect sentiment scoring for identical content.
    
    Args:
        ticker: Stock ticker symbol being analyzed
        company_name: Company name to use in the analysis (real or anonymous)
        quotes: List of extracted quotes to be filtered and scored
        theme: Theme description for context in filtering
        output_dir: Directory to save analysis results
        
    The results are saved as JSON files with naming pattern: {ticker}_{company_name}.json
    This enables easy comparison of results across different company name variants.
    """
    try:
        filter_result, _ = await fetch_filtered_output(
            company_name=company_name,
            quotes=quotes,
            extraction_type="theme",
            fetch_type="openai",
            theme=theme,
            num_split=15,
        )
        
        # Save results with company name in filename for comparison
        output_path = os.path.join(output_dir, f"{ticker}_{company_name.replace(' ', '_')}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filter_result.model_dump(), f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error processing {ticker} with {company_name}: {str(e)}")


async def main():
    """
    Main function for executing the bias reproduction experiment.
    
    This function loads previously extracted quotes from the specified input file
    and processes them with multiple company names (both real and anonymous) to
    test for potential naming bias in sentiment analysis.
    
    The experiment design:
    1. Load extracted quotes for target tickers
    2. Process each ticker's quotes with its real company name
    3. Process the same quotes with multiple anonymous company names
    4. Save all results for comparative analysis
    
    This setup enables statistical analysis of scoring consistency across
    different company name variations.
    """
    input_file = "/Users/junekwon/Desktop/Projects/linq-scoring-agent/data/final/data/2022_1Q/22_1q_theme_oil_and_gas.jsonl"
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process input file to extract quotes and metadata
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            
            # Extract ticker symbol from custom_id format
            if custom_id.startswith("task-"):
                ticker = custom_id.split("-")[1]
                if ticker in TARGET_TICKERS:
                    extracted_output = data.get("extracted_theme_output", {})
                    quotes = extracted_output.get("quotes", [])
                    theme = data.get("theme", "")
                    
                    # Process with multiple company names for bias testing
                    tasks = []
                    # Include real company name as baseline
                    tasks.append(process_ticker(ticker, TICKER_TO_COMPANY[ticker], quotes, theme, output_dir))
                    # Include anonymous company names for comparison
                    for company_name in COMPANY_NAMES:
                        tasks.append(process_ticker(ticker, company_name, quotes, theme, output_dir))
                    
                    # Execute all variations in parallel
                    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())