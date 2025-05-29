"""
LINQ Scoring Agent - Main Processing Pipeline

This module orchestrates the end-to-end pipeline for extracting and filtering 
financial themes from earnings call transcripts. It processes both overall 
sentiment and theme-specific analyses asynchronously.

Key Features:
- Asynchronous processing of multiple themes
- Resume capability for interrupted runs
- Batch processing with progress tracking
- Error handling and recovery
"""

import asyncio
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, Literal

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from src.scoring.dataset import get_dataset
from src.scoring.fetch import fetch_extracted_output, fetch_filtered_output
from src.scoring.utils import get_company_name

# Set random seed for reproducibility
random.seed(2025)


async def _main(
    file_name: str,
    theme_dict: Dict[str, str],
    fetch_type: Literal["groq", "furiosa", "openai"],
    example: Dict,
    processed_tickers: Dict[str, set],
) -> (Dict[str, dict], Dict[str, dict]):
    """
    Process a single earnings call transcript through extraction and filtering pipelines.
    
    This function handles both overall sentiment analysis and theme-specific analyses 
    for a given company's earnings call transcript. It implements skip logic to avoid 
    reprocessing already completed tickers.
    
    Args:
        file_name: Identifier for the processing batch (e.g., "22_3Q_THEME")
        theme_dict: Mapping of theme keys to theme descriptions
        fetch_type: API provider to use for LLM calls ("groq", "furiosa", "openai")
        example: Single transcript record with ticker, date, and text
        processed_tickers: Set of already processed tickers per theme/overall
        
    Returns:
        Tuple containing:
        - overall_result: Dict with overall sentiment analysis results
        - theme_results: Dict mapping theme keys to theme-specific results
        
    Note:
        Uses separate async functions for parallel processing of overall and theme pipelines
    """
    ticker = example["ticker"]
    event_date_str = example["event_start_at_et"]  # Format: "2022-01-01 00:00:00.000000"
    event_date = datetime.strptime(event_date_str, "%Y-%m-%d %H:%M:%S.%f")
    date = event_date.strftime("%y-%m-%d")
    transcript = example["text"]
    company_name = get_company_name(ticker)
    custom_id = f"task-{ticker}-{date}-{file_name}"

    overall_result = {}
    theme_results = {}

    # Process overall sentiment analysis pipeline
    async def process_overall():
        """Extract and filter overall sentiment from transcript."""
        nonlocal overall_result
        if ticker in processed_tickers["overall"]:
            return  # Skip if ticker has already been processed
        try:
            # Extract overall quotes from transcript
            extract_overall_result, extract_overall_usage = await fetch_extracted_output(
                company_name=company_name,
                text=transcript,
                extraction_type="overall",
                fetch_type="openai",
                theme=None,
                num_split=20,
            )

            # Filter extracted quotes for relevance
            filter_overall_result, filter_overall_usage = await fetch_filtered_output(
                company_name=company_name,
                quotes=extract_overall_result.quotes,
                extraction_type="overall",
                fetch_type=fetch_type,
                theme=None,
                num_split=15,
            )

            overall_result = {
                "custom_id": custom_id,
                "extracted_overall_output": extract_overall_result.model_dump(),
                "filtered_overall_output": filter_overall_result.model_dump(),
                "extraction_overall_usages": extract_overall_usage,
                "filtering_overall_usages": filter_overall_usage,
            }
        except Exception as e:
            overall_result = {
                "custom_id": custom_id,
                "error": f"[OVERALL PIPELINE ERROR] {str(e)}",
            }

    # Process theme-specific analysis pipeline
    async def process_theme(theme_key: str, theme_str: str):
        """Extract and filter theme-specific content from transcript."""
        if ticker in processed_tickers[theme_key]:
            return  # Skip if ticker has already been processed
        try:
            # Extract theme-specific quotes from transcript
            extract_theme_result, extract_theme_usage = await fetch_extracted_output(
                company_name=company_name,
                text=transcript,
                extraction_type="theme",
                fetch_type=fetch_type,
                theme=theme_str,
                num_split=20,
            )
            
            # Filter extracted quotes for theme relevance
            filter_theme_result, filter_theme_usage = await fetch_filtered_output(
                company_name=company_name,
                quotes=extract_theme_result.quotes,
                extraction_type="theme",
                fetch_type=fetch_type,
                theme=theme_str,
                num_split=15,
            )

            theme_results[theme_key] = {
                "custom_id": custom_id,
                "extracted_theme_output": extract_theme_result.model_dump(),
                "filtered_theme_output": filter_theme_result.model_dump(),
                "extraction_theme_usages": extract_theme_usage,
                "filtering_theme_usages": filter_theme_usage,
            }
        except Exception as e:
            theme_results[theme_key] = {
                "custom_id": custom_id,
                "error": str(e),
            }

    # Execute pipelines in parallel (currently only overall is enabled)
    # tasks = [process_overall()] + [process_theme(theme_key, theme_str) for theme_key, theme_str in theme_dict.items()]
    tasks = [process_overall()]
    # tasks = [process_theme(theme_key, theme_str) for theme_key, theme_str in theme_dict.items()]
    await tqdm_asyncio.gather(*tasks, desc="Processing overall and themes", leave=False)
    time.sleep(1)  # Rate limiting

    return overall_result, theme_results


async def main(
    file_name: str,
    theme_dict: Dict[str, str],
    start_date: str,
    output_dir: str,
    fetch_type: Literal["groq", "furiosa", "openai"],
):
    """
    Main orchestration function for batch processing earnings call transcripts.
    
    This function implements a complete pipeline with resume functionality:
    1. Loads existing output files to determine processed tickers
    2. Filters dataset to only unprocessed transcripts
    3. Processes each transcript through extraction and filtering
    4. Saves results incrementally to JSONL files
    
    Args:
        file_name: Batch identifier for output files
        theme_dict: Mapping of theme keys to descriptions
        start_date: Filter transcripts from this date onwards (YYYY-MM-DD)
        output_dir: Directory to save output JSONL files
        fetch_type: LLM API provider to use
        
    The function creates separate output files for overall results and each theme,
    allowing for independent processing and analysis of different aspects.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tracking for processed tickers
    processed_tickers: Dict[str, set] = {"overall": set()}
    for theme_key in theme_dict.keys():
        processed_tickers[theme_key] = set()

    # Load already processed tickers from overall output file
    overall_out_path = os.path.join(output_dir, f"{file_name.lower()}_overall.jsonl")
    if os.path.exists(overall_out_path):
        with open(overall_out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    custom_id = data.get("custom_id", "")
                    if custom_id.startswith("task-"):
                        parts = custom_id.split("-")
                        if len(parts) > 1:
                            existing_ticker = parts[1]
                            processed_tickers["overall"].add(existing_ticker)
                except json.JSONDecodeError:
                    continue

    # Load already processed tickers from each theme output file
    for theme_key in theme_dict.keys():
        out_path = os.path.join(output_dir, f"{file_name.lower()}_{theme_key}.jsonl")
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        custom_id = data.get("custom_id", "")
                        if custom_id.startswith("task-"):
                            parts = custom_id.split("-")
                            if len(parts) > 1:
                                existing_ticker = parts[1]
                                processed_tickers[theme_key].add(existing_ticker)
                    except json.JSONDecodeError:
                        continue

    # Load dataset and filter unprocessed tickers
    dataset = get_dataset(start_date=start_date)

    print(dataset)
    print("Total unique tickers:", len(set(dataset.unique('ticker'))))
    time.sleep(1)

    # Filter dataset to only include unprocessed tickers
    # Currently only filtering for overall pipeline
    filtered_dataset = [
        example for example in dataset if any(
            example["ticker"] not in processed_tickers[key] for key in ["overall"]
        )
    ]

    print(f"Total dataset length: {len(dataset)}")
    print(f"Filtered dataset length (not fully processed): {len(filtered_dataset)}")

    # Process each transcript sequentially with progress tracking
    for example in tqdm(filtered_dataset, total=len(filtered_dataset)):
        overall_result, theme_results = await _main(
            file_name=file_name,
            theme_dict=theme_dict,
            example=example,
            processed_tickers=processed_tickers,
            fetch_type=fetch_type,
        )

        # Save overall results incrementally
        if overall_result:
            overall_out_path = os.path.join(output_dir, f"{file_name.lower()}_overall.jsonl")
            with open(overall_out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(overall_result, ensure_ascii=False) + "\n")

        # Save theme results incrementally
        for theme_key, result in theme_results.items():
            out_path = os.path.join(output_dir, f"{file_name.lower()}_{theme_key}.jsonl")
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    """
    Entry point for batch processing different quarterly themes.
    
    Each execution block processes a specific quarter's themes using the Groq API.
    The date ranges correspond to earnings call periods for each quarter.
    """
    from src.scoring.themes import (
        THEME_2022_1Q,
        THEME_2022_2Q,
        THEME_2022_3Q,
        THEME_2022_4Q,
        THEME_2023_1Q,
        THEME_2023_2Q,
        THEME_2023_3Q,
    )

    # Process 2022 Q3 themes
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2022_3Q)}
    asyncio.run(
        main(
            file_name="22_3Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2022-10-01", 
            output_dir="./data/4o-mini/2022_3Q-groq", 
            fetch_type="groq")
    )
    
    # Process 2022 Q4 themes
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2022_4Q)}
    asyncio.run(
        main(
            file_name="22_4Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2023-01-01", 
            output_dir="./data/4o-mini/2022_4Q-groq", 
            fetch_type="groq")
    )
    
    # Process 2023 Q1 themes
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2023_1Q)}
    asyncio.run(
        main(
            file_name="23_1Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2023-04-01", 
            output_dir="./data/4o-mini/2023_1Q-groq", 
            fetch_type="groq")
    )
    
    # Process 2023 Q2 themes
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2023_2Q)}
    asyncio.run(
        main(
            file_name="23_2Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2023-07-01", 
            output_dir="./data/4o-mini/2023_2Q-groq", 
            fetch_type="groq")
    )
    
    # Process 2023 Q3 themes
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2023_3Q)}
    asyncio.run(
        main(
            file_name="23_3Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2023-10-01", 
            output_dir="./data/4o-mini/2023_3Q-groq", 
            fetch_type="groq")
    )
    