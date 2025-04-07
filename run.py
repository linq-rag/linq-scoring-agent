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

random.seed(2025)


async def _main(
    file_name: str,
    theme_dict: Dict[str, str],
    fetch_type: Literal["groq", "furiosa", "openai"],
    example: Dict,
    processed_tickers: Dict[str, set],
) -> (Dict[str, dict], Dict[str, dict]):
    """
    Asynchronously processes the overall pipeline and each theme-specific pipeline for a single example.
    Skips already processed tickers and returns separate dictionaries for overall and theme results.
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

    # Asynchronous function for processing the overall pipeline
    async def process_overall():
        nonlocal overall_result
        if ticker in processed_tickers["overall"]:
            return  # Skip if ticker has already been processed
        try:
            extract_overall_result, extract_overall_usage = await fetch_extracted_output(
                company_name=company_name,
                text=transcript,
                extraction_type="overall",
                fetch_type="openai",
                theme=None,
                num_split=20,
            )

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

    # Asynchronous function for processing each theme
    async def process_theme(theme_key: str, theme_str: str):
        if ticker in processed_tickers[theme_key]:
            return  # Skip if ticker has already been processed
        try:
            extract_theme_result, extract_theme_usage = await fetch_extracted_output(
                company_name=company_name,
                text=transcript,
                extraction_type="theme",
                fetch_type=fetch_type,
                theme=theme_str,
                num_split=20,
            )
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

    # tasks = [process_overall()] + [process_theme(theme_key, theme_str) for theme_key, theme_str in theme_dict.items()]
    tasks = [process_overall()]
    # tasks = [process_theme(theme_key, theme_str) for theme_key, theme_str in theme_dict.items()]
    await tqdm_asyncio.gather(*tasks, desc="Processing overall and themes", leave=False)
    time.sleep(1)

    return overall_result, theme_results



async def main(
    file_name: str,
    theme_dict: Dict[str, str],
    start_date: str,
    output_dir: str,
    fetch_type: Literal["groq", "furiosa", "openai"],
):
    os.makedirs(output_dir, exist_ok=True)

    processed_tickers: Dict[str, set] = {"overall": set()}
    for theme_key in theme_dict.keys():
        processed_tickers[theme_key] = set()

    # Process the overall file
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

    # Process each theme file
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

    # --------------------------------------------------------------------------------
    # Step 2: Load dataset and filter unprocessed tickers
    # --------------------------------------------------------------------------------
    dataset = get_dataset(start_date=start_date)#(range(1))  # Load 1 example for testing

    print(dataset)
    print("Total unique tickers:", len(set(dataset.unique('ticker'))))
    time.sleep(1)

    # filtered_dataset = [
    #     example for example in dataset if any(
    #         example["ticker"] not in processed_tickers[key] for key in (["overall"]+ list(theme_dict.keys())
    #     )
    # ]

    # filtered_dataset = [
    #     example for example in dataset if any(
    #         example["ticker"] not in processed_tickers[key] for key in list(theme_dict.keys())
    #     )
    # ]
    filtered_dataset = [
        example for example in dataset if any(
            example["ticker"] not in processed_tickers[key] for key in ["overall"]
        )
    ]

    print(f"Total dataset length: {len(dataset)}")
    print(f"Filtered dataset length (not fully processed): {len(filtered_dataset)}")

    # --------------------------------------------------------------------------------
    # Step 3: Process each example sequentially
    # --------------------------------------------------------------------------------
    for example in tqdm(filtered_dataset, total=len(filtered_dataset)):
        overall_result, theme_results = await _main(
            file_name=file_name,
            theme_dict=theme_dict,
            example=example,
            processed_tickers=processed_tickers,
            fetch_type=fetch_type,
        )

        # Save overall results (if the ticker was not already processed)
        if overall_result:
            overall_out_path = os.path.join(output_dir, f"{file_name.lower()}_overall.jsonl")
            with open(overall_out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(overall_result, ensure_ascii=False) + "\n")

        # Save each theme result
        for theme_key, result in theme_results.items():
            out_path = os.path.join(output_dir, f"{file_name.lower()}_{theme_key}.jsonl")
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    from src.scoring.themes import (
        THEME_2022_1Q,
        THEME_2022_2Q,
        THEME_2022_3Q,
        THEME_2022_4Q,
        THEME_2023_1Q,
        THEME_2023_2Q,
        THEME_2023_3Q,
    )
    
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2022_3Q)}
    asyncio.run(
        main(
            file_name="22_3Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2022-10-01", 
            output_dir="./data/4o-mini/2022_3Q-groq", 
            fetch_type="groq")
    )
    
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2022_4Q)}
    asyncio.run(
        main(
            file_name="22_4Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2023-01-01", 
            output_dir="./data/4o-mini/2022_4Q-groq", 
            fetch_type="groq")
    )
    
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2023_1Q)}
    asyncio.run(
        main(
            file_name="23_1Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2023-04-01", 
            output_dir="./data/4o-mini/2023_1Q-groq", 
            fetch_type="groq")
    )
    
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2023_2Q)}
    asyncio.run(
        main(
            file_name="23_2Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2023-07-01", 
            output_dir="./data/4o-mini/2023_2Q-groq", 
            fetch_type="groq")
    )
    
    _theme_dict = {v.replace(" ", "_").lower(): v for v in sorted(THEME_2023_3Q)}
    asyncio.run(
        main(
            file_name="23_3Q_THEME", 
            theme_dict=_theme_dict,
            start_date="2023-10-01", 
            output_dir="./data/4o-mini/2023_3Q-groq", 
            fetch_type="groq")
    )
    