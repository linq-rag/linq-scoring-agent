import os
from datetime import datetime, timedelta
from typing import Dict

from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from a `.env` file
load_dotenv()


def get_dataset(start_date: str):
    dataset = load_dataset(
        "Linq-AI-Research/FinancialANN_merged",
        token=os.getenv("HF_TOKEN"),
        split="train"
    )
    _start_date = datetime.strptime(start_date, "%Y-%m-%d")
    _end_date = _start_date + timedelta(days=60)

    def filter_by_date_and_ticker(dict_example: Dict):
        event_date = datetime.strptime(dict_example['event_start_at_et'], "%Y-%m-%d %H:%M:%S.%f")
        return (
                _start_date <= event_date <= _end_date
                and dict_example['type'] == "earnings_call"
        )

    filtered_dataset = dataset.filter(filter_by_date_and_ticker)

    return filtered_dataset


def get_transcript(ticker: str, target_date: datetime):
    # Load the dataset
    dataset = load_dataset(
        "Linq-AI-Research/FinancialANN_merged",
        token=os.getenv("HF_TOKEN"),
        split="train"
    )

    # Convert target_date to datetime object for comparison
    target_date = target_date.date()

    def filter_by_date_and_ticker(dict_example: Dict):
        event_date = datetime.strptime(dict_example['event_start_at_et'], "%Y-%m-%d %H:%M:%S.%f")
        return (
                datetime.strptime("2022-01-01", "%Y-%m-%d") <= event_date
                and dict_example['type'] == "earnings_call"
        )

    dataset = dataset.filter(filter_by_date_and_ticker)

    # Filter the dataset using list comprehension for performance
    texts = [
        entry["text"]
        for entry in dataset
        if entry["ticker"] == ticker
           and entry["type"] == "earnings_call"
           and datetime.strptime(entry["event_start_at_et"], "%Y-%m-%d %H:%M:%S.%f").date() == target_date
    ]

    return texts[0]


if __name__ == "__main__":
    from datasets import concatenate_datasets

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
    for start_date in start_dates:
        dataset = concatenate_datasets([get_dataset(start_date) for start_date in start_dates])
        ticker_list = dataset.unique('ticker')
        for ticker in ticker_list:
            print(ticker, file=open(f"ticker_list.txt", "a"))
