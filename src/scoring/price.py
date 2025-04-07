from datetime import datetime

import pandas as pd


def get_daily_stock_price_table(ticker: str, start_date: datetime, end_date: datetime):

    dataframe = pd.read_csv("./data/CRSP_DAILY_STOCK_PRICES.csv")

    # Select relevant columns
    selected_columns = ["date", "TICKER", "BIDLO", "ASKHI", "PRC", "BID", "ASK", "OPENPRC", "RET"]
    dataframe = dataframe[selected_columns]

    dataframe.rename(
        columns={
            "TICKER": "Ticker",
            "BIDLO": "Low",
            "ASKHI": "High",
            "PRC": "Close",
            "OPENPRC": "Open",
            "RET": "Return"
        }, inplace=True
    )

    dataframe.set_index("date", inplace=True)

    dataframe.index = pd.to_datetime(dataframe.index)

    dataframe = dataframe[
        (dataframe["Ticker"] == ticker) & (dataframe.index >= start_date) & (dataframe.index <= end_date)]

    return dataframe
