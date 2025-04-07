from datetime import datetime

import pandas as pd

from src.scoring.price import get_daily_stock_price_table

class Stock:
    ticker: str
    start_date: datetime
    end_date: datetime

    def __init__(self, ticker: str, start_date: datetime, end_date: datetime):
        self.ticker: ticker = ticker
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.data: pd.DataFrame = get_daily_stock_price_table(ticker, start_date, end_date)

    def get_market_return_by_date(
            self, target_date: datetime
    ):
        return self.data.loc[target_date, "Return"]
    