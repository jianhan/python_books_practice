import pandas as pd
import numpy as np
from darts import TimeSeries
from pathlib import Path
import yfinance as yf

def download_data():
    file_path = Path("darts/crypto.csv")
    if file_path.exists():
        return pd.read_csv(file_path)

    btc = yf.download("BTC-USD", period="1d", interval="5m")
    btc.to_csv(file_path)
    return btc

ts = TimeSeries.from_dataframe(download_data())

print(ts)
