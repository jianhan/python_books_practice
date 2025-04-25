import matplotlib as mpl
import yfinance as yf

tickers = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD"]
crypto_prices = yf.download(
    tickers=tickers,
    period="2y",
    interval="1d",
    group_by="ticker",
)

close_prices = crypto_prices.loc[:, (tickers, "Close")]
close_prices.columns = close_prices.columns.droplevel(1)

print(close_prices)

close_prices.plot(
    title="Cryptocurrency Prices",
    figsize=(12, 6),
    fontsize=12,
    grid=True,
    color=["blue", "orange", "green", "red"],
    subplots=True,
)

mpl.pyplot.show()
