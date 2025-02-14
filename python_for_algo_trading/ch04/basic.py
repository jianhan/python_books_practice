from turtle import title
import pandas as pd
import numpy as np

raw = pd.read_csv(
    "./ch04/pyalgo_eikon_eod_data.csv", index_col=0, parse_dates=True
).dropna()

data = pd.DataFrame(raw["EUR="])
data.rename(columns={"EUR=": "price"}, inplace=True)

# calculate mean
data["SMA1"] = data["price"].rolling(window=42).mean()
data["SMA2"] = data["price"].rolling(window=252).mean()

from pylab import mpl, plt

print(plt.style.available)

plt.style.use("seaborn-v0_8-pastel")
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["font.family"] = "Arial"

# data.plot(title="EUR/USD | 42 & 252 SMA", figsize=(10, 6))


data["position"] = np.where(data["SMA1"] > data["SMA2"], 1, -1)
data.dropna(inplace=True)
# data["position"].plot(ylim=(-1.1, 1.1), title="marketing position", figsize=(10, 6))

# calculate performance of strategy

data["returns"] = np.log(data["price"] / data["price"].shift(1))
data["strategy"] = data["position"].shift(1) * data["returns"]

data[["returns", "strategy"]].sum().apply(np.exp)

# plt.show()

# print(raw.info())
# print(raw.head())
# print(data.head())
print(data.info())
# print(data.head(10))
print(data[1:5])
