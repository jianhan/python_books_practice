# https://unit8.com/resources/darts-time-series-made-easy-in-python/

from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ExponentialSmoothing

df = pd.read_csv("darts/AirPassengers.csv")
# df.plot()
# plt.show()

print(df.info())

series = TimeSeries.from_dataframe(df, "Month", "#Passengers")
train, test = series.split_before(pd.Timestamp("19580101"))
model = ExponentialSmoothing()
model.fit(train)
predictions = model.predict(len(test))
series.plot(label="Actual")
predictions.plot(label="Predictions", lw=3)
# plt.legend()
# plt.show()
