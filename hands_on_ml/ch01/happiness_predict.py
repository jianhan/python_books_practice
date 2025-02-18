import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat["Life satisfaction"].values

# visualize the data
lifesat.plot(x="GDP per capita (USD)", y="Life satisfaction", kind="scatter", grid=True)
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# select a linear model
model = LinearRegression()
model.fit(X, y)

# make predictions
X_new = [[37_655.2]]
print(model.predict(X_new))
