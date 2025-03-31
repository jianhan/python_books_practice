import numpy as np
import pandas as pd
from loguru import logger

df = pd.DataFrame([10, 20, 30, 40], columns=["numbers"], index=["a", "b", "c", "d"])
logger.info(f"df is {df}")

# operations
index = df.index
logger.info(f"index is {index}")

columns = df.columns
logger.info(f"columns is {columns}")

values = df.values
logger.info(f"values is {values}")

c = df.loc["c"]
logger.info(f"c is {c}")

a_d = df.loc[["a", "d"]]
logger.info(f"a_d is {a_d}")

one_three = df.iloc[1:3]
logger.info(f"one_three is {one_three}")

sum = df.sum()
logger.info(f"sum is {sum}")

lambda_demo = df.apply(lambda x: x * 2)
logger.info(f"lambda_demo is {lambda_demo}")

# enlarge
df["floats"] = (1.5, 2.5, 3.5, 4.5)
logger.info(f"df is {df}")

# define a new column
df["name"] = pd.DataFrame(["alice", "bob", "charlie", "david"], index=df.index)
logger.info(f"df is {df}")

# append with index ignore
df_appended_ignore_index = df._append(
    {"numbers": 50, "floats": 5.5, "name": "eve"}, ignore_index=True
)
logger.info(f"df_appended_ignore_index is {df_appended_ignore_index}")

df_appended_with_index = df._append(
    pd.DataFrame(
        {"numbers": 50, "floats": 5.5, "name": "eve"},
        index=[
            "y",
        ],
    ),
)
logger.info(f"df_appended_with_index is {df_appended_with_index}")

# append without value will assigned to NaN
df_with_missing_data = df._append(pd.DataFrame({"name": "JACK"}, index=["z"]))
logger.info(f"df_with_missing_data is {df_with_missing_data}")

np.random.seed(42)
a = np.random.standard_normal((9, 4))
logger.info(f"a is {a}")

df = pd.DataFrame(a)
logger.info(f"df is {df}")

# change column names
df.columns = ["A", "B", "C", "D"]
a_mean = df["A"].mean()
logger.info(f"a_mean is {a_mean}")

# add date
dates = pd.date_range("2019-01-01", periods=9, freq="M")
logger.info(f"dates is {dates}")
df.index = dates
logger.info(f"df is {df}")

# useful functions
df.info()
print(df.describe())
print(df.sum())
