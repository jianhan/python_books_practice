from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(exist_ok=True, parents=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


housing = load_housing_data()
housing.info()
print("----------------")
print(housing["ocean_proximity"].value_counts())
print("----------------")
print(housing.describe())

housing["income_cat"] = pd.cut(
    housing["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5]
)

# print("---------------- plot the income categories")
# housing["income_cat"].value_counts().sort_index().plot(kind="bar")
# plt.xlabel("Income category")
# plt.ylabel("Number of districts")
# plt.show()

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, random_state=42, stratify=housing["income_cat"]
)

# drop the income_cat attribute
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# exploration of data
# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
# plt.show()

# -----------------------------------------------------------------------
# housing.plot(
#     kind="scatter",
#     x="longitude",
#     y="latitude",
#     c="median_house_value",
#     legend=True,
#     sharex=False,
#     figsize=(10, 8),
#     grid=True,
#     s=housing["population"] / 100,
#     label="population",
#     cmap=plt.get_cmap("jet"),
#     colorbar=True,
#     alpha=0.2,
# )
# plt.show()

# housing.describe()
# corr_matrix = housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)

print("--------------------imputer.statistics_----------------------------")
print(imputer.statistics_)
# Now you can use this “trained” imputer to transform the training set by replacing missing values with the learned medians:
X = imputer.transform(housing_num)
print("--------------------X----------------------------")
print(X)


# start to encode
ordinal_encoder = OrdinalEncoder()
housing_cat = housing[["ocean_proximity"]]
print("--------------------housing_cat----------------------------")
print(housing_cat.info())
