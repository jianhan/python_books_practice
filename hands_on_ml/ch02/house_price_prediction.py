from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.linear_model import LinearRegression

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
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

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
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:])
print(ordinal_encoder.categories_)

one_hot_encoder = OneHotEncoder()
housing_cat_oh = one_hot_encoder.fit_transform(housing_cat)
print("--------------------One Hot Encoding----------------------------")
print(housing_cat_oh.toarray())

print("--------------------House Num----------------------------")
print(housing_num.head())

std_scaler = StandardScaler()
housing_num_scaled = std_scaler.fit_transform(housing_num)
min_max_scaler = MinMaxScaler()
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

print("--------------------housing_num before scale----------------------------")
print(housing_num)
print("--------------------housing_num_scaled----------------------------")
print(housing_num_scaled)
print("--------------------housing_num_min_max_scaled----------------------------")
print(housing_num_min_max_scaled)

# pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
housing_num_prepared = num_pipeline.fit_transform(housing_num)
print("--------------------housing_num_prepared----------------------------")
print(housing_num_prepared[:2].round(2))
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared,
    columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index,
)
print("--------------------df_housing_num_prepared----------------------------")
print(df_housing_num_prepared)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
)

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_exclude=np.number)),
)
housing_prepared = preprocessing.fit_transform(housing)
print("--------------------preprocessing----------------------------")
print(type(housing_prepared))


# refactored code
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


X = np.array([[10, 2], [20, 5], [30, 6]])
print(column_ratio(X))
