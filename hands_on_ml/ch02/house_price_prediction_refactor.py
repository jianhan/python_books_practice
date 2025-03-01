
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
    FunctionTransformer,
)
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )

def log_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out='one-to-one'),
        StandardScaler(),
    )

def cat_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(exist_ok=True, parents=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),StandardScaler()) 
log_pipeline = log_pipeline()
cat_pipeline = cat_pipeline()

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population","households", "median_income"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],remainder=default_num_pipeline)

housing = load_housing_data()
housing["income_cat"] = pd.cut(
    housing["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5]
)
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, random_state=42, stratify=housing["income_cat"]
)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
housing_train = strat_train_set.drop("median_house_value", axis=1)
housing_train_labels = strat_train_set["median_house_value"].copy()


lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing_train, housing_train_labels)
housing_predictions = lin_reg.predict(housing_train)
lin_mse = mean_squared_error(housing_train_labels, housing_predictions)

print('------------linear regression------------')
print(housing_predictions[:5].round(-2))
print(np.sqrt(lin_mse))

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing_train, housing_train_labels)
housing_predictions = tree_reg.predict(housing_train)

print('------------decision tree------------')
print(housing_predictions[:5].round(-2))
print(tree_reg.score(housing_train, housing_train_labels))

# print('------------cross validation------------')
# tree_rmses = -cross_val_score(tree_reg, housing_train, housing_train_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(tree_rmses).describe())

# print('------------random forest------------')
# forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
# forest_rmses = -cross_val_score(forest_reg, housing_train, housing_train_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(forest_rmses).describe())

# print('------------support vector regression------------')
# svr_reg = make_pipeline(preprocessing, SVR(kernel="poly"))
# svr_mse = -cross_val_score(svr_reg, housing_train, housing_train_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(svr_mse).describe())

# print('------------lasso------------')
# lasso_reg = make_pipeline(preprocessing, Lasso())
# lasso_mse = -cross_val_score(lasso_reg, housing_train, housing_train_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(lasso_mse).describe())

# print('------------ridge------------')
# ridge_reg = make_pipeline(preprocessing, Ridge())
# ridge_mse = -cross_val_score(ridge_reg, housing_train, housing_train_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(ridge_mse).describe())

print('------------knn------------')
knn_reg = make_pipeline(preprocessing, KNeighborsRegressor(n_neighbors=9))
knn_mse = -cross_val_score(knn_reg, housing_train, housing_train_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(knn_mse).describe())


# print('------------mlp------------')
# mlp_reg = make_pipeline(preprocessing, MLPRegressor(random_state=42, max_iter=500))
# mlp_mse = -cross_val_score(mlp_reg, housing_train, housing_train_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(mlp_mse).describe())