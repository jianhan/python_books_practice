import pandas as pd
import numpy as np
from darts import TimeSeries
from pathlib import Path
import yfinance as yf
import matplotlib.pyplot as plt
from darts.models import TBATS, AutoARIMA, ExponentialSmoothing, Theta, LinearRegressionModel, NBEATSModel, Prophet, RegressionModel, NaiveSeasonal, NaiveDrift, RandomForest, RNNModel
from darts.metrics import mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset, SunspotsDataset

def download_data():
    file_path = Path("darts/crypto.csv")
    if file_path.exists():
        return pd.read_csv(file_path)

    df = yf.download("BTC-USD", interval="1d", period='6y', multi_level_index=False)
    df.to_csv(file_path)
    return df

if __name__ == '__main__':

    data = download_data()
    # data.plot(x='Date', y='Close')
    # plt.title('BTC-USD Closing Price')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.show()

    # print(data.describe())
    # data.info()

    series = TimeSeries.from_dataframe(df=data, time_col='Date', value_cols='Close')
    train, val = series.split_before(pd.Timestamp("20240501"))
    # train.plot(label="training")
    # val.plot(label="validation")
    # plt.show()
    # raise Exception('ERR')


    # print(series.columns)
    # raise Exception('ERR')

    # def eval_model(model):
    #     model.fit(train)
    #     forecast = model.predict(len(val))
    #     print(f"model {model} obtains MAPE: {mape(val, forecast):.2f}%")


    # eval_model(ExponentialSmoothing())
    # eval_model(TBATS())
    # eval_model(AutoARIMA())
    # eval_model(Theta())

    # thetas = 2 - np.linspace(-10, 10, 50)

    # best_mape = float("inf")
    # best_theta = 0

    # for theta in thetas:
    #     model = Theta(theta)
    #     model.fit(train)
    #     pred_theta = model.predict(len(val))
    #     res = mape(val, pred_theta)

    #     if res < best_mape:
    #         best_mape = res
    #         best_theta = theta

    # print(best_theta)

    # best_theta_model = Theta(8)
    # best_theta_model.fit(train)
    # pred_best_theta = best_theta_model.predict(len(val))
    # print(f"model AutoARIMA obtains MAPE: {mape(val, pred_best_theta):.2f}%")

    # series = AirPassengersDataset().load()
    # print(series.shape)
    # raise Exception('ERR')

    # Read data:
    # series = AirPassengersDataset().load()

    # Create training and validation sets:
    # train, val = series.split_after(pd.Timestamp("19590101"))

    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    # transformer = Scaler()
    # train_transformed = transformer.fit_transform(train)
    # val_transformed = transformer.transform(val)
    # series_transformed = transformer.transform(series)
    # print(series_transformed.pd_dataframe().head())
    # raise Exception('ERR')

    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)
    series_transformed = transformer.transform(series)

    # create month and year covariate series
    # year_series = datetime_attribute_timeseries(
    #     pd.date_range(start=series.start_time(), freq=series.freq_str, periods=1800),
    #     attribute="year",
    #     one_hot=False,
    # )

    # year_series = Scaler().fit_transform(year_series)
    # month_series = datetime_attribute_timeseries(
    #     year_series, attribute="month", one_hot=True
    # )
    # covariates = year_series.stack(month_series)
    # cov_train, cov_val = covariates.split_after(pd.Timestamp("20240501"))

    my_model = RNNModel(
        model="LSTM",
        hidden_dim=20,
        dropout=0,
        batch_size=16,
        n_epochs=300,
        optimizer_kwargs={"lr": 1e-3},
        model_name="Air_RNN",
        log_tensorboard=True,
        random_state=42,
        training_length=20,
        input_chunk_length=14,
        force_reset=True,
        save_checkpoints=True,
    )

    my_model.fit(
        train_transformed,
        val_series=val_transformed,
        verbose=True,
    )

    pred_series = my_model.predict(len(val))
    plt.figure(figsize=(8, 5))
    series_transformed.plot(label="actual")
    pred_series.plot(label="forecast")
    plt.title(f"MAPE: {mape(pred_series, val_transformed):.2f}%")
    plt.legend()

    # print(f"Lowest MAPE is: {mape(val, pred_best_theta):.2f}, with theta = {best_theta}.")
    # train.plot(label="train")
    # val.plot(label="true")
    # pred.plot(label="prediction")
    # plt.legend()
    # plt.show()

