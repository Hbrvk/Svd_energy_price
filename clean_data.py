import sqlalchemy
import pandas as pd
import numpy as np
from typing import Literal
from statsmodels.tsa.stattools import adfuller
import datetime as dt
from numpy.linalg import svd
import statsmodels.api as sm
from numpy.typing import ArrayLike, NDArray


data_path = "sqlite:///data/sql.db"
table = "electricity_dah_prices"


def import_data() -> pd.DataFrame:
    engine = sqlalchemy.create_engine(data_path)
    connection = engine.connect()
    df = pd.read_sql_table(table, connection)
    return df


def reframe_df(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.strftime("%m/%d/%Y")
    return df


def select_country(country: Literal["france", "italy", "belgium", "spain", "uk", "germany"],
                   df: pd.DataFrame) -> pd.DataFrame:
    df = df.pivot_table(
        index="date",
        columns="hour",
        values=country
    )
    df.columns = list(range(1,25))
    return df


def interpolate_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate("linear")


def data_pipeline(country: Literal["france", "italy", "belgium", "spain", "uk", "germany"]) -> pd.DataFrame:
    df = import_data()
    df = reframe_df(df)
    df = select_country(country, df)
    df = interpolate_na(df)
    return df

data_pipeline("france").to_csv("data/clean_france.csv")


def clean_weekdays(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index, format="%m/%d/%Y")
    df["day"] = df.index.map(dt.date.weekday)
    df = df[df["day"] <= 4]
    df = df.drop(columns="day")
    return df


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df.loc["2022-04-04", ["8","9"]] = np.nan
    return interpolate_na(df)


def adf_columns(df: pd.DataFrame) -> None:
    for col in df.columns:
        result = adfuller(df[col])[1]
        if result < 0.05:
            print(f"Prices in {col}:00 are stationary throughout the year.")


def make_stationary(df: pd.DataFrame) -> pd.DataFrame:
    df = df - df.shift(1)
    df = df.dropna()
    return df


def demean_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = df[col] - df[col].mean()
    return df


def clean_principal_components(df: pd.DataFrame) -> pd.DataFrame:
    U, s, Vt = svd(df)
    Vt = pd.DataFrame(Vt)
    Vt.index = [f"V_{i}" for i in range(1, 25)]

    return Vt


def get_singular_values(df: pd.DataFrame) -> pd.DataFrame:
    U, s ,Vt = svd(df)
    singular_values = pd.Series(s)
    x = np.square(singular_values) / sum(np.square(singular_values))
    var_series = pd.Series([sum(x[:i]) for i in range(1, len(singular_values) + 1)]).round(3)
    singular_table = pd.concat([singular_values, var_series], axis=1)
    singular_table.columns = ["singular", "var"]

    return singular_table


def regression(x: ArrayLike, matrix: NDArray) -> None:
    model = sm.OLS(x, matrix)
    res = model.fit()
    print(res.summary())