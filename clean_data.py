import sqlalchemy
import pandas as pd
from typing import Literal
from statsmodels.tsa.stattools import adfuller # pyright: ignore[reportUnknownVariableType,reportMissingTypeStubs]
import datetime as dt

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
