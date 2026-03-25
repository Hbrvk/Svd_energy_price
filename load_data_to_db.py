import pandas as pd
import sqlite3


def load_data_table(path: str, name: str) -> None:
    df = pd.read_csv(path)
    connection = sqlite3.connect("data/sql.db")
    _ = df.to_sql(name=name, con=connection, if_exists="replace")
    connection.close()


load_data_table("data/electricity_dah_prices.csv", "electricity_dah_prices")
