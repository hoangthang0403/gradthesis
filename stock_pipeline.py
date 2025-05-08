from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pyodbc
from vnstock3 import Vnstock
import pandas as pd
import numpy as np
import time
import glob
import os

# Defining a function to fetch all tickers
def get_tickers():
    df = pd.read_csv('/var/tmp/tickers.csv')
    df = df.ticker.unique().tolist()
    return df


def clean_trade_history_files(folder="/var/tmp"):

    files_to_delete = glob.glob(os.path.join(folder, "trade_history*"))
    print("Found files:", files_to_delete)  # Debugging output

    if not files_to_delete:
        print("No matching files found.")
        return

    for file_path in files_to_delete:
        os.remove(file_path)
        print(f"Deleted: {file_path}")

    print(f"Cleanup completed. {len(files_to_delete)} files removed.")


# Defining a function to fetch trade history
def fetching_trade_history(**kwargs):
    current_date = kwargs['_date']
    print(f'Fetching data on {current_date}')
    tickers = get_tickers()

    # Initialize a counter for missing data
    no_missing_data = 0
    for i in range(1598):
        time.sleep(1)
        try:
            # Initializing stock object then fetch trade history data into a DataFrame
            stock = Vnstock().stock(symbol=tickers[i], source='TCBS')
            stock = stock.quote.history(start='2025-01-01', end='2025-03-31')
            stock = stock[stock.time >= '2025-03-26'    ]
            # Adding a ticker column
            stock.insert(0, 'ticker', tickers[i])
            stock.to_csv(f'/var/tmp/trade_history{i}.csv', index=False)

        except:
            print("Missing data of", tickers[i])
            no_missing_data += 1
            continue
    print("Fetching complete!")
    print(f'Total number of missing data on {current_date} is {no_missing_data}')

# Defining a function to fetch trade history
def loading_data(conn):

    trade_history = pd.read_csv('/var/tmp/template.csv')
    trade_history = trade_history.drop(trade_history.index)
    for i in range(1598):
        try:
            df = pd.read_csv(f'/var/tmp/trade_history{i}.csv')
            trade_history = pd.concat([trade_history, df], ignore_index=True)
        except:
            continue
    trade_history["time"] = pd.to_datetime(trade_history["time"])
    trade_history["time"] = trade_history["time"].dt.strftime("%Y-%m-%d")  # Convert datetime to string format
    trade_history["volume"] = trade_history["volume"].astype(int)

    # Convert DataFrame to list of tuples for bulk insert
    values = [tuple(map(lambda x: int(x) if isinstance(x, (np.int64, np.int32)) else x, row)) for row in
              trade_history.to_records(index=False)]

    try:
        server = "host.docker.internal"
        database = "stock_dw"
        username = "hoangthang0403"
        password = "admin"
        driver = "ODBC Driver 18 for SQL Server"
        connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes"
        conn = pyodbc.connect(connection_string)
        print("Database connection successful.")
    except Exception as e:
        print(f"Database connection failed: {e}")

    try:
        cursor = conn.cursor()
        sql = """
           INSERT INTO staging (ticker, [date], [open], high, low, [close], volume)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           """

        cursor.fast_executemany = True
        cursor.executemany(sql, values)

        conn.commit()
        cursor.close()
        conn.close()
        print("Data inserted successfully.")
    except Exception as e:
        print(f"Database insertion failed: {e}")

    clean_trade_history_files()

with DAG(
    dag_id='stock_pipeline',
    start_date=datetime(2025, 3, 25),
    schedule_interval='@once',  # Executing task everyday
    max_active_runs=1,
) as dag:
    fetch_data = PythonOperator(
        task_id='fetch_data',
        python_callable=fetching_trade_history,
        depends_on_past=True,
        op_kwargs={
            "_date": "{{ execution_date.strftime('%Y-%m-%d') }}",
        }
    )

    load_data = PythonOperator(
        task_id='load_data',
        python_callable=loading_data,
        depends_on_past=True
    )

fetch_data >> load_data