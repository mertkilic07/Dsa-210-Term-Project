import pandas as pd
import numpy as np
import os

INPUT_FILE = "../csv_files/bitcoin_price.csv"
OUTPUT_FILE = "bitcoin_modified.csv"


def process_local_price_file():
    print(f"{INPUT_FILE}...")

    if not os.path.exists(INPUT_FILE):
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"{len(df)}")
    except Exception as e:
        print(f"{e}")
        return

    df.columns = [col.capitalize().strip() for col in df.columns]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.set_index('Date').sort_index()
    else:
        print(f"{df.columns.tolist()}")
        return


    df['Daily_Return'] = df['Close'].pct_change()

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    df['Volatility_7d'] = df['Log_Return'].rolling(window=7).std()

    df['Momentum_3d'] = df['Log_Return'].rolling(window=3).mean()

    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['Trend_Score'] = df['Close'] / df['MA_7']

    df['Next_Day_Return'] = df['Daily_Return'].shift(-1)
    df['Target_Direction'] = (df['Next_Day_Return'] > 0).astype(int)

    df_clean = df.dropna()

    df_clean.to_csv(OUTPUT_FILE)



if __name__ == "__main__":
    process_local_price_file()