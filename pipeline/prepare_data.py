
import os, requests, zipfile
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
UCI_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"

def download_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "electricity.zip"
    if zip_path.exists():
        print("Already downloaded.")
    else:
        print("Downloading UCI Electricity dataset (~13MB)...")
        r = requests.get(UCI_URL, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)

def load_raw():
    txt_path = RAW_DIR / "LD2011_2014.txt"
    df = pd.read_csv(txt_path, sep=";", index_col=0, parse_dates=True, decimal=",")
    df.index.name = "timestamp"
    df.columns = [f"client_{i}" for i in range(len(df.columns))]
    return df

def engineer_features(series):
    df = series.to_frame(name="load")
    df["time_idx"] = np.arange(len(df))
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["lag_24h"] = df["load"].shift(24)
    df["lag_168h"] = df["load"].shift(168)
    df["series_id"] = "aggregate"
    return df.dropna()

def split_and_save(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    n = len(df)
    test_len = val_len = 30 * 24
    train = df.iloc[: n - val_len - test_len]
    val   = df.iloc[n - val_len - test_len : n - test_len]
    test  = df.iloc[n - test_len :]
    train.to_parquet(PROCESSED_DIR / "train.parquet")
    val.to_parquet(PROCESSED_DIR / "val.parquet")
    test.to_parquet(PROCESSED_DIR / "test.parquet")
    print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

def main():
    download_data()
    print("Loading...")
    df_raw = load_raw()
    aggregate = df_raw.sum(axis=1).rename("load").resample("h").mean()
    print("Engineering features...")
    df_features = engineer_features(aggregate)
    split_and_save(df_features)
    print("Data pipeline complete.")

if __name__ == "__main__":
    main()
