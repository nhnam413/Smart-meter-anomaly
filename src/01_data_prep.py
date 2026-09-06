"""
01_data_prep.py - Pipeline doc, lam sach, phan chia Train/Demo va tao du lieu bat thuong gia lap.
"""

import os
import sys
import numpy as np
import pandas as pd

from config import (
    RAW_DATA_PATH, DATA_DIR,
    TRAIN_HOURLY_PATH, DEMO_STREAM_PATH,
    TARGET_COL, VOLTAGE_DROP_THRESHOLD,
    setup_encoding,
)

setup_encoding()

#Doc du lieu tho UCI, xu ly missing data va aggregate theo gio (1h).
def load_and_clean_raw_data(file_path: str) -> pd.DataFrame:  
    if not os.path.exists(file_path):
        print(f"Loi: Khong tim thay file tai: {file_path}")
        sys.exit(1)

    print("[1/3] Dang doc va lam sach du lieu...")
    df = pd.read_csv(file_path, sep=";", na_values=["?"], low_memory=False)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["Date", "Time"], inplace=True)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all").ffill().bfill()
    df = df[~df.index.duplicated(keep="first")]

    df_hourly = df.resample("h").mean().dropna()
    print(f"  -> Gom nhom 1 gio hoan tat: {len(df_hourly):,} dong.")
    return df_hourly

#Thong ke co ban tren tap du lieu Train.
def run_eda(df: pd.DataFrame) -> dict:
    v_diff = df["Voltage"].diff().dropna()
    q1, q3 = float(v_diff.quantile(0.25)), float(v_diff.quantile(0.75))
    iqr = q3 - q1

    profile = {
        "voltage_mean": float(df["Voltage"].mean()),
        "voltage_diff_min": float(v_diff.min()),
        "voltage_diff_max": float(v_diff.max()),
        "power_mean": float(df[TARGET_COL].mean()),
        "iqr_voltage_diff": iqr,
    }

    print("  -> Thong ke co ban tap Train:")
    print(f"     - Dien ap trung binh: {profile['voltage_mean']:.1f} V")
    print(f"     - Bien dong dien ap (min/max): {profile['voltage_diff_min']:+.1f} V / {profile['voltage_diff_max']:+.1f} V")
    print(f"     - Cong suat tieu thu trung binh: {profile['power_mean']:.2f} kW")
    print(f"     - IQR bien dong dien ap: {iqr:.2f}")
    return profile

#Gia lap 3 dang loi (Power Surge, Voltage Drop, Night Spike) vao tap Demo.
def inject_synthetic_anomalies(df_demo: pd.DataFrame) -> pd.DataFrame:
    print("[3/3] Dang gia lap cac dang loi vao tap Demo...")
    df = df_demo.copy()
    df["is_anomaly"] = 0
    df["anomaly_type"] = "normal"

    rng = np.random.default_rng(42)
    n = len(df)

    # 1. Dot bien cong suat ban ngay 3%
    n_surge = int(n * 0.03)
    day_indices = df[(df.index.hour >= 8) & (df.index.hour <= 22)].index
    surge_idx = rng.choice(day_indices, size=min(n_surge, len(day_indices)), replace=False)
    df.loc[surge_idx, TARGET_COL] *= rng.uniform(3.0, 5.0, size=len(surge_idx))
    df.loc[surge_idx, "is_anomaly"] = 1
    df.loc[surge_idx, "anomaly_type"] = "power_surge"

    # 2. Sut dien ap 3%
    n_drop = int(n * 0.03)
    normal_idx = df[df["is_anomaly"] == 0].index
    drop_idx = rng.choice(normal_idx, size=min(n_drop, len(normal_idx)), replace=False)
    df.loc[drop_idx, "Voltage"] -= rng.uniform(20.0, 40.0, size=len(drop_idx))
    df.loc[drop_idx, "is_anomaly"] = 1
    df.loc[drop_idx, "anomaly_type"] = "voltage_drop"

    # 3. Dot bien dem khuya 2%
    n_night = int(n * 0.02)
    night_idx = df[(df.index.hour >= 1) & (df.index.hour <= 5) & (df["is_anomaly"] == 0)].index
    night_choices = rng.choice(night_idx, size=min(n_night, len(night_idx)), replace=False)
    df.loc[night_choices, TARGET_COL] *= rng.uniform(2.0, 3.5, size=len(night_choices))
    df.loc[night_choices, "is_anomaly"] = 1
    df.loc[night_choices, "anomaly_type"] = "night_spike"

    total_anom = int(df["is_anomaly"].sum())
    print(f"  -> Da tao: {total_anom:,}/{n:,} diem bat thuong ({total_anom/n*100:.1f}%).")
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Doc va lam sach du lieu
    df_hourly = load_and_clean_raw_data(RAW_DATA_PATH)

    # 2. Chia tap Train (80%) va Demo (20%)
    print("[2/3] Phan chia tap Train/Demo va thong ke EDA...")
    split_idx = int(len(df_hourly) * 0.8)
    df_train, df_demo = df_hourly.iloc[:split_idx], df_hourly.iloc[split_idx:]
    print(f"  -> Train: {len(df_train):,} mau | Demo: {len(df_demo):,} mau.")

    run_eda(df_train)

    df_train.to_csv(TRAIN_HOURLY_PATH)
    print(f"  -> Da luu tap Train: {TRAIN_HOURLY_PATH}")

    # 3. Tao loi va luu tap Demo
    df_demo_labeled = inject_synthetic_anomalies(df_demo)
    df_demo_labeled.to_csv(DEMO_STREAM_PATH)
    print(f"  -> Da luu tap Demo Stream: {DEMO_STREAM_PATH}")
    print("Hoan tat chuan bi du lieu.")


if __name__ == "__main__":
    main()
