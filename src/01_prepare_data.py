"""
01_prepare_data.py — Tiền xử lý dữ liệu & Phân tích EDA
"""

import os
import sys
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import (
    RAW_DATA_PATH, PROCESSED_DIR, DEMO_DIR, TARGET_COL,
    CONTAMINATION_FILE,
)


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Đọc dữ liệu thô, định dạng ngày tháng và ép kiểu dữ liệu số."""
    df = pd.read_csv(file_path, sep=";", na_values=["?"], low_memory=False)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["Date", "Time"], inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"1. Đã tải dữ liệu thô: {df.shape[0]:,} dòng x {df.shape[1]} cột")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý giá trị trống và trùng lặp chỉ số thời gian."""
    before_nulls = df.isnull().sum().sum()
    df = df.dropna(how="all").ffill().bfill()
    after_nulls = df.isnull().sum().sum()

    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        df = df[~df.index.duplicated(keep="first")]

    print(f"2. Làm sạch dữ liệu: Ô trống {before_nulls:,} -> {after_nulls:,} | Trùng lặp: {dup_count:,}")
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Gom nhóm dữ liệu từ 1 phút thành 1 giờ (lấy trung bình)."""
    df_hourly = df.resample("h").mean().dropna()
    print(f"3. Gom nhóm 1 giờ: {df.shape[0]:,} -> {df_hourly.shape[0]:,} dòng")
    return df_hourly


def run_eda(df_hourly: pd.DataFrame) -> float:
    """Phân tích EDA cơ bản và tính tỷ lệ ngoại lệ (Contamination)."""
    print("\n--- Phân tích EDA ---")
    q1 = df_hourly[TARGET_COL].quantile(0.25)
    q3 = df_hourly[TARGET_COL].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df_hourly[(df_hourly[TARGET_COL] < lower_bound) | (df_hourly[TARGET_COL] > upper_bound)]
    contamination = len(outliers) / len(df_hourly)

    print(f"IQR Outliers: {len(outliers):,} / {len(df_hourly):,} mẫu | Contamination: {contamination*100:.2f}%\n")
    return float(contamination)


def save_contamination(contamination: float) -> None:
    """Lưu tỷ lệ contamination ra file text."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(CONTAMINATION_FILE, "w", encoding="utf-8") as f:
        f.write(str(contamination))


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    demo_ratio: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia tập dữ liệu theo thứ tự thời gian."""
    n = len(df)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))

    df_train = df.iloc[:train_end].copy()
    df_test = df.iloc[train_end:test_end].copy()
    df_demo = df.iloc[test_end:].copy()

    print(f"4. Phân chia tập dữ liệu ({n:,} mẫu): Train={len(df_train):,} | Test={len(df_test):,} | Demo={len(df_demo):,}")
    return df_train, df_test, df_demo


def save_datasets(df_train: pd.DataFrame, df_test: pd.DataFrame, df_demo: pd.DataFrame) -> None:
    """Lưu các tập dữ liệu ra file CSV."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    df_train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    df_test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    df_demo.to_csv(os.path.join(DEMO_DIR, "demo.csv"))
    print("5. Đã lưu thành công các file CSV trong data/processed/ và data/demo/\n")


if __name__ == "__main__":
    raw_path = os.path.normpath(RAW_DATA_PATH)
    if not os.path.exists(raw_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {raw_path}")
        sys.exit(1)

    df_raw = load_raw_data(raw_path)
    df_clean = handle_missing_values(df_raw)
    df_hourly = resample_hourly(df_clean)
    contamination_rate = run_eda(df_hourly)
    save_contamination(contamination_rate)
    train, test, demo = split_data(df_hourly)
    save_datasets(train, test, demo)
    print("Hoàn tất tiền xử lý dữ liệu.")
