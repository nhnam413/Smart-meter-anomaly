"""
============================================================
01_prepare_data.py — Tiền xử lý dữ liệu tiêu thụ điện năng
============================================================
Mục đích:
    - Đọc dataset gốc UCI "Individual Household Electric Power Consumption".
    - Xử lý giá trị thiếu (missing values) — dataset gốc dùng dấu '?' cho NaN.
    - Resample dữ liệu từ tần suất 1 phút → 1 giờ (giảm nhiễu, tối ưu tốc độ huấn luyện).
    - Chia tập dữ liệu thành 3 phần: Train / Test / Demo.
    - Lưu kết quả ra thư mục data/processed/ và data/demo/.

Tác giả: Sinh viên + AI Advisor
Ngày tạo: 2026-07-12
============================================================
"""

import os
import sys
import pandas as pd
import numpy as np

# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

# Đường dẫn gốc của dự án (thư mục cha của src/)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn tới file dataset gốc (đã chuyển vào data/raw/)
RAW_DATA_PATH = os.path.join(
    PROJECT_DIR, "data", "raw",
    "household_power_consumption.txt"
)

# Thư mục lưu kết quả
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
DEMO_DIR = os.path.join(PROJECT_DIR, "data", "demo")


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Đọc file dataset gốc UCI (sep=';', na='?').
    Ghép Date+Time → datetime index, ép kiểu float.
    """
    print("=" * 60)
    print("📂 BƯỚC 1: Đọc dữ liệu gốc...")
    print(f"   File: {os.path.abspath(file_path)}")
    print("=" * 60)

    df = pd.read_csv(file_path, sep=";", na_values=["?"], low_memory=False)

    # Ghép Date+Time → datetime index (format châu Âu: dd/mm/yyyy)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    df = df.set_index("datetime")
    df = df.drop(columns=["Date", "Time"])

    # Ép kiểu float (cột có thể là object do '?')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"✅ Đọc xong: {df.shape[0]:,} dòng × {df.shape[1]} cột")
    print(f"   Khoảng thời gian: {df.index.min()} → {df.index.max()}")
    print(f"   Tổng giá trị thiếu: {df.isnull().sum().sum():,}")
    print()

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý giá trị thiếu: dropna(all) → ffill → bfill.
    Dùng ffill/bfill thay vì mean vì dữ liệu chuỗi thời gian có tính tuần hoàn.
    """
    print("=" * 60)
    print("🔧 BƯỚC 2: Xử lý giá trị thiếu (Missing Values)...")
    print("=" * 60)

    before_count = df.isnull().sum().sum()

    df = df.dropna(how="all")
    df = df.ffill()
    df = df.bfill()

    after_count = df.isnull().sum().sum()

    print(f"   NaN trước xử lý : {before_count:,}")
    print(f"   NaN sau xử lý   : {after_count:,}")
    print(f"   Số dòng còn lại : {df.shape[0]:,}")
    print()

    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1 phút → 1 giờ (mean). Giảm ~60x mẫu nhưng giữ chu kỳ ngày/đêm.
    """
    print("=" * 60)
    print("⏱️  BƯỚC 3: Resample dữ liệu → 1 GIỜ / mẫu...")
    print("=" * 60)

    df_hourly = df.resample("h").mean()
    df_hourly = df_hourly.dropna()

    print(f"   Kích thước trước resample : {df.shape[0]:,} dòng (phút)")
    print(f"   Kích thước sau resample   : {df_hourly.shape[0]:,} dòng (giờ)")
    print(f"   Tỉ lệ giảm               : {df.shape[0] / df_hourly.shape[0]:.1f}x")
    print()

    return df_hourly


def split_data(df: pd.DataFrame,
               train_ratio: float = 0.7,
               test_ratio: float = 0.2,
               demo_ratio: float = 0.1) -> tuple:
    """
    Chia dữ liệu thành 3 tập: Train / Test / Demo.

    ⚠️ QUAN TRỌNG: Với dữ liệu chuỗi thời gian, KHÔNG được shuffle (xáo trộn)!
       Phải chia theo thứ tự thời gian để tránh "data leakage"
       (dữ liệu tương lai lọt vào tập huấn luyện).

    Phân bổ mặc định:
        - Train (70%): Dùng huấn luyện model Isolation Forest.
        - Test  (20%): Dùng đánh giá hiệu năng model.
        - Demo  (10%): Dùng để giả lập streaming real-time.

    Args:
        df          : DataFrame đã xử lý
        train_ratio : Tỉ lệ tập train (mặc định 70%)
        test_ratio  : Tỉ lệ tập test (mặc định 20%)
        demo_ratio  : Tỉ lệ tập demo (mặc định 10%)

    Returns:
        (df_train, df_test, df_demo)
    """
    print("=" * 60)
    print("✂️  BƯỚC 4: Chia tập dữ liệu (Train / Test / Demo)...")
    print("=" * 60)

    n = len(df)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))

    # Chia theo thứ tự thời gian (KHÔNG shuffle — tránh data leakage)
    df_train = df.iloc[:train_end].copy()
    df_test  = df.iloc[train_end:test_end].copy()
    df_demo  = df.iloc[test_end:].copy()

    print(f"   Tổng cộng   : {n:,} mẫu (hourly)")
    print(f"   ├── Train   : {len(df_train):,} mẫu "
          f"({df_train.index.min().date()} → {df_train.index.max().date()})")
    print(f"   ├── Test    : {len(df_test):,} mẫu "
          f"({df_test.index.min().date()} → {df_test.index.max().date()})")
    print(f"   └── Demo    : {len(df_demo):,} mẫu "
          f"({df_demo.index.min().date()} → {df_demo.index.max().date()})")
    print()

    return df_train, df_test, df_demo


def save_datasets(df_train: pd.DataFrame,
                  df_test: pd.DataFrame,
                  df_demo: pd.DataFrame) -> None:
    """
    Lưu 3 tập dữ liệu ra file CSV.

    Cấu trúc file output:
        data/processed/train.csv    — Tập huấn luyện
        data/processed/test.csv     — Tập kiểm thử
        data/demo/demo.csv          — Tập demo streaming
    """
    print("=" * 60)
    print("💾 BƯỚC 5: Lưu dữ liệu ra file CSV...")
    print("=" * 60)

    # Đảm bảo thư mục tồn tại
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Lưu với index=True vì cột datetime là index
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    demo_path = os.path.join(DEMO_DIR, "demo.csv")

    df_train.to_csv(train_path)
    df_test.to_csv(test_path)
    df_demo.to_csv(demo_path)

    print(f"   ✅ Train : {train_path}")
    print(f"   ✅ Test  : {test_path}")
    print(f"   ✅ Demo  : {demo_path}")
    print()

    # In thống kê mô tả tập Train để kiểm tra nhanh
    print("=" * 60)
    print("📊 Thống kê mô tả (Descriptive Statistics) — Tập Train:")
    print("=" * 60)
    print(df_train.describe().round(3).to_string())
    print()


# ============================================================
# MAIN — Chạy toàn bộ pipeline tiền xử lý
# ============================================================

if __name__ == "__main__":
    print()
    print("🔌 SMART METER ANOMALY DETECTION — TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    print()

    # --- Kiểm tra file tồn tại ---
    raw_path = os.path.normpath(RAW_DATA_PATH)
    if not os.path.exists(raw_path):
        print(f"❌ KHÔNG tìm thấy file dataset: {raw_path}")
        print("   Vui lòng tải từ: https://archive.ics.uci.edu/dataset/235")
        sys.exit(1)

    # --- Pipeline tiền xử lý ---
    df = load_raw_data(raw_path)              # Bước 1: Đọc file gốc
    df = handle_missing_values(df)            # Bước 2: Xử lý NaN
    df_hourly = resample_hourly(df)           # Bước 3: Resample → 1 giờ
    train, test, demo = split_data(df_hourly) # Bước 4: Chia tập dữ liệu
    save_datasets(train, test, demo)          # Bước 5: Lưu CSV

    print("🎉 HOÀN TẤT! Dữ liệu đã sẵn sàng cho bước tiếp theo.")
    print("   → Chạy tiếp: python src/02_train_model.py")
    print()
