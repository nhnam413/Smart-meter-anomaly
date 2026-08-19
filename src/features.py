"""
features.py — Module tạo đặc trưng (Feature Engineering)

Phiên bản tối ưu: Z-Score chuẩn hóa, Reactive Power Ratio,
Deviation tương đối 24h, và phân tách ngữ cảnh ban đêm (is_night).
"""

from typing import Optional
import numpy as np
import pandas as pd
from config import TARGET_COL

# Hằng số tránh chia cho 0
_EPS = 1e-6


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng tinh gọn và tối ưu cho Isolation Forest."""
    df = df.copy()

    # Loại bỏ các cột trùng lặp / dư thừa
    redundant_cols = ["Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    for col in redundant_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    hour = df.index.hour

    # Mã hóa chu kỳ giờ
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Đặt độ trễ & Vi phân công suất
    df["power_lag_1h"] = df[TARGET_COL].shift(1)
    df["power_diff_1h"] = df[TARGET_COL] - df["power_lag_1h"]
    df["power_lag_24h"] = df[TARGET_COL].shift(24)

    # Power Deviation tương đối so với cùng giờ hôm qua
    df["power_deviation_24h"] = (df[TARGET_COL] - df["power_lag_24h"]) / (df["power_lag_24h"].abs() + _EPS)

    # Z-Score cục bộ 6h
    power_rmean = df[TARGET_COL].rolling(window=6, min_periods=1).mean()
    power_rstd = df[TARGET_COL].rolling(window=6, min_periods=1).std().fillna(0)
    df["power_zscore_6h"] = (df[TARGET_COL] - power_rmean) / (power_rstd + _EPS)

    # Thống kê điện áp — Z-Score
    df["voltage_lag_1h"] = df["Voltage"].shift(1)
    df["voltage_diff_1h"] = df["Voltage"] - df["voltage_lag_1h"]
    voltage_rmean = df["Voltage"].rolling(window=6, min_periods=1).mean()
    voltage_rstd = df["Voltage"].rolling(window=6, min_periods=1).std().fillna(0)
    df["voltage_zscore_6h"] = (df["Voltage"] - voltage_rmean) / (voltage_rstd + _EPS)

    # Reactive Power Ratio
    if "Global_reactive_power" in df.columns:
        df["reactive_ratio"] = np.clip(
            df["Global_reactive_power"] / (df[TARGET_COL].abs() + _EPS),
            -10, 10
        )
        df.drop(columns=["Global_reactive_power"], inplace=True)

    # Baseline giờ & Ngữ cảnh ban đêm
    hourly_mean = df.groupby(hour)[TARGET_COL].transform("mean")
    df["power_hourly_diff"] = df[TARGET_COL] - hourly_mean
    df["is_night"] = ((hour >= 1) & (hour <= 5)).astype(int)

    return df.dropna()


def create_features_realtime(buffer_df: pd.DataFrame) -> Optional[pd.Series]:
    """Tạo đặc trưng cho điểm dữ liệu mới nhất trong luồng Real-Time."""
    if len(buffer_df) < 25:
        return None
    feat_df = create_features(buffer_df)
    if feat_df.empty:
        return None
    return feat_df.iloc[-1]



ENGINEERED_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "power_lag_1h",
    "power_diff_1h",
    "power_lag_24h",
    "power_deviation_24h",
    "power_zscore_6h",
    "voltage_lag_1h",
    "voltage_diff_1h",
    "voltage_zscore_6h",
    "reactive_ratio",
    "power_hourly_diff",
    "is_night",
]
