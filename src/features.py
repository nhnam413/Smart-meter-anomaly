"""
features.py — Module tạo đặc trưng (Feature Engineering)

Phiên bản cải tiến: Z-Score chuẩn hóa, Reactive Power Ratio,
Deviation tương đối 24h, và sửa lỗi Night Spike binary gating.
"""

import numpy as np
import pandas as pd
from config import TARGET_COL

# Hằng số tránh chia cho 0
_EPS = 1e-6


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng tinh gọn và tối ưu cho Isolation Forest."""
    df = df.copy()

    # Loại bỏ các cột trùng lặp / dư thừa
    # Giữ lại Global_reactive_power để tính reactive_ratio
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

    # --- MỚI: Power Deviation tương đối so với cùng giờ hôm qua ---
    df["power_deviation_24h"] = (df[TARGET_COL] - df["power_lag_24h"]) / (df["power_lag_24h"].abs() + _EPS)

    # --- MỚI: Z-Score cục bộ 6h (thay thế rolling mean/std riêng lẻ) ---
    power_rmean = df[TARGET_COL].rolling(window=6, min_periods=1).mean()
    power_rstd = df[TARGET_COL].rolling(window=6, min_periods=1).std().fillna(0)
    df["power_zscore_6h"] = (df[TARGET_COL] - power_rmean) / (power_rstd + _EPS)

    # Thống kê điện áp — Z-Score
    df["voltage_lag_1h"] = df["Voltage"].shift(1)
    df["voltage_diff_1h"] = df["Voltage"] - df["voltage_lag_1h"]
    voltage_rmean = df["Voltage"].rolling(window=6, min_periods=1).mean()
    voltage_rstd = df["Voltage"].rolling(window=6, min_periods=1).std().fillna(0)
    df["voltage_zscore_6h"] = (df["Voltage"] - voltage_rmean) / (voltage_rstd + _EPS)

    # --- MỚI: Reactive Power Ratio ---
    if "Global_reactive_power" in df.columns:
        df["reactive_ratio"] = np.clip(
            df["Global_reactive_power"] / (df[TARGET_COL].abs() + _EPS),
            -10, 10
        )
        # Bỏ cột gốc sau khi đã trích feature
        df.drop(columns=["Global_reactive_power"], inplace=True)

    # Baseline giờ & Ngữ cảnh ban đêm (sửa lỗi binary gating)
    hourly_mean = df.groupby(hour)[TARGET_COL].transform("mean")
    df["power_hourly_diff"] = df[TARGET_COL] - hourly_mean

    # --- SỬA: Tách is_night binary thay vì night_power_spike nhân mask ---
    df["is_night"] = ((hour >= 1) & (hour <= 5)).astype(int)

    return df.dropna()


def create_features_realtime(buffer_df: pd.DataFrame) -> pd.Series | None:
    """Tạo đặc trưng cho điểm dữ liệu mới nhất trong luồng Real-Time."""
    if len(buffer_df) < 25:
        return None

    latest = buffer_df.iloc[-1].copy()
    hour = buffer_df.index[-1].hour

    # Loại bỏ các cột dư thừa (giữ lại Global_reactive_power tạm)
    for col in ["Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        if col in latest.index:
            latest = latest.drop(labels=[col])

    latest["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    latest["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    latest["power_lag_1h"] = buffer_df[TARGET_COL].iloc[-2]
    latest["power_diff_1h"] = latest[TARGET_COL] - latest["power_lag_1h"]
    latest["power_lag_24h"] = buffer_df[TARGET_COL].iloc[-25]

    # Power Deviation tương đối so với cùng giờ hôm qua
    lag_24h_val = latest["power_lag_24h"]
    latest["power_deviation_24h"] = (latest[TARGET_COL] - lag_24h_val) / (abs(lag_24h_val) + _EPS)

    # Z-Score cục bộ 6h — Power
    last_6 = buffer_df[TARGET_COL].iloc[-6:]
    p_mean = last_6.mean()
    p_std = last_6.std()
    if pd.isna(p_std):
        p_std = 0.0
    latest["power_zscore_6h"] = (latest[TARGET_COL] - p_mean) / (p_std + _EPS)

    # Voltage features — Z-Score
    latest["voltage_lag_1h"] = buffer_df["Voltage"].iloc[-2]
    latest["voltage_diff_1h"] = latest["Voltage"] - latest["voltage_lag_1h"]
    last_6v = buffer_df["Voltage"].iloc[-6:]
    v_mean = last_6v.mean()
    v_std = last_6v.std()
    if pd.isna(v_std):
        v_std = 0.0
    latest["voltage_zscore_6h"] = (latest["Voltage"] - v_mean) / (v_std + _EPS)

    # Reactive Power Ratio
    if "Global_reactive_power" in latest.index:
        latest["reactive_ratio"] = np.clip(
            latest["Global_reactive_power"] / (abs(latest[TARGET_COL]) + _EPS),
            -10, 10
        )
        latest = latest.drop(labels=["Global_reactive_power"])

    # Baseline giờ & Ngữ cảnh ban đêm
    same_hour_mask = buffer_df.index.hour == hour
    hourly_avg = buffer_df.loc[same_hour_mask, TARGET_COL].mean()
    latest["power_hourly_diff"] = latest[TARGET_COL] - hourly_avg

    latest["is_night"] = 1 if (1 <= hour <= 5) else 0

    return latest


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
