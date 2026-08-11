"""
features.py — Module tạo đặc trưng (Feature Engineering)
"""

import numpy as np
import pandas as pd
from config import TARGET_COL


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng tinh gọn và tối ưu cho Isolation Forest."""
    df = df.copy()

    # Loại bỏ các cột trùng lặp / dư thừa
    redundant_cols = ["Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "dow_sin", "dow_cos"]
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

    # Thống kê cuộn công suất
    df["power_rolling_mean_6h"] = df[TARGET_COL].rolling(window=6, min_periods=1).mean()
    df["power_rolling_std_6h"] = df[TARGET_COL].rolling(window=6, min_periods=1).std()

    # Thống kê điện áp
    df["voltage_lag_1h"] = df["Voltage"].shift(1)
    df["voltage_diff_1h"] = df["Voltage"] - df["voltage_lag_1h"]
    df["voltage_rolling_mean_6h"] = df["Voltage"].rolling(window=6, min_periods=1).mean()
    df["voltage_rolling_std_6h"] = df["Voltage"].rolling(window=6, min_periods=1).std()

    # Baseline giờ & Bất thường ban đêm
    hourly_mean = df.groupby(hour)[TARGET_COL].transform("mean")
    df["power_hourly_diff"] = df[TARGET_COL] - hourly_mean

    is_night = ((hour >= 1) & (hour <= 5)).astype(int)
    df["night_power_spike"] = df["power_hourly_diff"] * is_night

    return df.dropna()


def create_features_realtime(buffer_df: pd.DataFrame) -> pd.Series | None:
    """Tạo đặc trưng cho điểm dữ liệu mới nhất trong luồng Real-Time."""
    if len(buffer_df) < 25:
        return None

    latest = buffer_df.iloc[-1].copy()
    hour = buffer_df.index[-1].hour

    for col in ["Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "dow_sin", "dow_cos"]:
        if col in latest:
            latest.drop(labels=[col], inplace=True)

    latest["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    latest["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    latest["power_lag_1h"] = buffer_df[TARGET_COL].iloc[-2]
    latest["power_diff_1h"] = latest[TARGET_COL] - latest["power_lag_1h"]
    latest["power_lag_24h"] = buffer_df[TARGET_COL].iloc[-25]

    last_6 = buffer_df[TARGET_COL].iloc[-6:]
    latest["power_rolling_mean_6h"] = last_6.mean()
    latest["power_rolling_std_6h"] = last_6.std()

    latest["voltage_lag_1h"] = buffer_df["Voltage"].iloc[-2]
    latest["voltage_diff_1h"] = latest["Voltage"] - latest["voltage_lag_1h"]
    last_6v = buffer_df["Voltage"].iloc[-6:]
    latest["voltage_rolling_mean_6h"] = last_6v.mean()
    latest["voltage_rolling_std_6h"] = last_6v.std()

    same_hour_mask = buffer_df.index.hour == hour
    hourly_avg = buffer_df.loc[same_hour_mask, TARGET_COL].mean()
    latest["power_hourly_diff"] = latest[TARGET_COL] - hourly_avg

    is_night = 1 if (1 <= hour <= 5) else 0
    latest["night_power_spike"] = latest["power_hourly_diff"] * is_night

    return latest


ENGINEERED_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "power_lag_1h",
    "power_diff_1h",
    "power_lag_24h",
    "power_rolling_mean_6h",
    "power_rolling_std_6h",
    "voltage_lag_1h",
    "voltage_diff_1h",
    "voltage_rolling_mean_6h",
    "voltage_rolling_std_6h",
    "power_hourly_diff",
    "night_power_spike",
]
