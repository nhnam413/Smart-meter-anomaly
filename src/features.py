import math

import numpy as np
import pandas as pd

from config import (
    ENGINEERED_FEATURE_NAMES,
    FEATURE_LABELS,
    TARGET_COL,
    VOLTAGE_DROP_THRESHOLD,
)

_EPS = 1e-6
_ROLLING_WINDOW = 6
_MIN_STREAM_SAMPLES = 25


# Tính Z-score trên cửa sổ trượt sáu mẫu.
def _rolling_zscore(series: pd.Series) -> pd.Series:
    rolling = series.rolling(window=_ROLLING_WINDOW, min_periods=1)
    mean = rolling.mean()
    std = rolling.std().fillna(0)
    return (series - mean) / (std + _EPS)


# Trích xuất chín đặc trưng theo đúng thứ tự mô hình.
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df.index.hour

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_night"] = ((hour >= 1) & (hour <= 5)).astype(int)

    df["power_diff_1h"] = df[TARGET_COL] - df[TARGET_COL].shift(1)

    power_lag_24h = df[TARGET_COL].shift(24)
    df["power_dev_24h"] = (df[TARGET_COL] - power_lag_24h) / (
        power_lag_24h.abs() + _EPS
    )

    df["power_zscore_6h"] = _rolling_zscore(df[TARGET_COL])

    df["voltage_diff_1h"] = df["Voltage"] - df["Voltage"].shift(1)

    df["voltage_zscore_6h"] = _rolling_zscore(df["Voltage"])

    p = df[TARGET_COL]
    q = df["Global_reactive_power"] if "Global_reactive_power" in df.columns else 0.0
    apparent = np.sqrt(p**2 + q**2) + _EPS
    df["power_factor"] = np.clip(p / apparent, 0.0, 1.0)

    return df[ENGINEERED_FEATURE_NAMES].dropna()


# Trả về đặc trưng của mẫu mới nhất khi đủ dữ liệu nền.
def extract_latest(buffer_df: pd.DataFrame) -> pd.Series | None:
    if len(buffer_df) < _MIN_STREAM_SAMPLES:
        return None
    feat_df = extract_features(buffer_df)
    if feat_df.empty:
        return None
    return feat_df.iloc[-1]


# Quy đổi điểm mô hình sang mức độ từ 0 đến 1.
def calc_severity(raw_score: float) -> float:
    return 1.0 / (1.0 + math.exp(max(-500, min(500, raw_score * 30.0))))


# Phân cấp mức độ cảnh báo.
def get_severity_level(severity: float) -> str:
    if severity >= 0.70:
        return "critical"
    if severity >= 0.50:
        return "warning"
    return "normal"


# Gợi ý dạng bất thường từ các đặc trưng.
def classify_type(row: pd.Series | dict) -> str:
    voltage_diff = float(row.get("voltage_diff_1h", 0.0))
    power_z = float(row.get("power_zscore_6h", 0.0))
    is_night = int(row.get("is_night", 0))
    power_dev = float(row.get("power_dev_24h", 0.0))

    if voltage_diff <= VOLTAGE_DROP_THRESHOLD:
        return "voltage_drop"
    if is_night == 1 and (power_z > 0.8 or power_dev > 0.8):
        return "night_spike"
    return "power_surge"


# Tính median và IQR làm đường cơ sở cho XAI.
def calc_baseline_stats(
    df_feats: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    medians = {feat: float(df_feats[feat].median()) for feat in df_feats.columns}
    iqrs = {
        feat: float(df_feats[feat].quantile(0.75) - df_feats[feat].quantile(0.25))
        for feat in df_feats.columns
    }
    return medians, iqrs


# Mô tả các đặc trưng lệch nhiều nhất so với đường cơ sở.
def explain_anomaly(
    row: pd.Series | dict,
    medians: dict[str, float],
    iqrs: dict[str, float],
    top_n: int = 3,
) -> str:
    contributions = []
    for feat, median_val in medians.items():
        if feat not in row or feat not in iqrs:
            continue
        val = float(row[feat])
        iqr_val = iqrs[feat]
        deviation = abs(val - median_val) / (iqr_val + _EPS)
        if deviation > 0.5:
            label = FEATURE_LABELS.get(feat, feat)
            contributions.append((label, val, deviation))

    contributions.sort(key=lambda x: x[2], reverse=True)
    top = contributions[:top_n]
    return ", ".join(f"{label}={val:+.2f}" for label, val, _ in top) if top else "—"
