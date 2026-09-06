
import math
from typing import Optional, Union
import numpy as np
import pandas as pd
from config import (
    TARGET_COL, ENGINEERED_FEATURE_NAMES,
    VOLTAGE_DROP_THRESHOLD, FEATURE_LABELS,
)

_EPS = 1e-6


# 1. Trich xuat dac trung (The Sharp 9)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trich xuat 9 dac trung chinh tu du lieu dien ke:
    1-2. hour_sin, hour_cos: Chu ky 24 gio
    3. is_night: Gio dem (1h - 5h)
    4. power_diff_1h: Chenh lech cong suat 1 gio
    5. power_dev_24h: Do lech so voi cung gio hom truoc
    6. power_zscore_6h: Z-Score cong suat rolling 6 gio
    7. voltage_diff_1h: Chenh lech dien ap 1 gio
    8. voltage_zscore_6h: Z-Score dien ap rolling 6 gio
    9. power_factor: He so cong suat cos(phi)
    """
    df = df.copy()
    hour = df.index.hour

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_night"] = ((hour >= 1) & (hour <= 5)).astype(int)

    df["power_diff_1h"] = df[TARGET_COL] - df[TARGET_COL].shift(1)

    power_lag_24h = df[TARGET_COL].shift(24)
    df["power_dev_24h"] = (df[TARGET_COL] - power_lag_24h) / (power_lag_24h.abs() + _EPS)

    p_rmean = df[TARGET_COL].rolling(window=6, min_periods=1).mean()
    p_rstd = df[TARGET_COL].rolling(window=6, min_periods=1).std().fillna(0)
    df["power_zscore_6h"] = (df[TARGET_COL] - p_rmean) / (p_rstd + _EPS)

    df["voltage_diff_1h"] = df["Voltage"] - df["Voltage"].shift(1)

    v_rmean = df["Voltage"].rolling(window=6, min_periods=1).mean()
    v_rstd = df["Voltage"].rolling(window=6, min_periods=1).std().fillna(0)
    df["voltage_zscore_6h"] = (df["Voltage"] - v_rmean) / (v_rstd + _EPS)

    p = df[TARGET_COL]
    q = df["Global_reactive_power"] if "Global_reactive_power" in df.columns else 0.0
    apparent = np.sqrt(p**2 + q**2) + _EPS
    df["power_factor"] = np.clip(p / apparent, 0.0, 1.0)

    return df[ENGINEERED_FEATURE_NAMES].dropna()


def extract_latest(buffer_df: pd.DataFrame) -> Optional[pd.Series]:
    """Trich xuat dac trung cho diem du lieu moi nhat trong stream."""
    if len(buffer_df) < 25:
        return None
    feat_df = extract_features(buffer_df)
    if feat_df.empty:
        return None
    return feat_df.iloc[-1]


# 2. Tinh diem muc do nghiem trong (Severity Scoring)

def calc_severity(raw_score: float) -> float:
    """Chuyen doi score tu decision_function sang thang do [0, 1]."""
    return 1.0 / (1.0 + math.exp(max(-500, min(500, raw_score * 30.0))))


def get_severity_level(severity: float) -> str:
    """Phan cap muc do nghiem trong: normal (< 50%) / warning (50-70%) / critical (>= 70%)."""
    if severity >= 0.70:
        return "critical"
    elif severity >= 0.50:
        return "warning"
    return "normal"


# 3. Phan loai loi va giai thich nguyen nhan (XAI)

def classify_type(row: Union[pd.Series, dict]) -> str:
    """Phan loai loai bat thuong dua tren dac trung."""
    voltage_diff = float(row.get("voltage_diff_1h", 0.0))
    power_z = float(row.get("power_zscore_6h", 0.0))
    is_night = int(row.get("is_night", 0))
    power_dev = float(row.get("power_dev_24h", 0.0))

    if voltage_diff <= VOLTAGE_DROP_THRESHOLD:
        return "voltage_drop"
    if is_night == 1 and (power_z > 0.8 or power_dev > 0.8):
        return "night_spike"
    return "power_surge"


def calc_baseline_stats(df_feats: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Tinh median va IQR tren tap du lieu binh thuong de phuc vu giai thich XAI."""
    medians = {feat: float(df_feats[feat].median()) for feat in df_feats.columns}
    iqrs = {feat: float(df_feats[feat].quantile(0.75) - df_feats[feat].quantile(0.25)) for feat in df_feats.columns}
    return medians, iqrs


def explain_anomaly(
    row: Union[pd.Series, dict],
    medians: dict[str, float],
    iqrs: dict[str, float],
    top_n: int = 3
) -> str:
    """Giai thich Top N dac trung lech nhieu nhat so voi baseline."""
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
