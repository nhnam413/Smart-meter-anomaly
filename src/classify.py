"""
classify.py — Phân loại bất thường (Rule-based) & Giải thích đặc trưng đóng góp
"""

from typing import Union
import pandas as pd

# ── Ngưỡng phân loại bất thường ──────────────────────────────────
VOLTAGE_DROP_THRESHOLD = -15.0   # voltage_diff_1h (V)
POWER_SURGE_ZSCORE = 1.5         # power_zscore_6h
POWER_SURGE_DEVIATION = 1.5      # power_deviation_24h
NIGHT_SPIKE_ZSCORE = 1.0         # power_zscore_6h (1h - 5h)

# ── Nhãn hiển thị ─────────────────────────────────────────────────
TYPE_LABELS = {
    "power_surge": "Đột biến công suất",
    "voltage_drop": "Sụt áp điện",
    "night_spike": "Đột biến đêm",
    "unknown": "Chưa xác định",
    "normal": "Bình thường",
}

FEATURE_LABELS = {
    "power_zscore_6h": "Z-Score công suất 6h",
    "voltage_zscore_6h": "Z-Score điện áp 6h",
    "reactive_ratio": "Tỷ lệ phản kháng",
    "power_deviation_24h": "Chênh lệch 24h",
    "power_diff_1h": "Vi phân công suất",
    "voltage_diff_1h": "Vi phân điện áp",
    "power_hourly_diff": "Lệch TB giờ",
    "is_night": "Giờ đêm",
    "power_lag_1h": "CS trước 1h",
    "power_lag_24h": "CS hôm qua",
    "voltage_lag_1h": "ĐA trước 1h",
    "hour_sin": "Chu kỳ giờ (sin)",
    "hour_cos": "Chu kỳ giờ (cos)",
}


def classify_anomaly_type(row: Union[pd.Series, dict]) -> str:
    """Phân loại loại bất thường dựa trên quy tắc chuyên ngành."""
    voltage_diff = row.get("voltage_diff_1h", 0)
    power_z = row.get("power_zscore_6h", 0)
    is_night = row.get("is_night", 0)
    power_dev = row.get("power_deviation_24h", 0)

    if voltage_diff < VOLTAGE_DROP_THRESHOLD:
        return "voltage_drop"

    if power_z > POWER_SURGE_ZSCORE or power_dev > POWER_SURGE_DEVIATION:
        return "night_spike" if is_night == 1 else "power_surge"

    if is_night == 1 and power_z > NIGHT_SPIKE_ZSCORE:
        return "night_spike"

    return "unknown"


def classify_batch(df: pd.DataFrame) -> pd.Series:
    """Phân loại hàng loạt cho DataFrame."""
    return df.apply(classify_anomaly_type, axis=1)


def compute_feature_stats(df: pd.DataFrame, feature_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Tính median và IQR cho mỗi feature từ dữ liệu bình thường."""
    medians, iqrs = {}, {}
    for feat in feature_names:
        if feat in df.columns:
            q1, q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
            medians[feat] = float(df[feat].median())
            iqrs[feat] = float(q3 - q1)
    return medians, iqrs


def explain_anomaly(
    row: Union[pd.Series, dict],
    feature_names: list[str],
    medians: dict[str, float],
    iqrs: dict[str, float],
    top_n: int = 3
) -> str:
    """Trả về chuỗi giải thích Top N đặc trưng đóng góp lớn nhất vào bất thường."""
    contributions = []

    for feat in feature_names:
        if feat not in medians or feat not in iqrs:
            continue
        if feat not in row:
            continue

        val = float(row[feat])
        iqr = iqrs[feat]
        score = abs(val - medians[feat]) / iqr if iqr > 1e-9 else abs(val - medians[feat])

        if score > 0.5:
            label = FEATURE_LABELS.get(feat, feat)
            contributions.append((label, val, score))

    contributions.sort(key=lambda x: x[2], reverse=True)
    top = contributions[:top_n]

    return ", ".join(f"{label}={val:+.2f}" for label, val, _ in top) if top else "—"


def explain_batch(
    df: pd.DataFrame,
    feature_names: list[str],
    medians: dict[str, float],
    iqrs: dict[str, float],
    top_n: int = 3
) -> pd.Series:
    """Giải thích hàng loạt cho DataFrame."""
    return df.apply(lambda row: explain_anomaly(row, feature_names, medians, iqrs, top_n), axis=1)
