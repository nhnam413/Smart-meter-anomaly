"""
03_inject_anomalies.py — Giả lập bơm dữ liệu bất thường cho tập Demo
"""

import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import (
    DEMO_DIR,
    POWER_SURGE_RATIO,
    VOLTAGE_DROP_RATIO,
    NIGHT_SPIKE_RATIO,
    RANDOM_STATE as RANDOM_SEED,
)


def load_demo_data() -> pd.DataFrame:
    """Tải tập dữ liệu demo và khởi tạo cột nhãn bất thường."""
    demo_path = os.path.join(DEMO_DIR, "demo.csv")
    if not os.path.exists(demo_path):
        print(f"Lỗi: Không tìm thấy file {demo_path}")
        sys.exit(1)

    df = pd.read_csv(demo_path, index_col="datetime", parse_dates=True)
    df["is_anomaly"] = 0
    df["anomaly_type"] = "normal"
    print(f"1. Tải dữ liệu Demo: {len(df):,} mẫu")
    return df


def inject_power_surge(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Giả lập lỗi đột biến công suất (Power Surge)."""
    normal_indices = df.index[df["is_anomaly"] == 0]
    n_inject = int(len(df) * POWER_SURGE_RATIO)
    inject_indices = rng.choice(normal_indices, size=n_inject, replace=False)
    multipliers = rng.uniform(3.0, 5.0, size=n_inject)

    for idx, mult in zip(inject_indices, multipliers):
        df.loc[idx, "Global_active_power"] *= mult
        df.loc[idx, "Global_intensity"] *= mult
        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = "power_surge"

    print(f"  - Bơm lỗi đột biến công suất: {n_inject} mẫu")
    return df


def inject_voltage_drop(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Giả lập lỗi sụt áp (Voltage Drop)."""
    normal_indices = df.index[df["is_anomaly"] == 0]
    n_inject = int(len(df) * VOLTAGE_DROP_RATIO)
    inject_indices = rng.choice(normal_indices, size=n_inject, replace=False)
    voltage_drops = rng.uniform(20.0, 40.0, size=n_inject)

    for idx, drop in zip(inject_indices, voltage_drops):
        df.loc[idx, "Voltage"] -= drop
        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = "voltage_drop"

    print(f"  - Bơm lỗi sụt áp: {n_inject} mẫu")
    return df


def inject_night_spike(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Giả lập lỗi bất thường ban đêm (Night Spike 1h-5h AM)."""
    night_mask = (df.index.hour >= 1) & (df.index.hour <= 5) & (df["is_anomaly"] == 0)
    night_indices = df.index[night_mask]

    if len(night_indices) == 0:
        return df

    n_inject = min(int(len(df) * NIGHT_SPIKE_RATIO), len(night_indices))
    inject_indices = rng.choice(night_indices, size=n_inject, replace=False)
    multipliers = rng.uniform(2.0, 3.0, size=n_inject)

    for idx, mult in zip(inject_indices, multipliers):
        df.loc[idx, "Global_active_power"] *= mult
        df.loc[idx, "Global_intensity"] *= mult
        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = "night_spike"

    print(f"  - Bơm lỗi ban đêm: {n_inject} mẫu")
    return df


def save_demo_with_anomalies(df: pd.DataFrame) -> None:
    """Lưu tập dữ liệu demo đã bơm lỗi ra file CSV."""
    output_path = os.path.join(DEMO_DIR, "demo_with_anomalies.csv")
    df.to_csv(output_path)
    total = len(df)
    n_anomaly = int(df["is_anomaly"].sum())
    print(f"3. Đã lưu tập Demo hoàn chỉnh ({n_anomaly:,}/{total:,} điểm bất thường) -> {output_path}\n")


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)

    df = load_demo_data()
    print("2. Tiến hành giả lập các loại bất thường:")
    df = inject_power_surge(df, rng)
    df = inject_voltage_drop(df, rng)
    df = inject_night_spike(df, rng)
    save_demo_with_anomalies(df)
    print("Hoàn tất bơm dữ liệu bất thường giả lập.")
