"""
config.py - Cau hinh he thong, duong dan, dac trung va bang mau sac.
"""

import os
import sys


def setup_encoding() -> None:
    """Thiet lap UTF-8 cho Windows console."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# Duong dan thu muc va file
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "household_power_consumption.txt")
TRAIN_HOURLY_PATH = os.path.join(DATA_DIR, "train_hourly.csv")
DEMO_STREAM_PATH = os.path.join(DATA_DIR, "demo_stream.csv")
STREAM_BUFFER_PATH = os.path.join(DATA_DIR, "stream_buffer.jsonl")
MODEL_BUNDLE_PATH = os.path.join(MODELS_DIR, "model_bundle.pkl")
CSS_FILE = os.path.join(PROJECT_DIR, "src", "style.css")


# Danh sach dac trung va nguong suy luan
TARGET_COL = "Global_active_power"

ENGINEERED_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "is_night",
    "power_diff_1h",
    "power_dev_24h",
    "power_zscore_6h",
    "voltage_diff_1h",
    "voltage_zscore_6h",
    "power_factor",
]

VOLTAGE_DROP_THRESHOLD = -15.0


# Cau hinh phat luong va nhan hien thi
SEND_INTERVAL = 1.0
MAX_DISPLAY_POINTS = 100

TYPE_LABELS = {
    "power_surge": "Đột biến công suất",
    "voltage_drop": "Sụt điện áp",
    "night_spike": "Đột biến đêm",
    "unknown": "Chưa xác định",
    "normal": "Bình thường",
}

FEATURE_LABELS = {
    "power_zscore_6h": "Bất thường công suất 6 giờ",
    "voltage_zscore_6h": "Bất thường điện áp 6 giờ",
    "power_factor": "Hệ số công suất",
    "power_dev_24h": "Lệch công suất so với hôm qua",
    "power_diff_1h": "Biến động công suất 1 giờ",
    "voltage_diff_1h": "Biến động điện áp 1 giờ",
    "is_night": "Khung giờ đêm khuya",
    "hour_sin": "Chu kỳ thời gian sin",
    "hour_cos": "Chu kỳ thời gian cos",
}

# Bang mau sac tap trung
COLORS = {
    "primary": "#0066FF",
    "primary_dark": "#0050CC",
    "primary_light": "#EBF5FF",
    "primary_rgba_05": "rgba(0, 102, 255, 0.05)",
    "primary_rgba_10": "rgba(0, 102, 255, 0.10)",

    "accent": "#F59E0B",
    "accent_light": "#FFFBEB",
    "accent_dark": "#B45309",
    "danger": "#EF4444",
    "danger_light": "#FEF2F2",
    "danger_dark": "#991B1B",
    "danger_border": "#FCA5A5",
    "success": "#10B981",
    "success_light": "#ECFDF5",
    "success_dark": "#065F46",
    "success_border": "#A7F3D0",
    "success_rgba_08": "rgba(16, 185, 129, 0.08)",
    "warning": "#F59E0B",
    "warning_light": "#FFFBEB",
    "warning_dark": "#92400E",
    "warning_border": "#FDE68A",

    "text_primary": "#0F172A",
    "text_secondary": "#334155",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "border_subtle": "#CBD5E1",
    "grid_line": "#F1F5F9",
    "background": "#F8FAFC",
    "surface": "#FFFFFF",
    "surface_subtle": "#F1F5F9",
}

ANOMALY_TYPE_COLORS = {
    "power_surge": "#F59E0B",
    "voltage_drop": "#EF4444",
    "night_spike": "#0066FF",
    "normal": "#10B981",
    "unknown": "#64748B",
}
