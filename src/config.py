

import os
import sys


# Thiết lập UTF-8 cho console Windows.
def setup_encoding() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# Đường dẫn
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "household_power_consumption.txt")
TRAIN_HOURLY_PATH = os.path.join(DATA_DIR, "train_hourly.csv")
DEMO_STREAM_PATH = os.path.join(DATA_DIR, "demo_stream.csv")
STREAM_BUFFER_PATH = os.path.join(DATA_DIR, "stream_buffer.jsonl")
MODEL_BUNDLE_PATH = os.path.join(MODELS_DIR, "model_bundle.pkl")
CSS_FILE = os.path.join(PROJECT_DIR, "src", "style.css")

# Dữ liệu và ngưỡng nghiệp vụ
TARGET_COL = "Global_active_power"
VOLTAGE_DROP_THRESHOLD = -15.0

# Định dạng thời gian
DATE_FORMAT = "%d/%m/%Y"
DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DATETIME_MINUTE_FORMAT = "%d/%m/%Y %H:%M"

# Thứ tự đặc trưng của mô hình
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

# Luồng mô phỏng
SEND_INTERVAL = 1.0
MAX_DISPLAY_POINTS = 100

# Nhãn hiển thị
TYPE_LABELS = {
    "power_surge": "Đột biến công suất",
    "voltage_drop": "Sụt điện áp",
    "night_spike": "Đột biến đêm",
    "normal": "Bình thường",
    "unknown": "Chưa xác định",
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

# Bảng màu giao diện
COLORS = {
    "primary": "#0066FF",
    "primary_dark": "#0050CC",
    "primary_light": "#EBF5FF",
    "primary_rgba_05": "rgba(0, 102, 255, 0.05)",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "success": "#10B981",
    "success_dark": "#065F46",
    "success_rgba_08": "rgba(16, 185, 129, 0.08)",
    "text_primary": "#0F172A",
    "text_secondary": "#334155",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "grid_line": "#F1F5F9",
    "surface": "#FFFFFF",
}

ANOMALY_TYPE_COLORS = {
    "power_surge": "#F59E0B",
    "voltage_drop": "#EF4444",
    "night_spike": "#0066FF",
    "normal": "#10B981",
    "unknown": "#64748B",
}
