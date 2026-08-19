"""
config.py — Cấu hình tập trung cho hệ thống phát hiện bất thường điện năng
"""

import os

# --- Đường dẫn hệ thống ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(PROJECT_DIR, "data", "raw", "household_power_consumption.txt")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
DEMO_DIR = os.path.join(PROJECT_DIR, "data", "demo")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")

CONTAMINATION_FILE = os.path.join(PROCESSED_DIR, "contamination.txt")
STREAM_FILE = os.path.join(DEMO_DIR, "stream_buffer.jsonl")
DEMO_CSV = os.path.join(DEMO_DIR, "demo_with_anomalies.csv")
CSS_FILE = os.path.join(PROJECT_DIR, "src", "style.css")

# --- Phân chia tập dữ liệu (Tỷ lệ) ---
TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
DEMO_RATIO = 0.1

# --- Siêu tham số mô hình Isolation Forest ---
CONTAMINATION_TARGET = 0.08
CONTAMINATION_MIN = 0.07
CONTAMINATION_MAX = 0.12
N_ESTIMATORS = 200
MAX_SAMPLES = 512
MAX_FEATURES = 0.5
RANDOM_STATE = 42

# --- Tỷ lệ giả lập các loại bất thường ---
POWER_SURGE_RATIO = 0.03
VOLTAGE_DROP_RATIO = 0.03
NIGHT_SPIKE_RATIO = 0.02

# --- Cấu hình Producer Stream ---
SEND_INTERVAL = 1.0

# --- Cấu hình Dashboard ---
MAX_DISPLAY_POINTS = 100
BUFFER_SIZE = 30

# --- Bảng màu hệ thống (Design Tokens) ---
COLORS = {
    "primary": "#6B4CE6",
    "primary_dark": "#5338B5",
    "primary_light": "#F3F0FF",
    "accent": "#EF7D32",
    "accent_light": "#FFF4EB",
    "danger": "#EF4444",
    "danger_light": "#FEF2F2",
    "success": "#10B981",
    "success_light": "#ECFDF5",
    "warning": "#F59E0B",
    "warning_light": "#FFFBEB",
    "text_primary": "#0F172A",
    "text_secondary": "#334155",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "background": "#F8FAFC",
    "surface": "#FFFFFF",
}

# --- Danh sách cột cảm biến ---
SENSOR_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

TARGET_COL = "Global_active_power"
