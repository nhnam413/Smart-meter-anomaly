"""
04_producer.py — Trình phát luồng dữ liệu giả lập thời gian thực (JSONL Stream)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import DEMO_DIR, STREAM_FILE, SEND_INTERVAL


def load_demo_with_anomalies() -> pd.DataFrame:
    """Tải dữ liệu demo đã bơm bất thường."""
    demo_path = os.path.join(DEMO_DIR, "demo_with_anomalies.csv")
    if not os.path.exists(demo_path):
        print(f"Lỗi: Không tìm thấy file {demo_path}")
        sys.exit(1)

    return pd.read_csv(demo_path, index_col="datetime", parse_dates=True)


def row_to_json(row: pd.Series, timestamp: str) -> dict:
    """Chuyển đổi một dòng dữ liệu thành cấu trúc JSON message."""
    return {
        "timestamp": timestamp,
        "Global_active_power": round(float(row["Global_active_power"]), 4),
        "Global_reactive_power": round(float(row["Global_reactive_power"]), 4),
        "Voltage": round(float(row["Voltage"]), 4),
        "Global_intensity": round(float(row["Global_intensity"]), 4),
        "Sub_metering_1": round(float(row["Sub_metering_1"]), 4),
        "Sub_metering_2": round(float(row["Sub_metering_2"]), 4),
        "Sub_metering_3": round(float(row["Sub_metering_3"]), 4),
        "is_anomaly": int(row["is_anomaly"]),
        "anomaly_type": str(row["anomaly_type"]),
        "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def run_file_producer(df: pd.DataFrame, interval: float = SEND_INTERVAL, append: bool = False) -> None:
    """Ghi tin nhắn dạng luồng vào file JSONL theo từng khoảng thời gian."""
    file_mode = "a" if append else "w"
    print(f"Đang phát luồng dữ liệu ({len(df):,} tin nhắn) tới {STREAM_FILE}:")

    sent = 0
    anomaly_sent = 0

    try:
        with open(STREAM_FILE, file_mode, encoding="utf-8") as f:
            for idx, row in df.iterrows():
                timestamp = idx.strftime("%Y-%m-%d %H:%M:%S")
                message = row_to_json(row, timestamp)

                f.write(json.dumps(message) + "\n")
                f.flush()

                sent += 1
                if message["is_anomaly"] == 1:
                    anomaly_sent += 1

                print(f"  [{sent:>5}/{len(df)}] {timestamp} | Power: {message['Global_active_power']:>7.3f} kW | {message['anomaly_type']}")
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\nĐã dừng phát luồng (Ctrl+C).")

    finally:
        print(f"\nTổng kết: Đã gửi {sent:,}/{len(df):,} tin nhắn | Số điểm bất thường: {anomaly_sent:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Producer phát luồng dữ liệu Smart Meter")
    parser.add_argument("--speed", type=float, default=SEND_INTERVAL, help="Tốc độ gửi (giây/tin nhắn)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số tin nhắn")
    parser.add_argument("--skip", type=int, default=0, help="Bỏ qua N tin nhắn đầu")
    parser.add_argument("--append", action="store_true", help="Ghi tiếp vào file hiện tại")

    args = parser.parse_args()
    demo_df = load_demo_with_anomalies()

    if 0 < args.skip < len(demo_df):
        demo_df = demo_df.iloc[args.skip:]

    if args.limit > 0:
        demo_df = demo_df.head(args.limit)

    run_file_producer(demo_df, interval=args.speed, append=args.append)
