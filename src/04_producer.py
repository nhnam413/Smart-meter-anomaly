"""
============================================================
04_producer.py — File Producer (Giả lập Smart Meter Stream)
============================================================
Mục đích:
    - Đọc tập demo_with_anomalies.csv (đã bơm lỗi giả lập).
    - Đóng gói mỗi dòng dữ liệu thành 1 message JSON.
    - Ghi ra file JSONL (stream_buffer.jsonl) mỗi giây 1 lần.
    - Giả lập luồng dữ liệu thời gian thực từ đồng hồ điện thông minh.

Kiến trúc:
    [demo_with_anomalies.csv]
        → 04_producer.py (đọc từng dòng, mỗi giây 1 lần)
            → stream_buffer.jsonl
                → 05_dashboard.py (đọc file + Streamlit)

Tác giả: Sinh viên + AI Advisor
Ngày tạo: 2026-07-12
============================================================
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from datetime import datetime

# UTF-8 Encoding Fix for Windows Console / Subprocess Output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ============================================================
# 1. CẤU HÌNH
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(PROJECT_DIR, "data", "demo")
STREAM_FILE = os.path.join(DEMO_DIR, "stream_buffer.jsonl")

# Tốc độ gửi: mỗi bao nhiêu giây gửi 1 message
SEND_INTERVAL = 1.0  # giây


def load_demo_with_anomalies() -> pd.DataFrame:
    """
    Đọc tập demo đã bơm anomaly từ bước 03.

    Returns:
        pd.DataFrame với index là datetime
    """
    demo_path = os.path.join(DEMO_DIR, "demo_with_anomalies.csv")

    if not os.path.exists(demo_path):
        print(f"❌ Không tìm thấy: {demo_path}")
        print("   → Hãy chạy 03_inject_anomalies.py trước!")
        sys.exit(1)

    df = pd.read_csv(demo_path, index_col="datetime", parse_dates=True)
    return df


def row_to_json(row: pd.Series, timestamp: str) -> dict:
    """Chuyển 1 dòng DataFrame thành dict JSON với các trường sensor + metadata."""
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


# ============================================================
# FILE PRODUCER — Ghi ra JSONL (giả lập real-time streaming)
# ============================================================

def run_file_producer(df: pd.DataFrame, append: bool = False) -> None:
    """
    Ghi dữ liệu ra file JSONL (JSON Lines) — mỗi dòng 1 message.
    """
    mode_str = "Nối tiếp (APPEND)" if append else "Mới (OVERWRITE)"
    print(f"📁 Chế độ FILE ({mode_str}) — Ghi ra: {STREAM_FILE}")
    print(f"   Mỗi giây ghi 1 message (giả lập real-time)...\n")

    sent = 0
    anomaly_sent = 0
    file_mode = "a" if append else "w"

    try:
        with open(STREAM_FILE, file_mode, encoding="utf-8") as f:
            for idx, row in df.iterrows():
                timestamp = idx.strftime("%Y-%m-%d %H:%M:%S")
                message = row_to_json(row, timestamp)

                # Ghi 1 dòng JSON + xuống dòng
                f.write(json.dumps(message) + "\n")
                f.flush()  # Flush ngay để Dashboard đọc được

                sent += 1
                is_anom = message["is_anomaly"] == 1
                if is_anom:
                    anomaly_sent += 1

                print(f"   [{sent:>5}/{len(df)}] "
                      f"{'\ud83d\udd34 ANOMALY' if is_anom else '\ud83d\udfe2 Normal '} | "
                      f"{timestamp} | Power: {message['Global_active_power']:>7.3f} kW | "
                      f"{message['anomaly_type']}")

                time.sleep(SEND_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n⚠️ Dừng bởi người dùng (Ctrl+C)")

    finally:
        print(f"\n{'=' * 60}")
        print(f"📊 Kết quả: Đã ghi {sent:,} messages ({anomaly_sent} anomalies)")
        print(f"   File: {STREAM_FILE}")
        print(f"{'=' * 60}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Smart Meter File Producer — Giả lập streaming"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Tốc độ gửi (giây/message). Mặc định: 1.0"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số message gửi (0 = gửi hết). Mặc định: 0"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Bỏ qua N mẫu đầu tiên (dùng khi Resume/Tiếp tục chạy)"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Ghi tiếp vào file thay vì ghi đè từ đầu"
    )

    args = parser.parse_args()
    SEND_INTERVAL = args.speed

    print()
    print("📡 SMART METER ANOMALY DETECTION — FILE PRODUCER")
    print("=" * 60)
    print(f"   Tốc độ   : {SEND_INTERVAL}s / message")
    print(f"   Bỏ qua   : {args.skip} mẫu đầu")
    print(f"   Ghi file : {'APPEND (nối tiếp)' if args.append else 'OVERWRITE (mới)'}")
    print("=" * 60)
    print()

    # --- Đọc dữ liệu ---
    df = load_demo_with_anomalies()
    print(f"   ✅ Đọc xong: {len(df):,} mẫu "
          f"({df['is_anomaly'].sum()} anomalies)\n")

    # Bỏ qua args.skip mẫu đầu nếu có
    if args.skip > 0 and args.skip < len(df):
        df = df.iloc[args.skip:]
        print(f"   ⏩ Đã tiếp tục từ vị trí mẫu thứ {args.skip + 1}\n")

    # Giới hạn số message nếu có --limit
    if args.limit > 0:
        df = df.head(args.limit)
        print(f"   ⚡ Giới hạn: chỉ gửi {len(df)} messages\n")

    # --- Chạy Producer ---
    run_file_producer(df, append=args.append)

    print("\n🎉 Producer hoàn tất!")
    print("   → Mở terminal khác chạy: streamlit run src/05_dashboard.py")
    print()
