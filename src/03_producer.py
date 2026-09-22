

import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

from config import (
    DATETIME_FORMAT,
    DEMO_STREAM_PATH,
    SEND_INTERVAL,
    STREAM_BUFFER_PATH,
    setup_encoding,
)

setup_encoding()


# Đọc dữ liệu Demo dùng cho luồng phát.
def load_demo_data() -> pd.DataFrame:
    if not os.path.exists(DEMO_STREAM_PATH):
        print(
            f"Loi: Khong tim thay file {DEMO_STREAM_PATH}. Chay 01_data_prep.py truoc!"
        )
        sys.exit(1)
    return pd.read_csv(DEMO_STREAM_PATH, index_col="datetime", parse_dates=True)


# Chuyển một bản ghi Demo thành thông điệp JSON.
def build_message(timestamp: pd.Timestamp, row: pd.Series) -> dict:
    message = {
        "timestamp": timestamp.strftime(DATETIME_FORMAT),
        "sent_at": datetime.now().strftime(DATETIME_FORMAT),
        "is_anomaly": int(row.get("is_anomaly", 0)),
        "anomaly_type": str(row.get("anomaly_type", "normal")),
    }
    for column in row.index:
        if column not in ("is_anomaly", "anomaly_type"):
            message[column] = round(float(row[column]), 4)
    return message


# Phát từng bản ghi Demo vào tệp JSONL.
def stream_data(speed: float, limit: int = 0, append: bool = False) -> None:
    df = load_demo_data()

    mode = "a" if append else "w"

    print(
        f"Bat dau phat luong ({len(df):,} ban ghi) toi {STREAM_BUFFER_PATH} [Toc do: {speed}s/ban ghi]..."
    )

    count = 0
    anom_count = 0

    try:
        with open(STREAM_BUFFER_PATH, mode, encoding="utf-8") as f:
            for timestamp, row in df.iterrows():
                msg = build_message(timestamp, row)

                f.write(json.dumps(msg) + "\n")
                f.flush()

                count += 1
                if msg["is_anomaly"] == 1:
                    anom_count += 1

                print(
                    f"  [{count:>4}/{len(df)}] {msg['timestamp']} | P={msg['Global_active_power']:>6.3f}kW | V={msg['Voltage']:>5.1f}V | {msg['anomaly_type']}"
                )

                if limit > 0 and count >= limit:
                    break
                time.sleep(speed)

    except KeyboardInterrupt:
        print("\nDa dung phat luong du lieu.")
    finally:
        print(
            f"\nTong ket: Da phat {count:,} ban ghi | So diem bat thuong: {anom_count:,}"
        )


# Đọc tham số dòng lệnh của producer.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Stream Producer")
    parser.add_argument(
        "--speed",
        type=float,
        default=SEND_INTERVAL,
        help="Khoang nghi giua cac tin (giay)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Gioi han so ban ghi phat (0 = tat ca)"
    )
    parser.add_argument(
        "--append", action="store_true", help="Ghi tiep vao file buffer hien co"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    stream_data(speed=args.speed, limit=args.limit, append=args.append)
