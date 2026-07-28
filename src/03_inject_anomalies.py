"""
============================================================
03_inject_anomalies.py — Bơm lỗi giả lập vào tập Demo
============================================================
Mục đích:
    - Đọc tập demo (demo.csv) — dữ liệu hoàn toàn bình thường.
    - Bơm các loại bất thường (anomaly) nhân tạo để kiểm thử:
        + Power Surge   : Đột biến công suất (tăng gấp 3-5 lần).
        + Voltage Drop  : Sụt áp đột ngột (giảm 20-40V).
        + Night Spike   : Tiêu thụ cao bất thường vào ban đêm (1h-5h).
    - Thêm cột 'is_anomaly' (0/1) và 'anomaly_type' để đánh dấu.
    - Lưu file demo_with_anomalies.csv để dùng cho Producer (bước 04).

Tại sao cần bơm lỗi giả lập?
    - Trong Unsupervised Learning, ta KHÔNG có nhãn (label) thật.
    - Để ĐÁNH GIÁ và DEMO hệ thống, cần tạo anomaly đã biết trước
      rồi kiểm tra xem model có phát hiện được không.
    - Đây là kỹ thuật phổ biến trong nghiên cứu anomaly detection,
      gọi là "synthetic anomaly injection".

Tác giả: Sinh viên + AI Advisor
Ngày tạo: 2026-07-12
============================================================
"""

import os
import sys
import pandas as pd
import numpy as np

# ============================================================
# 1. CẤU HÌNH
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(PROJECT_DIR, "data", "demo")

# Tỷ lệ bơm anomaly cho mỗi loại (% trên tổng số mẫu demo)
POWER_SURGE_RATIO = 0.03    # 3% mẫu bị Power Surge
VOLTAGE_DROP_RATIO = 0.03   # 3% mẫu bị Voltage Drop
NIGHT_SPIKE_RATIO = 0.02    # 2% mẫu bị Night Spike

# Seed ngẫu nhiên để kết quả tái tạo được
RANDOM_SEED = 42


def load_demo_data() -> pd.DataFrame:
    """
    Đọc tập demo từ file CSV.

    Returns:
        pd.DataFrame với index là datetime
    """
    print("=" * 60)
    print("📂 BƯỚC 1: Đọc tập Demo gốc...")
    print("=" * 60)

    demo_path = os.path.join(DEMO_DIR, "demo.csv")

    if not os.path.exists(demo_path):
        print(f"❌ Không tìm thấy: {demo_path}")
        print("   → Hãy chạy 01_prepare_data.py trước!")
        sys.exit(1)

    df = pd.read_csv(demo_path, index_col="datetime", parse_dates=True)

    # Thêm 2 cột đánh dấu anomaly (ban đầu tất cả = 0 / "normal")
    df["is_anomaly"] = 0
    df["anomaly_type"] = "normal"

    print(f"   ✅ Đọc xong: {df.shape[0]:,} dòng")
    print(f"   Khoảng thời gian: {df.index.min()} → {df.index.max()}")
    print()

    return df


def inject_power_surge(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Bơm anomaly loại 1: POWER SURGE (Đột biến công suất).

    ─────────────────────────────────────────────────────────
    Mô tả hiện tượng thực tế:
        - Thiết bị công suất lớn bật đồng thời (máy sưởi + lò nướng + máy giặt).
        - Lỗi thiết bị gây tiêu thụ quá mức.
        - Rò điện hoặc đoản mạch nhẹ.

    Cách giả lập:
        - Chọn ngẫu nhiên một số mẫu (theo POWER_SURGE_RATIO).
        - Nhân Global_active_power với hệ số 3x → 5x.
        - Tăng Global_intensity tương ứng (P = V × I).
    ─────────────────────────────────────────────────────────

    Args:
        df  : DataFrame gốc
        rng : Random generator (để tái tạo kết quả)

    Returns:
        DataFrame đã bơm Power Surge
    """
    print("=" * 60)
    print("⚡ BƯỚC 2a: Bơm Power Surge (Đột biến công suất)...")
    print("=" * 60)

    # Chỉ bơm vào các mẫu chưa bị đánh dấu anomaly
    normal_mask = df["is_anomaly"] == 0
    normal_indices = df.index[normal_mask]

    # Số lượng mẫu cần bơm
    n_inject = int(len(df) * POWER_SURGE_RATIO)

    # Chọn ngẫu nhiên các vị trí bơm
    inject_indices = rng.choice(normal_indices, size=n_inject, replace=False)

    # Hệ số nhân ngẫu nhiên: 3x → 5x
    multipliers = rng.uniform(3.0, 5.0, size=n_inject)

    for idx, mult in zip(inject_indices, multipliers):
        # Tăng công suất tác dụng
        df.loc[idx, "Global_active_power"] *= mult

        # Tăng cường độ dòng điện tương ứng (I = P / V)
        # Giữ nguyên Voltage vì surge là do tải, không phải lưới
        df.loc[idx, "Global_intensity"] *= mult

        # Đánh dấu
        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = "power_surge"

    # Thống kê
    avg_power_surge = df.loc[inject_indices, "Global_active_power"].mean()
    avg_power_normal = df.loc[df["is_anomaly"] == 0, "Global_active_power"].mean()

    print(f"   Số mẫu bơm        : {n_inject}")
    print(f"   Hệ số nhân        : 3.0x → 5.0x")
    print(f"   Công suất TB surge : {avg_power_surge:.2f} kW")
    print(f"   Công suất TB bình  : {avg_power_normal:.2f} kW")
    print(f"   Tỷ lệ surge/bình  : {avg_power_surge/avg_power_normal:.1f}x")
    print()

    return df


def inject_voltage_drop(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Bơm anomaly loại 2: VOLTAGE DROP (Sụt áp đột ngột).

    ─────────────────────────────────────────────────────────
    Mô tả hiện tượng thực tế:
        - Lưới điện quá tải → điện áp giảm (brownout).
        - Sự cố trạm biến áp → điện áp không ổn định.
        - Tiêu chuẩn điện áp châu Âu: 230V ± 10%.
          Nếu < 207V → bất thường, có thể gây hỏng thiết bị.

    Cách giả lập:
        - Chọn ngẫu nhiên một số mẫu.
        - Trừ Voltage đi 20V → 40V (giảm xuống ~195-220V).
        - Giá trị này nằm NGOÀI khoảng bình thường (~235-245V).
    ─────────────────────────────────────────────────────────
    """
    print("=" * 60)
    print("🔋 BƯỚC 2b: Bơm Voltage Drop (Sụt áp)...")
    print("=" * 60)

    normal_mask = df["is_anomaly"] == 0
    normal_indices = df.index[normal_mask]

    n_inject = int(len(df) * VOLTAGE_DROP_RATIO)
    inject_indices = rng.choice(normal_indices, size=n_inject, replace=False)

    # Mức sụt áp ngẫu nhiên: 20V → 40V
    voltage_drops = rng.uniform(20.0, 40.0, size=n_inject)

    for idx, drop in zip(inject_indices, voltage_drops):
        df.loc[idx, "Voltage"] -= drop
        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = "voltage_drop"

    avg_volt_drop = df.loc[inject_indices, "Voltage"].mean()
    avg_volt_normal = df.loc[df["is_anomaly"] == 0, "Voltage"].mean()

    print(f"   Số mẫu bơm            : {n_inject}")
    print(f"   Mức sụt áp             : 20V → 40V")
    print(f"   Điện áp TB sau sụt     : {avg_volt_drop:.1f}V")
    print(f"   Điện áp TB bình thường : {avg_volt_normal:.1f}V")
    print()

    return df


def inject_night_spike(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Bơm anomaly loại 3: NIGHT SPIKE (Tiêu thụ cao bất thường ban đêm).

    ─────────────────────────────────────────────────────────
    Mô tả hiện tượng thực tế:
        - Đây là CONTEXTUAL ANOMALY (bất thường theo ngữ cảnh):
          cùng một giá trị công suất có thể bình thường vào ban ngày
          nhưng BẤT THƯỜNG vào ban đêm (1h-5h sáng).
        - Ví dụ: tiêu thụ 3 kW lúc 14h → bình thường (nấu ăn).
                 tiêu thụ 3 kW lúc 3h sáng → bất thường (ai dùng?).

    Cách giả lập:
        - Chỉ chọn các mẫu trong khung giờ đêm (1h → 5h).
        - Tăng Global_active_power lên mức ban ngày (2x → 3x).
    ─────────────────────────────────────────────────────────

    Tại sao loại này quan trọng?
        - Nó khác Power Surge: giá trị tuyệt đối không cực đoan,
          nhưng BẤT THƯỜNG KHI XÉT THEO THỜI GIAN.
        - Isolation Forest kết hợp hour_sin/hour_cos sẽ nắm bắt
          mối quan hệ (giờ, công suất) → phát hiện được.
    """
    print("=" * 60)
    print("🌙 BƯỚC 2c: Bơm Night Spike (Bất thường ban đêm)...")
    print("=" * 60)

    # Lọc các mẫu ban đêm (1h → 5h) chưa bị đánh dấu
    night_mask = (
        (df.index.hour >= 1) &
        (df.index.hour <= 5) &
        (df["is_anomaly"] == 0)
    )
    night_indices = df.index[night_mask]

    if len(night_indices) == 0:
        print("   ⚠️ Không có mẫu ban đêm để bơm!")
        return df

    n_inject = min(int(len(df) * NIGHT_SPIKE_RATIO), len(night_indices))
    inject_indices = rng.choice(night_indices, size=n_inject, replace=False)

    # Tăng công suất 2x → 3x (mô phỏng mức tiêu thụ ban ngày)
    multipliers = rng.uniform(2.0, 3.0, size=n_inject)

    for idx, mult in zip(inject_indices, multipliers):
        df.loc[idx, "Global_active_power"] *= mult
        df.loc[idx, "Global_intensity"] *= mult
        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = "night_spike"

    print(f"   Số mẫu ban đêm có sẵn : {len(night_indices)}")
    print(f"   Số mẫu bơm            : {n_inject}")
    print(f"   Khung giờ              : 1h → 5h sáng")
    print(f"   Hệ số nhân            : 2.0x → 3.0x")
    print()

    return df


def save_demo_with_anomalies(df: pd.DataFrame) -> None:
    """
    Lưu tập demo đã bơm anomaly ra file CSV.

    Output:
        data/demo/demo_with_anomalies.csv
    """
    print("=" * 60)
    print("💾 BƯỚC 3: Lưu tập Demo + Anomalies...")
    print("=" * 60)

    output_path = os.path.join(DEMO_DIR, "demo_with_anomalies.csv")
    df.to_csv(output_path)

    # Thống kê tổng hợp
    total = len(df)
    n_anomaly = df["is_anomaly"].sum()
    n_normal = total - n_anomaly

    print(f"   ✅ Đã lưu: {output_path}")
    print()
    print("   ┌─────────────────────────────────────────────┐")
    print("   │         THỐNG KÊ TỔNG HỢP ANOMALY          │")
    print("   ├─────────────────────────────────────────────┤")
    print(f"   │  Tổng mẫu demo     : {total:>6,}               │")
    print(f"   │  Bình thường        : {n_normal:>6,} ({n_normal/total*100:.1f}%)        │")
    print(f"   │  Bất thường (tổng)  : {n_anomaly:>6,} ({n_anomaly/total*100:.1f}%)         │")
    print("   │  ─────────────────────────────────────────  │")

    # Chi tiết từng loại
    for atype in ["power_surge", "voltage_drop", "night_spike"]:
        count = (df["anomaly_type"] == atype).sum()
        icon = {"power_surge": "⚡", "voltage_drop": "🔋", "night_spike": "🌙"}[atype]
        label = {"power_surge": "Power Surge ",
                 "voltage_drop": "Voltage Drop",
                 "night_spike": "Night Spike "}[atype]
        print(f"   │  {icon} {label}      : {count:>6,} ({count/total*100:.1f}%)         │")

    print("   └─────────────────────────────────────────────┘")
    print()


# ============================================================
# MAIN — Chạy toàn bộ pipeline bơm anomaly
# ============================================================

if __name__ == "__main__":
    print()
    print("🧪 SMART METER ANOMALY DETECTION — BƠM LỖI GIẢ LẬP")
    print("=" * 60)
    print()

    # Khởi tạo Random Generator (reproducible)
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Pipeline ---
    df = load_demo_data()                     # Bước 1: Đọc demo gốc
    df = inject_power_surge(df, rng)          # Bước 2a: Power Surge
    df = inject_voltage_drop(df, rng)         # Bước 2b: Voltage Drop
    df = inject_night_spike(df, rng)          # Bước 2c: Night Spike
    save_demo_with_anomalies(df)              # Bước 3: Lưu kết quả

    print("🎉 HOÀN TẤT! Tập demo đã có anomaly để kiểm thử.")
    print("   → Chạy tiếp: python src/04_producer.py")
    print()
