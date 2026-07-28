"""
============================================================
02_train_model.py — Feature Engineering & Huấn luyện Model
============================================================
Mục đích:
    - Đọc tập train đã tiền xử lý (train.csv).
    - Tạo các đặc trưng mới (Feature Engineering):
        + Cyclical Encoding: Mã hóa giờ trong ngày bằng sin/cos.
        + Lag Features: Giá trị công suất ở các thời điểm trước đó.
        + Rolling Statistics: Trung bình & độ lệch chuẩn trượt.
    - Huấn luyện model Isolation Forest (Unsupervised).
    - Lưu model (.pkl) và danh sách feature (.pkl).

Tác giả: Sinh viên + AI Advisor
Ngày tạo: 2026-07-12
============================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Tạo thư mục models nếu chưa có
os.makedirs(MODELS_DIR, exist_ok=True)


def load_train_data() -> pd.DataFrame:
    """
    Đọc tập train đã tiền xử lý từ bước 01.

    Returns:
        pd.DataFrame với index là datetime
    """
    print("=" * 60)
    print("📂 BƯỚC 1: Đọc tập Train...")
    print("=" * 60)

    train_path = os.path.join(PROCESSED_DIR, "train.csv")

    if not os.path.exists(train_path):
        print(f"❌ Không tìm thấy: {train_path}")
        print("   → Hãy chạy 01_prepare_data.py trước!")
        sys.exit(1)

    df = pd.read_csv(train_path, index_col="datetime", parse_dates=True)

    print(f"   ✅ Đọc xong: {df.shape[0]:,} dòng × {df.shape[1]} cột")
    print(f"   Khoảng thời gian: {df.index.min()} → {df.index.max()}")
    print()

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering — Tạo các đặc trưng mới từ dữ liệu thô.

    Gồm 3 nhóm chính:

    ─────────────────────────────────────────────────────────
    NHÓM 1: CYCLICAL ENCODING (Mã hóa vòng tròn cho giờ)
    ─────────────────────────────────────────────────────────
    Tại sao cần?
        - Giờ trong ngày (0-23) là biến TUẦN HOÀN: giờ 23 và giờ 0
          thực tế rất gần nhau, nhưng nếu dùng giá trị thô (23 vs 0)
          thì model hiểu sai là chúng cách xa nhau.
        - Giải pháp: Chiếu lên đường tròn bằng sin/cos.

    Công thức toán học:
        hour_sin = sin(2π × hour / 24)
        hour_cos = cos(2π × hour / 24)

        Ví dụ: giờ 0  → (sin=0.00, cos=1.00)
               giờ 6  → (sin=1.00, cos=0.00)
               giờ 12 → (sin=0.00, cos=-1.00)
               giờ 23 → (sin=-0.27, cos=0.97) ← gần với giờ 0!

    ─────────────────────────────────────────────────────────
    NHÓM 2: LAG FEATURES (Đặc trưng trễ)
    ─────────────────────────────────────────────────────────
    Tại sao cần?
        - Công suất tiêu thụ có tính TỰ TƯƠNG QUAN (autocorrelation):
          giá trị hiện tại phụ thuộc vào giá trị quá khứ.
        - Lag-1h: Công suất 1 giờ trước → bắt xu hướng ngắn hạn.
        - Lag-24h: Công suất cùng giờ hôm qua → bắt chu kỳ ngày.

    ─────────────────────────────────────────────────────────
    NHÓM 3: ROLLING STATISTICS (Thống kê trượt)
    ─────────────────────────────────────────────────────────
    Tại sao cần?
        - Rolling Mean (6h): Xu hướng trung bình trong 6 giờ gần nhất.
          Giúp model nhận biết mức nền (baseline) tiêu thụ.
        - Rolling Std (6h): Độ biến động trong 6 giờ gần nhất.
          Giá trị cao = tiêu thụ dao động mạnh → có thể bất thường.

    Args:
        df: DataFrame gốc (chỉ có 7 cột sensor)

    Returns:
        DataFrame đã bổ sung các feature mới, loại bỏ hàng NaN do shift/rolling
    """
    print("=" * 60)
    print("⚙️  BƯỚC 2: Feature Engineering...")
    print("=" * 60)

    df = df.copy()

    # ── NHÓM 1: Cyclical Encoding cho giờ trong ngày ──────────
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    print("   ✅ Cyclical Encoding: hour_sin, hour_cos")

    # Thêm: Ngày trong tuần (0=Mon, 6=Sun) — cũng mã hóa cyclical
    day_of_week = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    print("   ✅ Cyclical Encoding: dow_sin, dow_cos (ngày trong tuần)")

    # ── NHÓM 2: Lag Features ──────────────────────────────────
    target_col = "Global_active_power"

    df["power_lag_1h"] = df[target_col].shift(1)    # 1 giờ trước
    df["power_lag_2h"] = df[target_col].shift(2)    # 2 giờ trước
    df["power_lag_24h"] = df[target_col].shift(24)  # Cùng giờ hôm qua
    print("   ✅ Lag Features: 1h, 2h, 24h")

    # ── NHÓM 3: Rolling Statistics (cửa sổ 6 giờ) ────────────
    df["power_rolling_mean_6h"] = (
        df[target_col].rolling(window=6, min_periods=1).mean()
    )
    df["power_rolling_std_6h"] = (
        df[target_col].rolling(window=6, min_periods=1).std()
    )
    print("   ✅ Rolling Statistics: mean_6h, std_6h")

    # ── Loại bỏ hàng NaN do shift (lag) ──────────────────────
    before = len(df)
    df = df.dropna()
    after = len(df)

    print(f"\n   Số feature mới   : 8")
    print(f"   Tổng feature     : {df.shape[1]}")
    print(f"   Hàng bị loại NaN : {before - after}")
    print(f"   Dữ liệu còn lại : {after:,} mẫu")
    print()

    return df


def train_isolation_forest(df: pd.DataFrame,
                           contamination: float = 0.05,
                           n_estimators: int = 200,
                           random_state: int = 42) -> tuple:
    """
    Huấn luyện model Isolation Forest.

    ─────────────────────────────────────────────────────────
    ISOLATION FOREST — Giải thích cho báo cáo:
    ─────────────────────────────────────────────────────────

    Nguyên lý hoạt động:
        - Isolation Forest dựa trên ý tưởng: điểm bất thường (anomaly)
          thường BỊ CÔ LẬP nhanh hơn điểm bình thường.

        - Thuật toán xây dựng nhiều cây quyết định ngẫu nhiên (iTree).
          Mỗi cây chia dữ liệu bằng cách chọn NGẪU NHIÊN 1 feature
          và 1 ngưỡng cắt.

        - Điểm bất thường nằm xa phân phối chính → chỉ cần ít lần
          chia là bị tách ra → có path_length ngắn.

        - Anomaly Score = trung bình path_length trên tất cả các cây.
          Score thấp → bất thường. Score cao → bình thường.

    Ưu điểm so với các phương pháp khác:
        - KHÔNG cần label (Unsupervised) → phù hợp bài toán thực tế.
        - Tuyến tính với kích thước dữ liệu: O(n × log(n)).
        - Hoạt động tốt với dữ liệu nhiều chiều (high-dimensional).

    Tham số quan trọng:
        - contamination (0.05): Ước lượng tỷ lệ anomaly trong dữ liệu.
          5% nghĩa là model kỳ vọng ~5% mẫu là bất thường.
        - n_estimators (200): Số cây trong rừng. Càng nhiều → ổn định hơn.
        - random_state (42): Seed để tái tạo kết quả (reproducible).

    Args:
        df              : DataFrame đã có đầy đủ features
        contamination   : Tỷ lệ bất thường ước lượng (default=5%)
        n_estimators    : Số cây Isolation Tree (default=200)
        random_state    : Seed ngẫu nhiên

    Returns:
        (model, scaler, feature_names) — model đã train, scaler, tên features
    """
    print("=" * 60)
    print("🤖 BƯỚC 3: Huấn luyện Isolation Forest...")
    print("=" * 60)

    # ── Chọn các feature đầu vào cho model ───────────────────
    # Chỉ dùng các feature đã tạo + cột sensor gốc
    # KHÔNG dùng index (datetime) làm feature
    feature_names = df.columns.tolist()

    print(f"   Số features      : {len(feature_names)}")
    print(f"   Danh sách features:")
    for i, f in enumerate(feature_names, 1):
        print(f"      {i:2d}. {f}")
    print()

    X = df[feature_names].values

    # ── Chuẩn hóa dữ liệu (StandardScaler) ──────────────────
    # Tại sao cần chuẩn hóa?
    #   - Các feature có scale rất khác nhau:
    #     Voltage ~240V, Sub_metering ~0-20 Wh, hour_sin ∈ [-1, 1]
    #   - StandardScaler: z = (x - mean) / std → đưa về mean=0, std=1
    #   - Giúp Isolation Forest chia đều trên các chiều, không bị bias
    #     bởi feature có giá trị lớn.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"   ✅ Chuẩn hóa StandardScaler: mean=0, std=1")

    # ── Huấn luyện Isolation Forest ──────────────────────────
    print(f"\n   Đang huấn luyện Isolation Forest...")
    print(f"      n_estimators   = {n_estimators}")
    print(f"      contamination  = {contamination}")
    print(f"      random_state   = {random_state}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,          # Sử dụng tất cả CPU cores
        verbose=0
    )

    model.fit(X_scaled)

    # ── Đánh giá nhanh trên tập train ────────────────────────
    # predict() trả về: 1 = bình thường, -1 = bất thường
    y_pred = model.predict(X_scaled)
    n_anomalies = (y_pred == -1).sum()
    n_normal = (y_pred == 1).sum()

    print(f"\n   ✅ Huấn luyện hoàn tất!")
    print(f"   Kết quả trên tập Train:")
    print(f"      Bình thường : {n_normal:,} mẫu ({n_normal/len(y_pred)*100:.1f}%)")
    print(f"      Bất thường  : {n_anomalies:,} mẫu ({n_anomalies/len(y_pred)*100:.1f}%)")
    print()

    return model, scaler, feature_names


def save_artifacts(model, scaler, feature_names: list) -> None:
    """
    Lưu model, scaler và danh sách feature ra file .pkl.
    Scaler và feature_names cần giữ đồng bộ với lúc train để inference đúng.
    """
    print("=" * 60)
    print("💾 BƯỚC 4: Lưu Model & Artifacts...")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "isolation_forest_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)

    # In kích thước file
    for name, path in [("Model", model_path),
                       ("Scaler", scaler_path),
                       ("Features", features_path)]:
        size_kb = os.path.getsize(path) / 1024
        print(f"   ✅ {name:10s}: {path} ({size_kb:.1f} KB)")

    print()


# ============================================================
# MAIN — Chạy toàn bộ pipeline huấn luyện
# ============================================================

if __name__ == "__main__":
    print()
    print("🤖 SMART METER ANOMALY DETECTION — HUẤN LUYỆN MODEL")
    print("=" * 60)
    print()

    # --- Pipeline ---
    df_train = load_train_data()                           # Bước 1
    df_featured = create_features(df_train)                # Bước 2
    model, scaler, features = train_isolation_forest(      # Bước 3
        df_featured,
        contamination=0.05,
        n_estimators=200,
        random_state=42
    )
    save_artifacts(model, scaler, features)                # Bước 4

    print("🎉 HOÀN TẤT! Model đã sẵn sàng.")
    print("   → Chạy tiếp: python src/03_inject_anomalies.py")
    print()
