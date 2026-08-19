"""
02_train_model.py — Huấn luyện mô hình Isolation Forest
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from config import (
    PROCESSED_DIR, MODELS_DIR,
    CONTAMINATION_FILE, CONTAMINATION_TARGET, CONTAMINATION_MIN, CONTAMINATION_MAX,
    N_ESTIMATORS, MAX_SAMPLES, MAX_FEATURES, RANDOM_STATE, setup_encoding,
)

setup_encoding()

from features import create_features


os.makedirs(MODELS_DIR, exist_ok=True)


def load_contamination() -> float:
    """Tải tham số contamination đã lưu từ file."""
    if not os.path.exists(CONTAMINATION_FILE):
        return CONTAMINATION_TARGET

    with open(CONTAMINATION_FILE, "r", encoding="utf-8") as f:
        raw = float(f.read().strip())

    return max(CONTAMINATION_MIN, min(CONTAMINATION_MAX, raw))


def load_train_data() -> pd.DataFrame:
    """Đọc dữ liệu huấn luyện đã xử lý."""
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    if not os.path.exists(train_path):
        print(f"Lỗi: Không tìm thấy file {train_path}")
        sys.exit(1)

    df = pd.read_csv(train_path, index_col="datetime", parse_dates=True)
    print(f"1. Đã tải dữ liệu huấn luyện: {df.shape[0]:,} mẫu")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng mới cho tập huấn luyện."""
    df_featured = create_features(df)
    print(f"2. Tạo đặc trưng hoàn tất: {df_featured.shape[1]} cột đặc trưng")
    return df_featured


def train_isolation_forest(df: pd.DataFrame) -> tuple[IsolationForest, RobustScaler, list[str]]:
    """Huấn luyện mô hình Isolation Forest với RobustScaler."""
    contamination = load_contamination()
    feature_names = df.columns.tolist()
    X = df[feature_names].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        max_features=MAX_FEATURES,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_scaled)

    y_pred = model.predict(X_scaled)
    n_anomalies = int((y_pred == -1).sum())
    print(f"3. Huấn luyện IsolationForest thành công: {n_anomalies:,} mẫu bất thường ({n_anomalies/len(y_pred)*100:.1f}%)")

    return model, scaler, feature_names


def save_artifacts(model: IsolationForest, scaler: RobustScaler, feature_names: list[str]) -> None:
    """Lưu mô hình, bộ chuẩn hóa và danh sách tên đặc trưng."""
    joblib.dump(model, os.path.join(MODELS_DIR, "isolation_forest_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    print("4. Đã lưu thành công mô hình vào thư mục models/\n")


if __name__ == "__main__":
    df_train = load_train_data()
    df_featured = run_feature_engineering(df_train)
    trained_model, fitted_scaler, feat_names = train_isolation_forest(df_featured)
    save_artifacts(trained_model, fitted_scaler, feat_names)
    print("Hoàn tất quá trình huấn luyện mô hình.")
