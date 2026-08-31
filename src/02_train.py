"""
02_train.py - Pipeline huan luyen mo hinh Isolation Forest va danh gia tren tap Demo.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score

from config import (
    MODELS_DIR,
    TRAIN_HOURLY_PATH, DEMO_STREAM_PATH,
    MODEL_BUNDLE_PATH,
    ENGINEERED_FEATURE_NAMES,
    setup_encoding,
)
from features import extract_features, calc_baseline_stats

setup_encoding()


def evaluate_model(model: IsolationForest, scaler: RobustScaler, df_demo_labeled: pd.DataFrame):
    """Danh gia hieu nang mo hinh tren tap Demo (ROC-AUC, Precision, Recall, F1)."""
    print("\n[3/3] Danh gia hieu nang mo hinh tren tap Demo:")
    df_feat = extract_features(df_demo_labeled)
    X = scaler.transform(df_feat[ENGINEERED_FEATURE_NAMES].values)

    preds = (model.predict(X) == -1).astype(int)
    scores = model.decision_function(X)

    common_idx = df_feat.index.intersection(df_demo_labeled.index)
    y_true = df_demo_labeled.loc[common_idx, "is_anomaly"].values

    roc_auc = roc_auc_score(y_true, -scores)
    print(f"   ROC-AUC Score: {roc_auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, preds, target_names=["Bình thường", "Bất thường"], digits=4))


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Doc du lieu Train va trich xuat dac trung
    if not os.path.exists(TRAIN_HOURLY_PATH):
        print(f"Loi: Khong tim thay {TRAIN_HOURLY_PATH}. Chay 01_data_prep.py truoc!")
        sys.exit(1)

    print("[1/3] Doc tap Train va trich xuat 9 dac trung...")
    df_train = pd.read_csv(TRAIN_HOURLY_PATH, index_col="datetime", parse_dates=True)
    df_train_feats = extract_features(df_train)
    print(f"  -> Da doc {len(df_train):,} mau Train, trich xuat {len(df_train_feats):,} ban ghi dac trung.")

    # 2. Huan luyen Isolation Forest
    print("[2/3] Huan luyen mo hinh Isolation Forest...")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(df_train_feats[ENGINEERED_FEATURE_NAMES].values)

    model = IsolationForest(
        n_estimators=200,
        max_samples=512,
        max_features=1.0,
        contamination=0.08,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # Luu Model Bundle
    medians, iqrs = calc_baseline_stats(df_train_feats)
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": ENGINEERED_FEATURE_NAMES,
        "medians": medians,
        "iqrs": iqrs,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"  -> Da luu Model Bundle vao: {MODEL_BUNDLE_PATH}")

    # 3. Danh gia mo hinh
    if os.path.exists(DEMO_STREAM_PATH):
        df_demo_labeled = pd.read_csv(DEMO_STREAM_PATH, index_col="datetime", parse_dates=True)
        evaluate_model(model, scaler, df_demo_labeled)

    print("Hoan tat huan luyen va danh gia mo hinh.")


if __name__ == "__main__":
    main()
