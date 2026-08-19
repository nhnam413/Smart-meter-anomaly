"""
06_evaluate_model.py — Đánh giá hiệu năng mô hình (Model Evaluation & Metrics)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from config import MODELS_DIR, DEMO_DIR, SENSOR_COLUMNS, REPORTS_DIR
from features import create_features


def load_model():
    """Tải mô hình, bộ chuẩn hóa và danh sách đặc trưng."""
    model_path = os.path.join(MODELS_DIR, "isolation_forest_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    for path in [model_path, scaler_path, features_path]:
        if not os.path.exists(path):
            print(f"Lỗi: Không tìm thấy file tại {path}")
            sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    print(f"[1/4] Tải Artifacts: Model={type(model).__name__} | Scaler={type(scaler).__name__} | Features={len(feature_names)}")
    return model, scaler, feature_names


def load_demo_with_anomalies() -> pd.DataFrame:
    """Tải tập dữ liệu demo chứa ground truth bất thường."""
    demo_path = os.path.join(DEMO_DIR, "demo_with_anomalies.csv")
    if not os.path.exists(demo_path):
        print(f"Lỗi: Không tìm thấy file tại {demo_path}")
        sys.exit(1)

    df = pd.read_csv(demo_path, index_col="datetime", parse_dates=True)
    print(f"  Demo Data: {len(df):,} mẫu | GT Anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.1f}%)")
    return df


def run_prediction(df: pd.DataFrame, model, scaler, feature_names: list[str]) -> pd.DataFrame:
    """Dự đoán nhãn bất thường sử dụng Isolation Forest."""
    df_feat = create_features(df[SENSOR_COLUMNS])
    available_cols = [c for c in feature_names if c in df_feat.columns]

    X = df_feat[available_cols].values
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df_feat["predicted_anomaly"] = (predictions == -1).astype(int)
    df_feat["anomaly_score"] = scores

    common_idx = df_feat.index.intersection(df.index)
    df_feat["is_anomaly"] = df.loc[common_idx, "is_anomaly"]
    df_feat["anomaly_type"] = df.loc[common_idx, "anomaly_type"]

    n_pred = df_feat["predicted_anomaly"].sum()
    print(f"[2/4] Dự đoán: {len(df_feat):,} mẫu | Phát hiện: {n_pred:,} ({n_pred/len(df_feat)*100:.1f}%)")
    return df_feat


def compute_metrics(df: pd.DataFrame) -> dict:
    """Tính toán toàn bộ các chỉ số đo lường hiệu năng."""
    y_true = df["is_anomaly"].values
    y_pred = df["predicted_anomaly"].values
    scores = df["anomaly_score"].values

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, -scores)
    except ValueError:
        roc_auc = 0.0

    try:
        avg_precision = average_precision_score(y_true, -scores)
    except ValueError:
        avg_precision = 0.0

    precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_true, -scores)
    f1_scores_curve = 2 * (precisions_curve * recalls_curve) / (precisions_curve + recalls_curve + 1e-10)
    best_idx = np.argmax(f1_scores_curve)

    return {
        "confusion_matrix": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "opt_thresh_score": -thresholds_curve[best_idx] if best_idx < len(thresholds_curve) else 0.0,
        "opt_precision": precisions_curve[best_idx],
        "opt_recall": recalls_curve[best_idx],
        "opt_f1": f1_scores_curve[best_idx],
        "total": len(df),
        "total_gt_anomaly": int(y_true.sum()),
        "total_pred_anomaly": int(y_pred.sum()),
    }


def compute_per_type_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Tính tỷ lệ phát hiện theo từng loại bất thường."""
    results = []
    for atype in ["power_surge", "voltage_drop", "night_spike"]:
        mask = df["anomaly_type"] == atype
        if mask.sum() == 0:
            continue

        subset = df[mask]
        total = len(subset)
        detected = subset["predicted_anomaly"].sum()
        recall = detected / total if total > 0 else 0

        results.append({
            "Anomaly_Type": atype,
            "GT_Samples": total,
            "Detected": int(detected),
            "Missed": total - int(detected),
            "Detection_Rate": f"{recall * 100:.1f}%",
        })

    return pd.DataFrame(results)


def print_report(metrics: dict, per_type: pd.DataFrame, df: pd.DataFrame):
    """In báo cáo tóm tắt ra console."""
    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
    print(f"Tổng số mẫu: {metrics['total']:,} | Ground Truth: {metrics['total_gt_anomaly']:,} | Model phát hiện: {metrics['total_pred_anomaly']:,}")
    print(f"Confusion Matrix -> TN: {metrics['tn']:,} | FP: {metrics['fp']:,} | FN: {metrics['fn']:,} | TP: {metrics['tp']:,}")
    print(f"Chỉ số chính     -> Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['avg_precision']:.4f}")
    print(f"Ngưỡng tối ưu    -> Score: {metrics['opt_thresh_score']:+.4f} | Opt Precision: {metrics['opt_precision']:.4f} | Opt Recall: {metrics['opt_recall']:.4f} | Opt F1: {metrics['opt_f1']:.4f}")

    print("\nTỷ lệ phát hiện theo loại bất thường:")
    if not per_type.empty:
        print(per_type.to_string(index=False))

    print("\nClassification Report:")
    print(classification_report(
        df["is_anomaly"].values,
        df["predicted_anomaly"].values,
        target_names=["Normal", "Anomaly"],
        digits=4,
        zero_division=0,
    ))


def save_evaluation_charts(df: pd.DataFrame, metrics: dict, per_type: pd.DataFrame):
    """Tạo và lưu biểu đồ đánh giá ra file HTML."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Lưu ý: Plotly chưa cài đặt, bỏ qua tạo biểu đồ HTML.")
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Confusion Matrix",
            "Anomaly Score Distribution",
            "Precision-Recall Curve",
            "Detection Rate by Type"
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "bar"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    cm = metrics["confusion_matrix"]
    total = cm.sum()
    annotations = [[f"{cm[i][j]:,}\n({cm[i][j]/total*100:.1f}%)"
                     for j in range(2)] for i in range(2)]

    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=["Pred: Normal", "Pred: Anomaly"],
            y=["GT: Normal", "GT: Anomaly"],
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorscale=[[0, "#EFF6FF"], [1, "#2563EB"]],
            showscale=False,
        ),
        row=1, col=1,
    )

    normal_scores = df.loc[df["is_anomaly"] == 0, "anomaly_score"]
    anomaly_scores = df.loc[df["is_anomaly"] == 1, "anomaly_score"]

    fig.add_trace(
        go.Histogram(x=normal_scores, name="Normal", marker_color="#2563EB", opacity=0.7, nbinsx=50),
        row=1, col=2,
    )
    fig.add_trace(
        go.Histogram(x=anomaly_scores, name="Anomaly (GT)", marker_color="#EF4444", opacity=0.7, nbinsx=50),
        row=1, col=2,
    )
    fig.update_layout(barmode="overlay")

    precision_vals, recall_vals, _ = precision_recall_curve(df["is_anomaly"].values, -df["anomaly_score"].values)
    fig.add_trace(
        go.Scatter(
            x=recall_vals, y=precision_vals, mode="lines",
            name=f"PR Curve (AP={metrics['avg_precision']:.3f})",
            line=dict(color="#10B981", width=2),
            fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.1)",
        ),
        row=2, col=1,
    )

    if not per_type.empty:
        bar_labels = per_type["Anomaly_Type"].tolist()
        bar_detected = per_type["Detected"].values
        bar_missed = per_type["Missed"].values

        fig.add_trace(
            go.Bar(x=bar_labels, y=bar_detected, name="Detected", marker_color="#10B981", text=[str(d) for d in bar_detected], textposition="inside"),
            row=2, col=2,
        )
        fig.add_trace(
            go.Bar(x=bar_labels, y=bar_missed, name="Missed", marker_color="#EF4444", text=[str(m) for m in bar_missed], textposition="inside"),
            row=2, col=2,
        )
        fig.update_layout(barmode="stack")

    fig.update_layout(
        title=dict(
            text=f"Model Evaluation Report — Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f} | F1={metrics['f1']:.3f} | ROC-AUC={metrics['roc_auc']:.3f}",
            font=dict(size=15),
        ),
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        height=800,
        width=1100,
        showlegend=True,
    )

    html_path = os.path.join(REPORTS_DIR, "evaluation_report.html")
    fig.write_html(html_path)
    print(f"[3/4] Đã lưu báo cáo HTML: {html_path}")


def save_metrics_csv(metrics: dict, per_type: pd.DataFrame):
    """Lưu tóm tắt chỉ số hiệu năng ra các file CSV."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    summary = pd.DataFrame([
        {"Metric": "Precision", "Value": f"{metrics['precision']:.2f}"},
        {"Metric": "Recall", "Value": f"{metrics['recall']:.2f}"},
        {"Metric": "F1-Score", "Value": f"{metrics['f1']:.2f}"},
        {"Metric": "ROC-AUC", "Value": f"{metrics['roc_auc']:.2f}"},
        {"Metric": "PR-AUC", "Value": f"{metrics['avg_precision']:.2f}"},
        {"Metric": "True Positives", "Value": str(metrics['tp'])},
        {"Metric": "False Positives", "Value": str(metrics['fp'])},
        {"Metric": "True Negatives", "Value": str(metrics['tn'])},
        {"Metric": "False Negatives", "Value": str(metrics['fn'])},
    ])

    summary_path = os.path.join(REPORTS_DIR, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)

    if not per_type.empty:
        per_type_path = os.path.join(REPORTS_DIR, "metrics_per_type.csv")
        per_type.to_csv(per_type_path, index=False)

    print(f"[4/4] Đã lưu bảng chỉ số CSV: {summary_path}")


if __name__ == "__main__":
    trained_model, fitted_scaler, feature_list = load_model()
    demo_data = load_demo_with_anomalies()
    eval_df = run_prediction(demo_data, trained_model, fitted_scaler, feature_list)
    overall_metrics = compute_metrics(eval_df)
    breakdown_metrics = compute_per_type_metrics(eval_df)
    print_report(overall_metrics, breakdown_metrics, eval_df)
    save_evaluation_charts(eval_df, overall_metrics, breakdown_metrics)
    save_metrics_csv(overall_metrics, breakdown_metrics)
    print("Hoàn tất đánh giá mô hình.")
