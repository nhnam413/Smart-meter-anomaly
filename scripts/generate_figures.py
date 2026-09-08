import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Add src to path to import features
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import TRAIN_HOURLY_PATH, DEMO_STREAM_PATH, MODEL_BUNDLE_PATH, ENGINEERED_FEATURE_NAMES
from features import extract_features

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def save_fig(name):
    path = os.path.join(REPORTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

def generate_hinh_1_1(df_demo):
    # Find one instance of each anomaly type
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    types = ['power_surge', 'voltage_drop', 'night_spike']
    titles = ['Đột biến công suất (power_surge)', 'Sụt điện áp (voltage_drop)', 'Đột biến ban đêm (night_spike)']
    cols = ['Global_active_power', 'Voltage', 'Global_active_power']
    
    for ax, atype, title, col in zip(axes, types, titles, cols):
        anomalies = df_demo[df_demo['anomaly_type'] == atype]
        if not anomalies.empty:
            idx = anomalies.index[0]
            # Mở rộng khung cửa sổ lên 2 tuần (336 giờ)
            start_idx = df_demo.index.get_loc(idx) - 168
            end_idx = df_demo.index.get_loc(idx) + 168
            start_idx = max(0, start_idx)
            end_idx = min(len(df_demo), end_idx)
            
            sub_df = df_demo.iloc[start_idx:end_idx]
            window_anomalies = sub_df[sub_df['anomaly_type'] == atype]
            
            ax.plot(sub_df.index, sub_df[col], color='blue', label='Bình thường', alpha=0.7)
            ax.scatter(window_anomalies.index, window_anomalies[col], color='red', s=60, label='Sự cố', zorder=5)
            ax.set_title(title)
            ax.legend()
    save_fig("Hinh_1.1_MinhHoa_3_Loai_SuCo.png")

def generate_hinh_2_1(df_train):
    plt.figure(figsize=(8, 5))
    night_mask = (df_train.index.hour >= 1) & (df_train.index.hour <= 5)
    day_mask = ~night_mask
    sns.kdeplot(df_train[night_mask]['Global_active_power'], fill=True, label='Ban đêm (1h-5h)', color='blue', alpha=0.5)
    sns.kdeplot(df_train[day_mask]['Global_active_power'], fill=True, label='Ban ngày (6h-0h)', color='orange', alpha=0.5)
    plt.title("Phân phối đa đỉnh phi Gaussian của Công suất tác dụng")
    plt.xlabel("Công suất tác dụng (kW)")
    plt.ylabel("Mật độ phân phối")
    plt.legend()
    plt.xlim(-1, 6)
    save_fig("Hinh_2.1_PhanPhoi_DaDinh_PhiGaussian.png")

def generate_hinh_2_2(scores, y_true):
    plt.figure(figsize=(8, 5))
    normal_scores = -scores[y_true == 0]
    anomaly_scores = -scores[y_true == 1]
    sns.histplot(normal_scores, bins=50, color='blue', alpha=0.6, label='Bình thường', stat='density')
    sns.histplot(anomaly_scores, bins=50, color='red', alpha=0.6, label='Bất thường', stat='density')
    plt.title("Phân phối Điểm số Bất thường (Anomaly Score)")
    plt.xlabel("Score (Cao hơn là bất thường hơn)")
    plt.ylabel("Mật độ")
    plt.axvline(0, color='black', linestyle='--', label='Ngưỡng phân loại')
    plt.legend()
    save_fig("Hinh_2.2_PhanPhoi_AnomalyScore.png")

def generate_hinh_3_1(df_feat):
    plt.figure(figsize=(10, 8))
    corr = df_feat[ENGINEERED_FEATURE_NAMES].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Ma trận tương quan 9 Đặc trưng")
    save_fig("Hinh_3.1_MaTran_TuongQuan_9_DacTrung.png")

def generate_hinh_4_1(df_train, df_demo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Train vs Demo
    sizes1 = [len(df_train), len(df_demo)]
    ax1.pie(sizes1, labels=['Train (80%)', 'Demo (20%)'], autopct='%1.1f%%', colors=['#4C72B0', '#55A868'], startangle=90)
    ax1.set_title("Phân bổ dữ liệu huấn luyện và kiểm định")
    
    # Anomaly Types
    types = df_demo['anomaly_type'].value_counts()
    sizes2 = [types.get('normal', 0)] + [types.get(t, 0) for t in ['power_surge', 'voltage_drop', 'night_spike']]
    labels2 = ['Bình thường (92%)', 'Surge (3%)', 'Drop (3%)', 'Night Spike (2%)']
    colors2 = ['#DDDDDD', '#C44E52', '#8172B3', '#CCB974']
    ax2.pie(sizes2, labels=labels2, autopct='%1.1f%%', colors=colors2, startangle=90)
    ax2.set_title("Tỷ lệ tiêm lỗi giả lập (Tập Demo)")
    
    save_fig("Hinh_4.1_PhanBo_DuLieu.png")

def generate_hinh_5_1(scores, y_true):
    fpr, tpr, _ = roc_curve(y_true, -scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ cảnh báo giả (False Positive Rate)')
    plt.ylabel('Tỷ lệ phát hiện đúng (True Positive Rate)')
    plt.title('Đường cong ROC')
    plt.legend(loc="lower right")
    save_fig("Hinh_5.1_ROC_Curve.png")

def generate_hinh_5_2(preds, y_true):
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bình thường', 'Bất thường'], yticklabels=['Bình thường', 'Bất thường'])
    plt.ylabel('Thực tế (Ground Truth)')
    plt.xlabel('Dự đoán (Prediction)')
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    save_fig("Hinh_5.2_Confusion_Matrix.png")

def generate_hinh_5_3_fixed(df_demo_labeled, preds, common_idx):
    y_true = df_demo_labeled.loc[common_idx, "is_anomaly"].values
    anomaly_types = df_demo_labeled.loc[common_idx, "anomaly_type"].values
    
    unique_types = [t for t in sorted(list(set(anomaly_types))) if t != 'normal']
    recalls = {}
    
    for atype in unique_types:
        mask = (anomaly_types == atype)
        total = mask.sum()
        caught = (preds[mask] == 1).sum()
        recalls[atype] = (caught / total) * 100 if total > 0 else 0
        
    plt.figure(figsize=(8, 5))
    bars = plt.bar(recalls.keys(), recalls.values(), color=['#C44E52', '#8172B3', '#CCB974'])
    plt.ylim(0, 110)
    plt.ylabel('Recall (%)')
    plt.title('Tỷ lệ phát hiện (Recall) theo từng loại sự cố')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    save_fig("Hinh_5.3_Recall_Theo_Tung_Loai_Loi.png")

def main():
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_HOURLY_PATH, index_col="datetime", parse_dates=True)
    df_demo = pd.read_csv(DEMO_STREAM_PATH, index_col="datetime", parse_dates=True)
    
    print("Loading model...")
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle['model']
    scaler = bundle['scaler']
    
    print("Extracting features...")
    df_train_feats = extract_features(df_train)
    df_demo_feats = extract_features(df_demo)
    
    common_idx = df_demo_feats.index.intersection(df_demo.index)
    X_demo = scaler.transform(df_demo_feats.loc[common_idx, ENGINEERED_FEATURE_NAMES].values)
    
    preds = (model.predict(X_demo) == -1).astype(int)
    scores = model.decision_function(X_demo)
    y_true = df_demo.loc[common_idx, "is_anomaly"].values
    
    print("Generating figures...")
    generate_hinh_1_1(df_demo)
    generate_hinh_2_1(df_train)
    generate_hinh_2_2(scores, y_true)
    generate_hinh_3_1(df_train_feats)
    generate_hinh_4_1(df_train, df_demo)
    generate_hinh_5_1(scores, y_true)
    generate_hinh_5_2(preds, y_true)
    generate_hinh_5_3_fixed(df_demo, preds, common_idx)
    
    print("All figures generated successfully.")

if __name__ == "__main__":
    main()
