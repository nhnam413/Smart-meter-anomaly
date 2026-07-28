"""
============================================================
05_dashboard.py — Smart Meter Anomaly Detection Dashboard
============================================================
Mục đích:
    - Dashboard giám sát bất thường tiêu thụ điện năng thời gian thực.
    - Kiến trúc F-Z Pattern: KPIs → Main Power Chart → Detail Charts → Table.
    - 2 Chế độ: Phân tích lịch sử (History) + Giám sát thời gian thực (Real-time).
    - Tone sáng, nền trắng, typography chuẩn Inter, độ tương phản cao.

Tác giả: Sinh viên + AI Advisor
============================================================
"""

import os
import sys
import json
import time
import subprocess
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

# ============================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐƯỜNG DẪN
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DEMO_DIR = os.path.join(PROJECT_DIR, "data", "demo")
STREAM_FILE = os.path.join(DEMO_DIR, "stream_buffer.jsonl")
DEMO_CSV = os.path.join(DEMO_DIR, "demo_with_anomalies.csv")


# Bảng màu thiết kế — White/Ocean-Blue/Navy Palette
COLORS = {
    "primary": "#2563EB",        # Ocean Blue chính
    "primary_dark": "#1E3A8A",   # Deep Navy Accent
    "navy": "#0F172A",           # Deep Navy cho text/headers
    "primary_light": "#EFF6FF",  # Xanh nhạt background
    "danger": "#EF4444",         # Đỏ cảnh báo
    "danger_light": "#FEF2F2",   # Đỏ nhạt background
    "success": "#10B981",        # Xanh lá
    "success_light": "#ECFDF5",  # Xanh lá nhạt
    "warning": "#F59E0B",        # Cam / Vàng cảnh báo
    "warning_light": "#FFFBEB",  # Cam / Vàng nhạt
    "text_primary": "#0F172A",   # Navy đậm - Tương phản cao
    "text_secondary": "#334155", # Slate - Tương phản rõ nét
    "text_muted": "#64748B",     # Slate Muted
    "border": "#E2E8F0",         # Viền xám sáng
    "background": "#FFFFFF",     # Nền trắng
    "surface": "#F8FAFC",        # Surface xám nhạt
}

MAX_DISPLAY_POINTS = 100
BUFFER_SIZE = 30


# ============================================================
# 2. LOAD MODEL & DỮ LIỆU CACHE
# ============================================================

@st.cache_resource
def load_model():
    """Load Isolation Forest Model, Scaler & Feature Names (chỉ load 1 lần)."""
    model = joblib.load(os.path.join(MODELS_DIR, "isolation_forest_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return model, scaler, feature_names


@st.cache_data
def load_full_demo_data() -> pd.DataFrame:
    """Load dữ liệu demo có nhãn anomaly cho chế độ Phân tích lịch sử."""
    if not os.path.exists(DEMO_CSV):
        return pd.DataFrame()
    return pd.read_csv(DEMO_CSV, index_col="datetime", parse_dates=True)


# ============================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG & DỰ ĐOÁN
# ============================================================

def create_features_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Feature Engineering tập trung cho chế độ Batch (Batch Mode)."""
    df = df.copy()
    target_col = "Global_active_power"

    # Cyclical Encoding — Giờ & Ngày trong tuần
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Lag Features
    df["power_lag_1h"] = df[target_col].shift(1)
    df["power_lag_2h"] = df[target_col].shift(2)
    df["power_lag_24h"] = df[target_col].shift(24)

    # Rolling Statistics
    df["power_rolling_mean_6h"] = df[target_col].rolling(window=6, min_periods=1).mean()
    df["power_rolling_std_6h"] = df[target_col].rolling(window=6, min_periods=1).std()

    return df.dropna()


def create_features_realtime(buffer_df: pd.DataFrame) -> pd.Series | None:
    """Trích xuất đặc trưng cho điểm mới nhất trong luồng Real-Time."""
    if len(buffer_df) < 25:
        return None

    target_col = "Global_active_power"
    latest = buffer_df.iloc[-1].copy()
    ts = buffer_df.index[-1]

    latest["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    latest["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    latest["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
    latest["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)

    latest["power_lag_1h"] = buffer_df[target_col].iloc[-2]
    latest["power_lag_2h"] = buffer_df[target_col].iloc[-3]
    latest["power_lag_24h"] = buffer_df[target_col].iloc[-25]

    last_6 = buffer_df[target_col].iloc[-6:]
    latest["power_rolling_mean_6h"] = last_6.mean()
    latest["power_rolling_std_6h"] = last_6.std()

    return latest


def predict_batch(df: pd.DataFrame, model, scaler, feature_names) -> pd.DataFrame:
    """Dự đoán bất thường theo lô (Batch Prediction)."""
    df_feat = create_features_for_df(df)
    sensor_and_feat_cols = [c for c in feature_names if c in df_feat.columns]
    
    X = df_feat[sensor_and_feat_cols].values
    X_scaled = scaler.transform(X)

    df_feat["predicted_anomaly"] = model.predict(X_scaled) == -1
    df_feat["anomaly_score"] = model.decision_function(X_scaled)

    return df_feat


def predict_single(features, model, scaler, feature_names):
    """Dự đoán bất thường cho 1 mẫu dữ liệu."""
    X = features[feature_names].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    score = model.decision_function(X_scaled)[0]
    return prediction == -1, score


# ============================================================
# 4. ĐỌC DỮ LIỆU STREAMING
# ============================================================

def read_new_messages(last_position: int) -> tuple:
    """Đọc dữ liệu mới từ file JSONL theo vị trí con trỏ."""
    messages = []
    if not os.path.exists(STREAM_FILE):
        return messages, last_position

    with open(STREAM_FILE, "r", encoding="utf-8") as f:
        f.seek(last_position)
        for line in f:
            line = line.strip()
            if line:
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        new_position = f.tell()
    return messages, new_position


# ============================================================
# 5. TRÌNH VẼ BIỂU ĐỒ PLOTLY
# ============================================================

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, sans-serif", color="#0F172A"),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FAFC",
    margin=dict(l=50, r=20, t=50, b=50),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(size=11, color="#0F172A")
    ),
)

AXIS_STYLE = dict(
    gridcolor="#E0E2E6",
    tickfont=dict(color="#2D3436"),
    linecolor="#B0B8C1",
)


def _extract_chart_data(display_data, value_col, df_col):
    """Trích xuất dữ liệu chung cho chart từ DataFrame hoặc list[dict]."""
    if isinstance(display_data, pd.DataFrame):
        timestamps = display_data.index
        values = display_data[df_col]
        is_anomaly = display_data["predicted_anomaly"]
    else:
        timestamps = [d["timestamp"] for d in display_data]
        values = [d[value_col] for d in display_data]
        is_anomaly = [d["predicted_anomaly"] for d in display_data]
    return timestamps, values, is_anomaly


def build_main_power_chart(display_data, title_suffix="") -> go.Figure:
    """Biểu đồ chính — Công suất tiêu thụ."""
    if display_data is None or (isinstance(display_data, pd.DataFrame) and display_data.empty):
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=420, title="⏳ Chưa có dữ liệu...")
        return fig

    timestamps, powers, is_anomaly = _extract_chart_data(display_data, "power", "Global_active_power")

    # Scores cần riêng cho hover
    if isinstance(display_data, pd.DataFrame):
        scores = display_data["anomaly_score"]
    else:
        scores = [d["anomaly_score"] for d in display_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=powers, mode="lines",
        name="Công suất (kW)",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.08)",
        hovertemplate="<b>%{x}</b><br>Công suất: %{y:.3f} kW<extra></extra>"
    ))

    # Chấm đỏ bất thường
    if isinstance(display_data, pd.DataFrame):
        anom_mask = display_data["predicted_anomaly"]
        anom_ts, anom_pw = display_data.index[anom_mask], display_data.loc[anom_mask, "Global_active_power"]
        anom_sc = display_data.loc[anom_mask, "anomaly_score"].values
    else:
        anom_ts = [t for t, a in zip(timestamps, is_anomaly) if a]
        anom_pw = [p for p, a in zip(powers, is_anomaly) if a]
        anom_sc = [s for s, a in zip(scores, is_anomaly) if a]

    if len(anom_ts) > 0:
        fig.add_trace(go.Scatter(
            x=anom_ts, y=anom_pw, mode="markers",
            name="Bất thường (Anomaly)",
            marker=dict(color=COLORS["danger"], size=10, symbol="circle", line=dict(width=2, color="white")),
            customdata=anom_sc,
            hovertemplate="<b>🔴 BẤT THƯỜNG</b><br>Thời điểm: %{x}<br>Công suất: %{y:.3f} kW<br>Score: %{customdata:.4f}<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT, height=420,
        title=dict(text=f"Biểu đồ công suất tiêu thụ điện năng {title_suffix}",
                   font=dict(size=15, color=COLORS["text_primary"]), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Thời gian", font=dict(size=12, color="#2D3436")),
                   showgrid=True, **AXIS_STYLE),
        yaxis=dict(title=dict(text="Global Active Power (kW)", font=dict(size=12, color="#2D3436")),
                   showgrid=True, zeroline=True, zerolinecolor="#B0B8C1", **AXIS_STYLE),
    )
    return fig


def build_voltage_chart(display_data) -> go.Figure:
    """Biểu đồ phụ — Giám sát điện áp."""
    if display_data is None or (isinstance(display_data, pd.DataFrame) and display_data.empty):
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=320)
        return fig

    timestamps, voltages, is_anomaly = _extract_chart_data(display_data, "voltage", "Voltage")

    fig = go.Figure()
    fig.add_hrect(
        y0=220, y1=250, fillcolor="rgba(16, 185, 129, 0.06)", line_width=0,
        annotation_text="Vùng an toàn (220–250V)",
        annotation_position="top left",
        annotation_font=dict(color=COLORS["success"], size=10),
    )
    fig.add_trace(go.Scatter(
        x=timestamps, y=voltages, mode="lines",
        name="Điện áp (V)", line=dict(color=COLORS["warning"], width=2),
    ))

    # Chấm đỏ bất thường
    if isinstance(display_data, pd.DataFrame):
        anom_mask = display_data["predicted_anomaly"]
        anom_ts, anom_v = display_data.index[anom_mask], display_data.loc[anom_mask, "Voltage"]
    else:
        anom_ts = [t for t, a in zip(timestamps, is_anomaly) if a]
        anom_v = [v for v, a in zip(voltages, is_anomaly) if a]

    if len(anom_ts) > 0:
        fig.add_trace(go.Scatter(
            x=anom_ts, y=anom_v, mode="markers", name="Bất thường",
            marker=dict(color=COLORS["danger"], size=8, symbol="circle", line=dict(width=1.5, color="white")),
        ))

    fig.update_layout(
        **CHART_LAYOUT, height=320,
        title=dict(text="Điện áp (Voltage)", font=dict(size=14, color=COLORS["text_primary"]), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Thời gian", font=dict(size=12, color="#2D3436")), **AXIS_STYLE),
        yaxis=dict(title=dict(text="Voltage (V)", font=dict(size=12, color="#2D3436")), **AXIS_STYLE),
    )
    return fig


def build_anomaly_pie(df: pd.DataFrame) -> go.Figure:
    """Biểu đồ tròn — Phân bổ chủng loại bất thường."""
    if df.empty or "anomaly_type" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=320)
        return fig

    counts = df["anomaly_type"].value_counts()
    color_map = {
        "normal": "#C8D6E5",
        "power_surge": COLORS["danger"],
        "voltage_drop": COLORS["warning"],
        "night_spike": "#7B1FA2",
    }
    colors = [color_map.get(name, "#999") for name in counts.index]
    label_map = {
        "normal": "Bình thường",
        "power_surge": "Đột biến CS",
        "voltage_drop": "Sụt áp",
        "night_spike": "Đột biến đêm",
    }
    labels = [label_map.get(name, name) for name in counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent",
        textposition="inside",
        textfont=dict(size=11, color="#FFFFFF"),
        hovertemplate="%{label}: %{value} mẫu (%{percent})<extra></extra>",
    )])

    pie_layout = {k: v for k, v in CHART_LAYOUT.items() if k != "legend"}
    fig.update_layout(
        **pie_layout, height=320,
        title=dict(text="Phân bổ loại bất thường", font=dict(size=13, color=COLORS["text_primary"]), x=0, xanchor="left"),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
            font=dict(size=10, color=COLORS["text_secondary"]),
        ),
    )
    return fig


def build_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Biểu đồ cột — Trung bình công suất tiêu thụ theo giờ trong ngày."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=320)
        return fig

    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    hourly_avg = df_copy.groupby("hour")["Global_active_power"].mean()

    fig = go.Figure(data=[go.Bar(
        x=hourly_avg.index,
        y=hourly_avg.values,
        marker=dict(
            color=hourly_avg.values,
            colorscale=[[0, COLORS["primary_light"]], [0.5, COLORS["primary"]], [1, "#0D47A1"]],
            line=dict(width=0),
            cornerradius=4,
        ),
        hovertemplate="Giờ %{x}h<br>TB: %{y:.3f} kW<extra></extra>",
    )])

    fig.update_layout(
        **CHART_LAYOUT, height=320,
        title=dict(text="Trung bình công suất theo giờ", font=dict(size=14, color=COLORS["text_primary"]), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Giờ", font=dict(size=12, color="#2D3436")), dtick=2, **AXIS_STYLE),
        yaxis=dict(title=dict(text="Power (kW)", font=dict(size=12, color="#2D3436")), **AXIS_STYLE),
        bargap=0.15,
    )
    return fig


CSS_FILE = os.path.join(PROJECT_DIR, "src", "style.css")


def load_css():
    """Nhúng style.css vào Streamlit."""
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE, "r", encoding="utf-8") as f:
            st.markdown(f"<style>\n{f.read()}\n</style>", unsafe_allow_html=True)


def spacer(px: int = 12):
    """Render khoảng cách dọc."""
    st.markdown(f"<div style='height:{px}px;'></div>", unsafe_allow_html=True)


# ============================================================
# 6. THÀNH PHẦN UI
# ============================================================

def render_metric_card(value: str, label: str,
                       color: str = "#0F172A", delta: str = None,
                       delta_color: str = "#10B981",
                       variant: str = None):
    """Render 1 thẻ metric — hỗ trợ màu sắc tùy chỉnh qua CSS Variant."""
    delta_html = ""
    if delta:
        delta_html = f'<div class="metric-delta" style="color:{delta_color};">{delta}</div>'

    variant_class = f" {variant}" if variant else ""
    val_color = f' style="color:{color} !important;"' if color and not variant else ''

    st.markdown(f"""
    <div class="metric-card{variant_class}">
        <div class="metric-value"{val_color}>{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_alert_bar(anomaly_type: str, timestamp: str,
                     power: float, voltage: float, score: float):
    """Render thanh cảnh báo bất thường nhấp nháy đỏ."""
    type_labels = {
        "power_surge": "Đột biến công suất",
        "voltage_drop": "Sụt áp điện",
        "night_spike": "Bất thường ban đêm",
        "normal": "Bất thường phát hiện",
    }
    label = type_labels.get(anomaly_type, "Bất thường")

    st.markdown(f"""
    <div class="anomaly-alert-bar">
        <div class="alert-title">CANH BAO — {label.upper()}</div>
        <div class="alert-detail">
            Thời điểm: <b>{timestamp}</b> &nbsp;·&nbsp;
            Công suất: <b>{power:.3f} kW</b> &nbsp;·&nbsp;
            Điện áp: <b>{voltage:.1f}V</b> &nbsp;·&nbsp;
            Score: <b>{score:.4f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 7. TAB 1 — PHÂN TÍCH LỊCH SỬ
# ============================================================

def render_history_tab(model, scaler, feature_names):
    """
    Tab Phân tích lịch sử — Lọc khoảng ngày và thực hiện dự đoán theo lô (Batch).
    """
    df_full = load_full_demo_data()

    if df_full.empty:
        st.warning("Chưa có dữ liệu demo. Hãy chạy `03_inject_anomalies.py` trước!")
        return

    # F-Line 1: Date Filter
    st.markdown("#### Chọn khoảng thời gian phân tích")
    col_date1, col_date2 = st.columns([1, 1])

    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()

    with col_date1:
        start_date = st.date_input(
            "Từ ngày",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="history_start"
        )
    with col_date2:
        end_date = st.date_input(
            "Đến ngày",
            value=min(min_date + timedelta(days=14), max_date),
            min_value=min_date,
            max_value=max_date,
            key="history_end"
        )

    # Lọc dữ liệu theo ngày
    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_filtered = df_full[mask].copy()

    if df_filtered.empty:
        st.info("Không có dữ liệu trong khoảng ngày đã chọn.")
        return

    # Chạy prediction trên dữ liệu đã lọc
    sensor_cols = ["Global_active_power", "Global_reactive_power",
                   "Voltage", "Global_intensity",
                   "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df_predict = predict_batch(df_filtered[sensor_cols], model, scaler, feature_names)

    if "is_anomaly" in df_filtered.columns:
        common_idx = df_predict.index.intersection(df_filtered.index)
        df_predict.loc[common_idx, "anomaly_type"] = df_filtered.loc[common_idx, "anomaly_type"]
        df_predict.loc[common_idx, "is_anomaly_gt"] = df_filtered.loc[common_idx, "is_anomaly"]

    # F-Line 2: KPI Metrics Cards
    total = len(df_predict)
    n_pred_anomaly = df_predict["predicted_anomaly"].sum()
    n_gt_anomaly = int(df_predict.get("is_anomaly_gt", pd.Series(dtype=int)).sum())
    avg_power = df_predict["Global_active_power"].mean()
    anomaly_rate = (n_pred_anomaly / total * 100) if total > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card(f"{total:,}", "Tổng mẫu", color=COLORS["primary"])
    with c2:
        render_metric_card(f"{n_pred_anomaly:,}", "Model phát hiện", variant="metric-card-red")
    with c3:
        render_metric_card(f"{n_gt_anomaly}", "Nhãn thật (GT)", color="#7B1FA2")
    with c4:
        render_metric_card(f"{avg_power:.2f} kW", "Công suất TB", color=COLORS["primary"])
    with c5:
        render_metric_card(f"{anomaly_rate:.1f}%", "Tỷ lệ bất thường", variant="metric-card-green")

    spacer(16)

    # Z-Line: Main Chart
    date_range_str = f"({start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')})"
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(build_main_power_chart(df_predict, title_suffix=date_range_str), width="stretch", key="hist_power")
    st.markdown('</div>', unsafe_allow_html=True)

    # Z-Line: Secondary Charts
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(build_voltage_chart(df_predict), width="stretch", key="hist_voltage")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(build_hourly_heatmap(df_predict), width="stretch", key="hist_hourly")
        st.markdown('</div>', unsafe_allow_html=True)

    # Z-Line Bottom: Pie + Table
    col_pie, col_table = st.columns([1, 2])
    with col_pie:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if "anomaly_type" in df_predict.columns:
            st.plotly_chart(build_anomaly_pie(df_predict), width="stretch", key="hist_pie")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_table:
        st.markdown("#### Chi tiết các điểm bất thường")
        anomaly_rows = df_predict[df_predict["predicted_anomaly"]].copy()

        if not anomaly_rows.empty:
            display_cols = {
                "Global_active_power": "Công suất (kW)",
                "Voltage": "Điện áp (V)",
                "Global_intensity": "Dòng điện (A)",
                "anomaly_score": "Score",
            }
            if "anomaly_type" in anomaly_rows.columns:
                display_cols["anomaly_type"] = "Loại (GT)"

            table_df = anomaly_rows[list(display_cols.keys())].copy()
            table_df = table_df.rename(columns=display_cols)
            table_df.index = table_df.index.strftime("%d/%m/%Y %H:%M")
            table_df.index.name = "Thời gian"

            for col in ["Công suất (kW)", "Điện áp (V)", "Dòng điện (A)"]:
                if col in table_df.columns:
                    table_df[col] = table_df[col].round(3)
            if "Score" in table_df.columns:
                table_df["Score"] = table_df["Score"].round(4)

            st.dataframe(table_df.head(50), width="stretch", height=350)
        else:
            st.success("Không phát hiện bất thường trong khoảng thời gian này.")


# ============================================================
# 8. TAB 2 — GIÁM SÁT THỜI GIAN THỰC
# ============================================================

def render_realtime_tab(model, scaler, feature_names):
    """
    Tab Giám sát thời gian thực — đọc dữ liệu từ Producer và dự đoán trực tiếp.
    """
    refresh_rate = st.session_state.get("refresh_rate", 2)

    # ── Producer Control (PID Synchronized) ───────────────────
    if "producer_pid" not in st.session_state:
        st.session_state.producer_pid = None

    def _is_producer_pid_alive(pid: int | None) -> bool:
        if not pid or pid <= 0:
            return False
        if sys.platform == "win32":
            try:
                out = subprocess.check_output(
                    f'tasklist /fi "PID eq {pid}" /fo csv /nh',
                    shell=True, text=True, stderr=subprocess.DEVNULL
                )
                return str(pid) in out and "No tasks" not in out
            except Exception:
                return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def _stop_producer_pid(pid: int | None):
        if sys.platform == "win32":
            if pid and pid > 0:
                try:
                    subprocess.run(f'taskkill /F /T /PID {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            try:
                subprocess.run('wmic process where "commandline like \'%04_producer.py%\'" call terminate', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        else:
            if pid and pid > 0:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass

    producer_is_running = _is_producer_pid_alive(st.session_state.producer_pid)
    if not producer_is_running and st.session_state.producer_pid is not None:
        st.session_state.producer_pid = None

    # Hàng 3 nút điều khiển tĩnh + Đèn trạng thái PID
    producer_col1, producer_col2, producer_col3, producer_col4 = st.columns([1.1, 1, 1.1, 2.2])

    # Nút 1: Khởi động mới (Trái)
    with producer_col1:
        if st.button(
            "Khởi động",
            key="btn_start_producer",
            use_container_width=True,
            type="primary" if not producer_is_running else "secondary",
            disabled=producer_is_running
        ):
            if os.path.exists(STREAM_FILE):
                open(STREAM_FILE, "w", encoding="utf-8").close()
            st.session_state.rt_display = []
            st.session_state.rt_buffer = []
            st.session_state.rt_file_pos = 0
            st.session_state.rt_total = 0
            st.session_state.rt_anomalies = 0
            st.session_state.rt_last_anomaly = None

            producer_script = os.path.join(PROJECT_DIR, "src", "04_producer.py")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            proc = subprocess.Popen(
                [sys.executable, producer_script, "--speed", "0.5"],
                cwd=PROJECT_DIR,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            st.session_state.producer_pid = proc.pid
            st.rerun()

    # Nút 2: Dừng (Giữa)
    with producer_col2:
        if st.button(
            "Dừng",
            key="btn_stop_producer",
            use_container_width=True,
            type="primary" if producer_is_running else "secondary",
            disabled=not producer_is_running
        ):
            _stop_producer_pid(st.session_state.producer_pid)
            st.session_state.producer_pid = None
            st.rerun()

    # Nút 3: Tiếp tục (Phải)
    with producer_col3:
        if st.button(
            "Tiếp tục",
            key="btn_resume_producer",
            use_container_width=True,
            type="primary" if not producer_is_running else "secondary",
            disabled=producer_is_running
        ):
            skip_count = st.session_state.get("rt_total", 0)
            producer_script = os.path.join(PROJECT_DIR, "src", "04_producer.py")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            proc = subprocess.Popen(
                [sys.executable, producer_script, "--speed", "0.5", "--append", "--skip", str(skip_count)],
                cwd=PROJECT_DIR,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            st.session_state.producer_pid = proc.pid
            st.rerun()

    # Khối đèn báo trạng thái PID
    with producer_col4:
        if producer_is_running:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; padding:10px 0;">
                <span class="status-dot online"></span>
                <span style="color:#10B981; font-weight:700; font-size:0.88em;">
                    Producer đang chạy (PID: {st.session_state.producer_pid})
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            has_data = st.session_state.get("rt_total", 0) > 0
            sub_text = f"Đang dừng (Đã nạp {st.session_state.get('rt_total', 0):,} mẫu)" if has_data else "Chưa chạy — bấm Khởi động mới"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; padding:10px 0;">
                <span class="status-dot offline"></span>
                <span style="color:#64748B; font-size:0.85em; font-weight:500;">
                    {sub_text}
                </span>
            </div>
            """, unsafe_allow_html=True)

    spacer(8)

    # Khởi tạo Real-Time State
    if "rt_display" not in st.session_state:
        st.session_state.rt_display = []
    if "rt_buffer" not in st.session_state:
        st.session_state.rt_buffer = []
    if "rt_file_pos" not in st.session_state:
        st.session_state.rt_file_pos = 0
    if "rt_total" not in st.session_state:
        st.session_state.rt_total = 0
    if "rt_anomalies" not in st.session_state:
        st.session_state.rt_anomalies = 0
    if "rt_last_anomaly" not in st.session_state:
        st.session_state.rt_last_anomaly = None

    # Đọc luồng dữ liệu mới (Chỉ nạp khi Producer đang hoạt động)
    new_messages = []
    if producer_is_running:
        new_messages, new_pos = read_new_messages(st.session_state.rt_file_pos)
        st.session_state.rt_file_pos = new_pos

    # Xử lý & Thực hiện dự đoán từng mẫu
    for msg in new_messages:
        st.session_state.rt_total += 1
        st.session_state.rt_buffer.append(msg)
        if len(st.session_state.rt_buffer) > BUFFER_SIZE:
            st.session_state.rt_buffer.pop(0)

        buf_df = pd.DataFrame(st.session_state.rt_buffer)
        buf_df["datetime"] = pd.to_datetime(buf_df["timestamp"])
        buf_df = buf_df.set_index("datetime")

        sensor_cols = ["Global_active_power", "Global_reactive_power",
                       "Voltage", "Global_intensity",
                       "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        buf_sensor = buf_df[sensor_cols].astype(float)

        features = create_features_realtime(buf_sensor)
        predicted_anomaly = False
        anomaly_score = 0.0

        if features is not None:
            predicted_anomaly, anomaly_score = predict_single(
                features, model, scaler, feature_names
            )

        if predicted_anomaly:
            st.session_state.rt_anomalies += 1

        point = {
            "timestamp": msg["timestamp"],
            "power": float(msg["Global_active_power"]),
            "voltage": float(msg["Voltage"]),
            "intensity": float(msg["Global_intensity"]),
            "predicted_anomaly": predicted_anomaly,
            "anomaly_score": anomaly_score,
            "ground_truth": msg.get("is_anomaly", 0) == 1,
            "anomaly_type": msg.get("anomaly_type", "normal"),
        }
        st.session_state.rt_display.append(point)

        max_pts = st.session_state.get("max_display", MAX_DISPLAY_POINTS)
        if len(st.session_state.rt_display) > max_pts:
            st.session_state.rt_display = st.session_state.rt_display[-max_pts:]

        if predicted_anomaly:
            st.session_state.rt_last_anomaly = point

    # Đèn báo kết nối dữ liệu
    is_receiving = len(new_messages) > 0
    status_dot = "online" if is_receiving else "offline"
    status_text = "Đang nhận dữ liệu" if is_receiving else "Chờ Producer..."

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <span class="status-dot {status_dot}"></span>
        <span style="color:{COLORS['text_secondary']}; font-size:0.9em;">
            {status_text} &nbsp;|&nbsp; Nguồn: FILE
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Hiển thị 4 Thẻ KPI Metrics Realtime
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            f"{st.session_state.rt_total:,}", "Tổng nhận",
            color=COLORS["primary"],
            delta=f"+{len(new_messages)}" if new_messages else None
        )
    with c2:
        render_metric_card(
            f"{st.session_state.rt_anomalies}", "Anomaly",
            variant="metric-card-red"
        )
    with c3:
        rate = (st.session_state.rt_anomalies / st.session_state.rt_total * 100
                if st.session_state.rt_total > 0 else 0)
        render_metric_card(
            f"{rate:.1f}%", "Tỷ lệ anomaly",
            variant="metric-card-green"
        )
    with c4:
        latest_power = (st.session_state.rt_display[-1]["power"]
                        if st.session_state.rt_display else 0)
        render_metric_card(
            f"{latest_power:.3f}", "Công suất (kW)",
            variant="metric-card-blue"
        )

    spacer(12)

    # Alert Bar khi phát hiện bất thường mới nhất
    if st.session_state.rt_last_anomaly:
        a = st.session_state.rt_last_anomaly
        render_alert_bar(
            a["anomaly_type"], a["timestamp"],
            a["power"], a["voltage"], a["anomaly_score"]
        )

    # Biểu đồ chính Realtime
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(
        build_main_power_chart(st.session_state.rt_display, title_suffix="— Thời gian thực"),
        width="stretch", key="rt_power"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Biểu đồ phụ & Log Anomaly
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(build_voltage_chart(st.session_state.rt_display), width="stretch", key="rt_voltage")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Log Anomaly gần nhất")
        anomaly_logs = [d for d in st.session_state.rt_display if d["predicted_anomaly"]]
        if anomaly_logs:
            log_df = pd.DataFrame(anomaly_logs[-15:][::-1])
            log_df = log_df[["timestamp", "power", "voltage", "anomaly_score", "anomaly_type"]]
            log_df.columns = ["Thời gian", "Công suất (kW)", "Điện áp (V)", "Score", "Loại"]
            st.dataframe(log_df, width="stretch", hide_index=True, height=320)
        else:
            st.info("Chưa phát hiện anomaly. Đang giám sát...")

    # Auto-refresh luồng (Chỉ tự động refresh khi Producer đang chạy)
    if producer_is_running:
        time.sleep(refresh_rate)
        st.rerun()


# ============================================================
# 9. ỨNG DỤNG CHÍNH
# ============================================================

def main():
    st.set_page_config(
        page_title="Smart Meter Anomaly Detection",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject CSS Custom System từ file style.css
    load_css()

    # Top Bar Header
    st.markdown("""
    <div class="top-bar">
        <div class="top-bar-title">
            <h1>Smart Meter Anomaly Detection</h1>
            <span class="top-bar-badge">REAL-TIME</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Cấu Hình Hệ Thống
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title-block">
            <h3 class="sidebar-title-text">Cấu hình hệ thống</h3>
            <span class="sidebar-subtitle-text">Smart Meter Anomaly Dashboard</span>
        </div>
        """, unsafe_allow_html=True)


        refresh_rate = st.slider(
            "Tốc độ cập nhật (giây):",
            min_value=1, max_value=10, value=2,
        )
        st.session_state.refresh_rate = refresh_rate

        max_display = st.slider(
            "Số điểm hiển thị:",
            min_value=30, max_value=200, value=100,
        )
        st.session_state.max_display = max_display

        st.divider()
        st.markdown("""
        <div style="text-align: center; margin-bottom: 8px;">
            <h4 style="margin: 0; font-size: 0.88em; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.4px;">Hướng dẫn sử dụng</h4>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **Tab Phân tích lịch sử:**
        1. Chọn khoảng ngày bắt đầu/kết thúc.
        2. Dashboard tự động chạy model và hiển thị.

        **Tab Giám sát thời gian thực:**
        1. Bấm nút **Khởi động** trên thanh điều khiển.
        2. Dashboard tự động nạp dữ liệu & phân tích.
        """)

        st.divider()
        st.markdown("""
        <div style="text-align:center; color:#475569; font-size:0.78em; padding:8px; font-weight: 500;">
            <b>Đồ án Đại học</b><br>
            Isolation Forest · Streamlit<br>
            © 2026
        </div>
        """, unsafe_allow_html=True)

    # Load Model AI
    try:
        model, scaler, feature_names = load_model()
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        st.info("Hãy chạy python src/02_train_model.py trước.")
        st.stop()

    # Điều hướng giữa 2 Chế độ Main Mode
    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "history"

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("📊 Phân tích lịch sử", key="btn_history",
                     use_container_width=True,
                     type="primary" if st.session_state.active_mode == "history" else "secondary"):
            st.session_state.active_mode = "history"
            st.rerun()
    with btn_col2:
        if st.button("📡 Giám sát thời gian thực", key="btn_realtime",
                     use_container_width=True,
                     type="primary" if st.session_state.active_mode == "realtime" else "secondary"):
            st.session_state.active_mode = "realtime"
            st.rerun()

    spacer(12)

    if st.session_state.active_mode == "history":
        render_history_tab(model, scaler, feature_names)
    else:
        render_realtime_tab(model, scaler, feature_names)


if __name__ == "__main__":
    if not st.runtime.exists():
        print("\n" + "=" * 65)
        print("[!] CANH BAO: Streamlit Dashboard khong the chay bang lenh 'python'!")
        print("    Hay khoi chay giao dien bang lenh Streamlit CLI:")
        print("    -> cd smart-meter-anomaly; streamlit run src/05_dashboard.py")
        print("    hoac:")
        print("    -> streamlit run smart-meter-anomaly/src/05_dashboard.py")
        print("=" * 65 + "\n")
        sys.exit(0)
    else:
        main()
