"""
05_dashboard.py — Smart Meter Anomaly Detection Dashboard
Giao diện giám sát tiêu thụ điện năng và phát hiện bất thường thời gian thực.
"""

import os
import sys
import json
import time
import signal
import subprocess
from datetime import datetime
from typing import Optional, Union

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    MODELS_DIR, STREAM_FILE, DEMO_CSV, CSS_FILE,
    COLORS, MAX_DISPLAY_POINTS, BUFFER_SIZE, SENSOR_COLUMNS,
    PROJECT_DIR, REPORTS_DIR,
)
from features import create_features, create_features_realtime
from classify import (
    classify_anomaly_type, classify_batch, explain_batch,
    compute_feature_stats, TYPE_LABELS,
)


# --- Cache & Data Loader ---

@st.cache_resource
def load_model():
    """Tải mô hình, scaler và danh sách đặc trưng đã lưu."""
    model = joblib.load(os.path.join(MODELS_DIR, "isolation_forest_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return model, scaler, feature_names


@st.cache_data
def load_full_demo_data() -> pd.DataFrame:
    """Tải toàn bộ tập dữ liệu demo (có nhãn bất thường)."""
    if not os.path.exists(DEMO_CSV):
        return pd.DataFrame()
    return pd.read_csv(DEMO_CSV, index_col="datetime", parse_dates=True)


# --- Inference Helpers ---

def predict_batch(df: pd.DataFrame, model, scaler, feature_names: list[str]) -> pd.DataFrame:
    """Dự đoán nhãn bất thường cho cả batch dữ liệu."""
    df_feat = create_features(df)
    sensor_and_feat_cols = [c for c in feature_names if c in df_feat.columns]

    X = df_feat[sensor_and_feat_cols].values
    X_scaled = scaler.transform(X)

    df_feat["predicted_anomaly"] = model.predict(X_scaled) == -1
    df_feat["anomaly_score"] = model.decision_function(X_scaled)
    return df_feat


def predict_single(features: pd.Series, model, scaler, feature_names: list[str]) -> tuple[bool, float]:
    """Dự đoán cho một điểm dữ liệu thời gian thực."""
    X = features[feature_names].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    score = float(model.decision_function(X_scaled)[0])
    return prediction == -1, score


def read_new_messages(last_position: int) -> tuple[list[dict], int]:
    """Đọc các bản tin mới nhất từ stream buffer JSONL."""
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


# --- Plotly Chart Builders ---

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, sans-serif", color="#0F172A"),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=45, r=15, t=35, b=35),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(size=11, color="#0F172A")
    ),
)

AXIS_STYLE = dict(
    gridcolor="#E2E8F0",
    tickfont=dict(color="#0F172A"),
    linecolor="#CBD5E1",
)


def _extract_chart_data(display_data: Union[pd.DataFrame, list[dict]], value_col: str, df_col: str):
    """Trích xuất danh sách timestamp, giá trị và cờ bất thường cho biểu đồ."""
    if isinstance(display_data, pd.DataFrame):
        timestamps = display_data.index
        values = display_data[df_col]
        is_anomaly = display_data["predicted_anomaly"]
    else:
        timestamps = [d["timestamp"] for d in display_data]
        values = [d[value_col] for d in display_data]
        is_anomaly = [d["predicted_anomaly"] for d in display_data]
    return timestamps, values, is_anomaly


def _build_sensor_chart(
    display_data: Union[pd.DataFrame, list[dict]],
    val_key: str,
    df_col: str,
    title: str,
    y_title: str,
    height: int = 350,
    line_name: str = "",
    add_hrect: bool = False
) -> go.Figure:
    """Helper chung xây dựng biểu đồ chuỗi thời gian kèm các điểm bất thường."""
    if display_data is None or (isinstance(display_data, pd.DataFrame) and display_data.empty):
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=height, title="Chưa có dữ liệu...")
        return fig

    timestamps, values, is_anomaly = _extract_chart_data(display_data, val_key, df_col)
    fig = go.Figure()

    if add_hrect:
        fig.add_hrect(
            y0=220, y1=250, fillcolor="rgba(16, 185, 129, 0.06)", line_width=0,
            annotation_text="Vùng an toàn (220–250V)",
            annotation_position="top left",
            annotation_font=dict(color=COLORS["success"], size=10),
        )

    fill_opt = "tozeroy" if val_key == "power" else None
    fill_color = "rgba(107, 76, 230, 0.06)" if val_key == "power" else None
    hover_fmt = f"<b>%{{x}}</b><br>{line_name}: %{{y:.3f}}<extra></extra>"

    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode="lines",
        name=line_name,
        line=dict(color=COLORS["primary"], width=2),
        fill=fill_opt, fillcolor=fill_color,
        hovertemplate=hover_fmt
    ))

    if isinstance(display_data, pd.DataFrame):
        anom_mask = display_data["predicted_anomaly"]
        anom_ts = display_data.index[anom_mask]
        anom_vals = display_data.loc[anom_mask, df_col]
        anom_sc = display_data.loc[anom_mask, "anomaly_score"].values if "anomaly_score" in display_data.columns else None
    else:
        anom_ts = [t for t, a in zip(timestamps, is_anomaly) if a]
        anom_vals = [v for v, a in zip(values, is_anomaly) if a]
        anom_sc = [d["anomaly_score"] for d in display_data if d["predicted_anomaly"]] if display_data and "anomaly_score" in display_data[0] else None

    if len(anom_ts) > 0:
        marker_size = 9 if val_key == "power" else 7
        fig.add_trace(go.Scatter(
            x=anom_ts, y=anom_vals, mode="markers",
            name="Bất thường (Anomaly)" if val_key == "power" else "Bất thường",
            marker=dict(color=COLORS["accent"], size=marker_size, symbol="circle", line=dict(width=1.2, color="white")),
            customdata=anom_sc,
            hovertemplate="<b>BẤT THƯỜNG</b><br>Thời điểm: %{x}<br>Công suất: %{y:.3f} kW<br>Score: %{customdata:.4f}<extra></extra>" if val_key == "power" else "<b>BẤT THƯỜNG</b><br>Thời điểm: %{x}<br>Điện áp: %{y:.1f}V<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT, height=height,
        title=dict(text=title, font=dict(size=14 if val_key == "power" else 13, color="#0F172A"), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Thời gian", font=dict(size=11, color="#0F172A")), showgrid=True, **AXIS_STYLE),
        yaxis=dict(title=dict(text=y_title, font=dict(size=11, color="#0F172A")), showgrid=True, zeroline=True, zerolinecolor="#CBD5E1", **AXIS_STYLE),
    )
    return fig


def build_main_power_chart(display_data: Union[pd.DataFrame, list[dict]], title_suffix: str = "") -> go.Figure:
    """Xây dựng biểu đồ đường thể hiện công suất tiêu thụ điện và điểm bất thường."""
    return _build_sensor_chart(
        display_data, "power", "Global_active_power",
        title=f"Biểu đồ công suất tiêu thụ điện năng {title_suffix}",
        y_title="Global Active Power (kW)", height=400, line_name="Công suất (kW)"
    )


def build_voltage_chart(display_data: Union[pd.DataFrame, list[dict]]) -> go.Figure:
    """Xây dựng biểu đồ giám sát điện áp và vùng an toàn (220-250V)."""
    return _build_sensor_chart(
        display_data, "voltage", "Voltage",
        title="Điện áp (Voltage)", y_title="Voltage (V)",
        height=300, line_name="Điện áp (V)", add_hrect=True
    )



def build_anomaly_timeline_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap 2D: trục X = ngày, trục Y = giờ, màu = số anomaly detected."""
    if df.empty or "predicted_anomaly" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=340)
        return fig

    df_copy = df.copy()
    df_copy["date"] = df_copy.index.date
    df_copy["hour"] = df_copy.index.hour

    pivot = df_copy.pivot_table(
        index="hour", columns="date",
        values="predicted_anomaly", aggfunc="sum"
    ).fillna(0).sort_index()

    date_labels = [d.strftime("%d/%m") for d in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=date_labels,
        y=pivot.index,
        colorscale=[
            [0, "#F8F6FF"],
            [0.25, "#E0D7FA"],
            [0.5, COLORS["primary"]],
            [0.75, COLORS["accent"]],
            [1, COLORS["danger"]],
        ],
        colorbar=dict(title="Anomaly", tickfont=dict(size=10, color="#0F172A")),
        hovertemplate="Ngày %{x}<br>Giờ %{y}h<br>Số bất thường: %{z}<extra></extra>",
        xgap=1, ygap=1,
    ))

    fig.update_layout(
        **CHART_LAYOUT, height=340,
        title=dict(text="Phân bố bất thường theo Giờ × Ngày", font=dict(size=13, color="#0F172A"), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Ngày", font=dict(size=11, color="#0F172A")), tickangle=-45, **AXIS_STYLE),
        yaxis=dict(title=dict(text="Giờ", font=dict(size=11, color="#0F172A")), dtick=2, autorange="reversed", **AXIS_STYLE),
    )
    return fig


def build_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Biểu đồ cột thể hiện mức tiêu thụ trung bình theo từng giờ trong ngày."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=300)
        return fig

    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    hourly_avg = df_copy.groupby("hour")["Global_active_power"].mean()

    fig = go.Figure(data=[go.Bar(
        x=hourly_avg.index,
        y=hourly_avg.values,
        marker=dict(
            color=hourly_avg.values,
            colorscale=[[0, COLORS["primary_light"]], [0.5, COLORS["primary"]], [1, COLORS["primary_dark"]]],
            line=dict(width=0),
            cornerradius=4,
        ),
        hovertemplate="Giờ %{x}h<br>TB: %{y:.3f} kW<extra></extra>",
    )])

    fig.update_layout(
        **CHART_LAYOUT, height=300,
        title=dict(text="Trung bình công suất theo giờ", font=dict(size=13, color="#0F172A"), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Giờ", font=dict(size=11, color="#0F172A")), dtick=2, **AXIS_STYLE),
        yaxis=dict(title=dict(text="Power (kW)", font=dict(size=11, color="#0F172A")), **AXIS_STYLE),
        bargap=0.15,
    )
    return fig


# --- UI Utilities & Components ---

def load_css() -> None:
    """Nạp file CSS tùy chỉnh."""
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE, "r", encoding="utf-8") as f:
            st.markdown(f"<style>\n{f.read()}\n</style>", unsafe_allow_html=True)


def spacer(px: int = 8) -> None:
    """Tạo khoảng đệm chiều dọc."""
    st.markdown(f"<div style='height:{px}px;'></div>", unsafe_allow_html=True)


def render_metric_card(
    value: str,
    label: str,
    variant: str = "metric-card-primary",
    pill_text: Optional[str] = None,
    pill_type: str = "purple",
    progress_pct: Optional[float] = None,
    delta: Optional[str] = None,
    delta_color: str = "#10B981"
) -> None:
    """Hiển thị một thẻ KPI gọn gàng, độ tương phản cao."""
    pill_html = f'<span class="metric-pill metric-pill-{pill_type}">{pill_text}</span>' if pill_text else ''
    delta_html = f'<div class="metric-delta" style="color:{delta_color}; font-size:0.75em; font-weight:600;">{delta}</div>' if delta else ''

    progress_html = ""
    if progress_pct is not None:
        pct = max(0.0, min(100.0, float(progress_pct)))
        bar_color = "#6B4CE6" if pill_type == "purple" else "#EF4444" if pill_type == "red" else "#EF7D32" if pill_type == "orange" else "#10B981"
        progress_html = f'<div class="metric-progress-container"><div class="metric-progress-bar" style="width: {pct:.1f}%; background-color: {bar_color};"></div></div>'

    card_html = f'<div class="metric-card {variant}"><div class="metric-card-header"><span class="metric-label">{label}</span>{pill_html}</div><div class="metric-value">{value}</div>{delta_html}{progress_html}</div>'

    st.markdown(card_html, unsafe_allow_html=True)


def render_alert_bar(anomaly_type: str, timestamp: str, power: float, voltage: float, score: float) -> None:
    """Hiển thị thanh cảnh báo tức thời khi phát hiện bất thường."""
    label = TYPE_LABELS.get(anomaly_type, "Bất thường")


    st.markdown(
        f'<div class="anomaly-alert-bar">'
        f'<div class="alert-title">CẢNH BÁO — {label.upper()}</div>'
        f'<div class="alert-detail">Thời điểm: <b>{timestamp}</b> &nbsp;·&nbsp; Công suất: <b>{power:.3f} kW</b> &nbsp;·&nbsp; Điện áp: <b>{voltage:.1f}V</b> &nbsp;·&nbsp; Score: <b>{score:.4f}</b></div>'
        f'</div>',
        unsafe_allow_html=True
    )


# --- Process Management ---

def _is_producer_pid_alive(pid: Optional[int]) -> bool:
    """Kiểm tra tiến trình producer có đang chạy hay không."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(f'tasklist /fi "PID eq {pid}" /fo csv /nh', shell=True, text=True, stderr=subprocess.DEVNULL)
            return str(pid) in out and "No tasks" not in out
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _stop_producer_pid(pid: Optional[int]) -> None:
    """Dừng dứt điểm tiến trình producer."""
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


# --- Top Control Bar ---

def render_top_control_bar(df_full: Optional[pd.DataFrame] = None) -> None:
    """Hiển thị thanh điều hướng chế độ và bộ lọc ngày/tiến trình."""
    mode_col1, mode_col2 = st.columns(2)

    with mode_col1:
        if st.button("Phân tích lịch sử", key="btn_history", use_container_width=True, type="primary" if st.session_state.active_mode == "history" else "secondary"):
            st.session_state.active_mode = "history"
            st.rerun()

    with mode_col2:
        if st.button("Giám sát thời gian thực", key="btn_realtime", use_container_width=True, type="primary" if st.session_state.active_mode == "realtime" else "secondary"):
            st.session_state.active_mode = "realtime"
            st.rerun()

    spacer(6)

    if st.session_state.active_mode == "history":
        if df_full is not None and not df_full.empty:
            min_date = df_full.index.min().date()
            max_date = df_full.index.max().date()

            if "history_start" not in st.session_state:
                st.session_state["history_start"] = min_date
            if "history_end" not in st.session_state:
                st.session_state["history_end"] = max_date

            presets = [
                ("Toàn bộ", "pre_all", min_date, max_date),
                ("T7/2010", "pre_jul", datetime(2010, 7, 1).date(), datetime(2010, 7, 31).date()),
                ("T8/2010", "pre_aug", datetime(2010, 8, 1).date(), datetime(2010, 8, 31).date()),
                ("T9/2010", "pre_sep", datetime(2010, 9, 1).date(), datetime(2010, 9, 30).date()),
                ("T10/2010", "pre_oct", datetime(2010, 10, 1).date(), datetime(2010, 10, 31).date()),
                ("T11/2010", "pre_nov", datetime(2010, 11, 1).date(), min(datetime(2010, 11, 30).date(), max_date)),
            ]

            p_cols = st.columns([1, 1, 1, 1, 1, 1, 1.4, 1.4])
            for idx, (label, key_name, s_date, e_date) in enumerate(presets):
                with p_cols[idx]:
                    if st.button(label, key=key_name, use_container_width=True):
                        st.session_state["history_start"] = s_date
                        st.session_state["history_end"] = e_date
                        st.rerun()

            with p_cols[6]:
                st.date_input("Từ ngày", min_value=min_date, max_value=max_date, format="DD/MM/YYYY", key="history_start", label_visibility="collapsed")
            with p_cols[7]:
                st.date_input("Đến ngày", min_value=min_date, max_value=max_date, format="DD/MM/YYYY", key="history_end", label_visibility="collapsed")
    else:
        if "producer_pid" not in st.session_state:
            st.session_state.producer_pid = None

        producer_is_running = _is_producer_pid_alive(st.session_state.producer_pid)
        if not producer_is_running and st.session_state.producer_pid is not None:
            st.session_state.producer_pid = None

        r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([1, 1, 1, 1.8, 1.2])

        with r_col1:
            if st.button("Khởi động", key="btn_start_producer", use_container_width=True, type="primary" if not producer_is_running else "secondary", disabled=producer_is_running):
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
                    cwd=PROJECT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                st.session_state.producer_pid = proc.pid
                st.rerun()

        with r_col2:
            if st.button("Dừng", key="btn_stop_producer", use_container_width=True, type="primary" if producer_is_running else "secondary", disabled=not producer_is_running):
                _stop_producer_pid(st.session_state.producer_pid)
                st.session_state.producer_pid = None
                st.rerun()

        with r_col3:
            if st.button("Tiếp tục", key="btn_resume_producer", use_container_width=True, type="primary" if not producer_is_running else "secondary", disabled=producer_is_running):
                skip_count = st.session_state.get("rt_total", 0)
                producer_script = os.path.join(PROJECT_DIR, "src", "04_producer.py")
                env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
                proc = subprocess.Popen(
                    [sys.executable, producer_script, "--speed", "0.5", "--append", "--skip", str(skip_count)],
                    cwd=PROJECT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                st.session_state.producer_pid = proc.pid
                st.rerun()

        with r_col4:
            if producer_is_running:
                st.markdown(f'<div style="display:flex; align-items:center; gap:6px; padding-top:6px;"><span class="status-dot online"></span><span style="color:#10B981; font-weight:600; font-size:0.85em;">Producer đang chạy (PID: {st.session_state.producer_pid})</span></div>', unsafe_allow_html=True)
            else:
                has_data = st.session_state.get("rt_total", 0) > 0
                sub_text = f"Dừng ({st.session_state.get('rt_total', 0):,} mẫu)" if has_data else "Chưa chạy"
                st.markdown(f'<div style="display:flex; align-items:center; gap:6px; padding-top:6px;"><span class="status-dot offline"></span><span style="color:#64748B; font-size:0.85em; font-weight:500;">{sub_text}</span></div>', unsafe_allow_html=True)

        with r_col5:
            refresh_rate = st.number_input("Cập nhật (s):", min_value=1, max_value=10, value=st.session_state.get("refresh_rate", 2), step=1, key="num_rf")
            st.session_state.refresh_rate = refresh_rate

    spacer(8)


# --- View Renderers ---

def render_history_tab(model, scaler, feature_names: list[str]) -> None:
    """Hiển thị tab phân tích dữ liệu lịch sử."""
    df_full = load_full_demo_data()

    if df_full.empty:
        st.warning("Chưa có dữ liệu demo. Hãy chạy 03_inject_anomalies.py trước!")
        return

    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()

    start_date = st.session_state.get("history_start", min_date)
    end_date = st.session_state.get("history_end", max_date)

    mask = (df_full.index.date >= start_date) & (df_full.index.date <= end_date)
    df_filtered = df_full[mask].copy()

    if df_filtered.empty:
        st.info("Không có dữ liệu trong khoảng ngày đã chọn.")
        return

    sensor_cols = SENSOR_COLUMNS
    df_predict = predict_batch(df_filtered[sensor_cols], model, scaler, feature_names)

    if "is_anomaly" in df_filtered.columns:
        common_idx = df_predict.index.intersection(df_filtered.index)
        df_predict.loc[common_idx, "anomaly_type"] = df_filtered.loc[common_idx, "anomaly_type"]
        df_predict.loc[common_idx, "is_anomaly_gt"] = df_filtered.loc[common_idx, "is_anomaly"]

    # Phân loại bất thường (rule-based)
    anomaly_mask = df_predict["predicted_anomaly"]
    df_predict["classified_type"] = "normal"
    if anomaly_mask.any():
        df_predict.loc[anomaly_mask, "classified_type"] = classify_batch(df_predict.loc[anomaly_mask])

    # Tính baseline stats từ dữ liệu bình thường để giải thích anomaly
    feat_cols = [c for c in feature_names if c in df_predict.columns]
    normal_data = df_predict[~anomaly_mask]
    medians, iqrs = compute_feature_stats(normal_data, feat_cols) if len(normal_data) > 0 else ({}, {})

    # Tính explanation cho anomalies
    df_predict["explanation"] = "—"
    if anomaly_mask.any() and medians:
        df_predict.loc[anomaly_mask, "explanation"] = explain_batch(
            df_predict.loc[anomaly_mask], feat_cols, medians, iqrs
        )

    total = len(df_predict)
    n_pred_anomaly = anomaly_mask.sum()
    avg_power = float(df_predict["Global_active_power"].mean())
    anomaly_rate = (n_pred_anomaly / total * 100) if total > 0 else 0.0

    # --- 2-COLUMN DASHBOARD LAYOUT ---
    col_left, col_right = st.columns([1.85, 1.15])

    with col_left:
        # 1. Main Power Chart
        date_range_str = f"({start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')})"
        st.plotly_chart(build_main_power_chart(df_predict, title_suffix=date_range_str), width="stretch", key="hist_power")

        # 2. Voltage Chart
        st.plotly_chart(build_voltage_chart(df_predict), width="stretch", key="hist_voltage")

        # 3. Anomaly Table + Smart Filter Bar
        anomaly_rows = df_predict[df_predict["predicted_anomaly"]].copy()
        n_anomaly_display = len(anomaly_rows)

        st.markdown(f"""
        <div class="anomaly-table-header">
            <h4>Chi tiết các điểm bất thường</h4>
            <span class="table-count">{n_anomaly_display} điểm</span>
        </div>
        """, unsafe_allow_html=True)

        if not anomaly_rows.empty:
            filter_option = st.radio(
                "Lọc theo loại bất thường:",
                ["Tất cả", "⚡ Đột biến công suất", "📉 Sụt áp điện", "🌙 Đột biến đêm", "❓ Chưa xác định"],
                horizontal=True,
                key="hist_anomaly_filter"
            )

            filter_map = {
                "⚡ Đột biến công suất": "power_surge",
                "📉 Sụt áp điện": "voltage_drop",
                "🌙 Đột biến đêm": "night_spike",
                "❓ Chưa xác định": "unknown",
            }
            if filter_option in filter_map:
                target_type = filter_map[filter_option]
                anomaly_rows = anomaly_rows[anomaly_rows["classified_type"] == target_type]

            display_cols = {
                "Global_active_power": "Công suất (kW)",
                "Voltage": "Điện áp (V)",
                "anomaly_score": "Score",
                "classified_type": "Loại (AI)",
                "explanation": "Nguyên nhân chính",
            }
            if "anomaly_type" in anomaly_rows.columns:
                display_cols["anomaly_type"] = "Loại (GT)"

            existing_cols = {k: v for k, v in display_cols.items() if k in anomaly_rows.columns}
            table_df = anomaly_rows[list(existing_cols.keys())].copy()

            if "classified_type" in table_df.columns:
                table_df["classified_type"] = table_df["classified_type"].map(
                    lambda x: TYPE_LABELS.get(x, x)
                )

            table_df = table_df.rename(columns=existing_cols)
            table_df.index = table_df.index.strftime("%d/%m/%Y %H:%M")
            table_df.index.name = "Thời gian"

            for col in ["Công suất (kW)", "Điện áp (V)"]:
                if col in table_df.columns:
                    table_df[col] = table_df[col].round(3)
            if "Score" in table_df.columns:
                table_df["Score"] = table_df["Score"].round(4)

            st.dataframe(table_df.head(100), width="stretch", height=400)

            csv_data = table_df.to_csv(encoding="utf-8-sig")
            st.download_button(
                label="Xuất báo cáo CSV",
                data=csv_data,
                file_name=f"anomaly_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="btn_export_csv",
            )
        else:
            st.success("Không phát hiện bất thường trong khoảng thời gian này.")

    with col_right:
        # KPI Grid 2x2
        k1, k2 = st.columns(2)
        with k1:
            render_metric_card(
                f"{total:,}", "Tổng mẫu",
                variant="metric-card-primary",
                pill_text="DỮ LIỆU", pill_type="purple",
                progress_pct=100.0
            )
        with k2:
            render_metric_card(
                f"{n_pred_anomaly:,}", "Model phát hiện",
                variant="metric-card-red",
                pill_text="ANOMALY", pill_type="red",
                progress_pct=min(100.0, anomaly_rate * 5)
            )

        k3, k4 = st.columns(2)
        with k3:
            render_metric_card(
                f"{avg_power:.2f} kW", "Công suất TB",
                variant="metric-card-primary",
                pill_text="TRUNG BÌNH", pill_type="purple",
                progress_pct=min(100.0, (avg_power / 5.0) * 100)
            )
        with k4:
            render_metric_card(
                f"{anomaly_rate:.1f}%", "Tỷ lệ anomaly",
                variant="metric-card-green",
                pill_text="TỶ LỆ %", pill_type="green",
                progress_pct=min(100.0, anomaly_rate * 5)
            )

        spacer(4)

        # Anomaly Timeline Heatmap
        st.plotly_chart(build_anomaly_timeline_heatmap(df_predict), width="stretch", key="hist_heatmap")

        # Hourly Heatmap / Chart
        st.plotly_chart(build_hourly_heatmap(df_predict), width="stretch", key="hist_hourly")

        # Model Performance Summary
        _render_performance_summary()


def _render_performance_summary() -> None:
    """Hiển thị tóm tắt hiệu năng mô hình từ reports/."""
    metrics_path = os.path.join(REPORTS_DIR, "metrics_summary.csv")
    per_type_path = os.path.join(REPORTS_DIR, "metrics_per_type.csv")

    if not os.path.exists(metrics_path):
        return

    st.markdown(
        f'<div class="anomaly-table-header">'
        f'<h4>Hiệu năng mô hình (Evaluation)</h4>'
        f'<span class="table-count">Isolation Forest</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    metrics_df = pd.read_csv(metrics_path)
    metrics_map = dict(zip(metrics_df["Metric"], metrics_df["Value"]))

    def _fmt2(val):
        try:
            return f"{float(val):.2f}"
        except (ValueError, TypeError):
            return str(val)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        render_metric_card(_fmt2(metrics_map.get("Precision", "—")), "Precision", variant="metric-card-primary")
    with mc2:
        render_metric_card(_fmt2(metrics_map.get("Recall", "—")), "Recall", variant="metric-card-primary")
    with mc3:
        render_metric_card(_fmt2(metrics_map.get("F1-Score", "—")), "F1-Score", variant="metric-card-accent")
    with mc4:
        render_metric_card(_fmt2(metrics_map.get("ROC-AUC", "—")), "ROC-AUC", variant="metric-card-green")

    if os.path.exists(per_type_path):
        spacer(4)
        per_type_df = pd.read_csv(per_type_path)
        st.dataframe(per_type_df, width="stretch", hide_index=True, height=180)


def render_realtime_tab(model, scaler, feature_names: list[str]) -> None:
    """Hiển thị tab giám sát luồng thời gian thực."""
    refresh_rate = st.session_state.get("refresh_rate", 2)
    producer_is_running = _is_producer_pid_alive(st.session_state.get("producer_pid"))

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

    new_messages = []
    if producer_is_running:
        new_messages, new_pos = read_new_messages(st.session_state.rt_file_pos)
        st.session_state.rt_file_pos = new_pos

    for msg in new_messages:
        st.session_state.rt_total += 1
        st.session_state.rt_buffer.append(msg)
        if len(st.session_state.rt_buffer) > BUFFER_SIZE:
            st.session_state.rt_buffer.pop(0)

        buf_df = pd.DataFrame(st.session_state.rt_buffer)
        buf_df["datetime"] = pd.to_datetime(buf_df["timestamp"])
        buf_df = buf_df.set_index("datetime")

        buf_sensor = buf_df[SENSOR_COLUMNS].astype(float)
        features = create_features_realtime(buf_sensor)
        predicted_anomaly = False
        anomaly_score = 0.0

        if features is not None:
            predicted_anomaly, anomaly_score = predict_single(features, model, scaler, feature_names)

        if predicted_anomaly:
            st.session_state.rt_anomalies += 1

        # Phân loại bất thường real-time
        classified = "normal"
        if predicted_anomaly and features is not None:
            classified = classify_anomaly_type(features)

        point = {
            "timestamp": msg["timestamp"],
            "power": float(msg["Global_active_power"]),
            "voltage": float(msg["Voltage"]),
            "intensity": float(msg.get("Global_intensity", 0)),
            "predicted_anomaly": predicted_anomaly,
            "anomaly_score": anomaly_score,
            "ground_truth": msg.get("is_anomaly", 0) == 1,
            "anomaly_type": msg.get("anomaly_type", "normal"),
            "classified_type": classified,
        }
        st.session_state.rt_display.append(point)

        max_pts = st.session_state.get("max_display", MAX_DISPLAY_POINTS)
        if len(st.session_state.rt_display) > max_pts:
            st.session_state.rt_display = st.session_state.rt_display[-max_pts:]

        if predicted_anomaly:
            st.session_state.rt_last_anomaly = point

    col_left, col_right = st.columns([1.85, 1.15])

    with col_left:
        if st.session_state.rt_last_anomaly:
            a = st.session_state.rt_last_anomaly
            alert_type = a.get("classified_type", a["anomaly_type"])
            render_alert_bar(alert_type, a["timestamp"], a["power"], a["voltage"], a["anomaly_score"])

        st.plotly_chart(build_main_power_chart(st.session_state.rt_display, title_suffix="— Thời gian thực"), width="stretch", key="rt_power")

        anomaly_logs = [d for d in st.session_state.rt_display if d["predicted_anomaly"]]
        n_logs = len(anomaly_logs)

        st.markdown(f"""
        <div class="anomaly-table-header">
            <h4>Log Anomaly gần nhất</h4>
            <span class="table-count">{n_logs} điểm</span>
        </div>
        """, unsafe_allow_html=True)

        if anomaly_logs:
            log_df = pd.DataFrame(anomaly_logs[-30:][::-1])
            log_cols = ["timestamp", "power", "voltage", "anomaly_score", "classified_type"]
            log_cols = [c for c in log_cols if c in log_df.columns]
            log_df = log_df[log_cols]
            col_rename = {"timestamp": "Thời gian", "power": "Công suất (kW)", "voltage": "Điện áp (V)", "anomaly_score": "Score", "classified_type": "Loại (AI)"}
            log_df = log_df.rename(columns={k: v for k, v in col_rename.items() if k in log_df.columns})
            if "Loại (AI)" in log_df.columns:
                log_df["Loại (AI)"] = log_df["Loại (AI)"].map(lambda x: TYPE_LABELS.get(x, x))
            st.dataframe(log_df, width="stretch", hide_index=True, height=360)
        else:
            st.info("Chưa phát hiện anomaly. Đang giám sát...")

    with col_right:
        rc1, rc2 = st.columns(2)
        with rc1:
            render_metric_card(
                f"{st.session_state.rt_total:,}", "Tổng nhận",
                variant="metric-card-primary",
                pill_text="REALTIME", pill_type="purple",
                delta=f"+{len(new_messages)}" if new_messages else None,
                delta_color=COLORS["success"]
            )
        with rc2:
            render_metric_card(
                f"{st.session_state.rt_anomalies}", "Anomaly",
                variant="metric-card-red",
                pill_text="BẤT THƯỜNG", pill_type="red"
            )

        rc3, rc4 = st.columns(2)
        with rc3:
            rate = (st.session_state.rt_anomalies / st.session_state.rt_total * 100 if st.session_state.rt_total > 0 else 0)
            render_metric_card(
                f"{rate:.1f}%", "Tỷ lệ anomaly",
                variant="metric-card-accent",
                pill_text="TỶ LỆ %", pill_type="orange",
                progress_pct=min(100.0, rate * 5)
            )
        with rc4:
            latest_power = st.session_state.rt_display[-1]["power"] if st.session_state.rt_display else 0
            render_metric_card(
                f"{latest_power:.3f} kW", "Công suất",
                variant="metric-card-primary",
                pill_text="HIỆN TẠI", pill_type="purple",
                progress_pct=min(100.0, (latest_power / 5.0) * 100)
            )

        spacer(4)
        _render_performance_summary()

    if producer_is_running:
        time.sleep(refresh_rate)
        st.rerun()


# --- Main Application Entry ---

def main():
    """Hàm khởi tạo ứng dụng chính."""
    st.set_page_config(
        page_title="Smart Meter Anomaly Detection",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    load_css()

    st.markdown('<div class="dashboard-main-title">Smart meter anomaly</div>', unsafe_allow_html=True)

    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "history"

    df_full = load_full_demo_data() if st.session_state.active_mode == "history" else None
    render_top_control_bar(df_full)

    try:
        model, scaler, feature_names = load_model()
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        st.info("Hãy chạy python src/02_train_model.py trước.")
        st.stop()

    if st.session_state.active_mode == "history":
        render_history_tab(model, scaler, feature_names)
    else:
        render_realtime_tab(model, scaler, feature_names)


if __name__ == "__main__":
    if not st.runtime.exists():
        print("Please run this dashboard using Streamlit CLI: python -m streamlit run Smart-meter-anomaly/src/05_dashboard.py")
        sys.exit(0)
    else:
        main()
