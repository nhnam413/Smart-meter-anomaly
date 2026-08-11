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
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

from config import (
    MODELS_DIR, STREAM_FILE, DEMO_CSV, CSS_FILE,
    COLORS, MAX_DISPLAY_POINTS, BUFFER_SIZE, SENSOR_COLUMNS,
    PROJECT_DIR,
)
from features import create_features, create_features_realtime


# --- Cache & Data Loader ---

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "isolation_forest_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return model, scaler, feature_names


@st.cache_data
def load_full_demo_data() -> pd.DataFrame:
    if not os.path.exists(DEMO_CSV):
        return pd.DataFrame()
    return pd.read_csv(DEMO_CSV, index_col="datetime", parse_dates=True)


# --- Inference Helpers ---

def predict_batch(df: pd.DataFrame, model, scaler, feature_names) -> pd.DataFrame:
    df_feat = create_features(df)
    sensor_and_feat_cols = [c for c in feature_names if c in df_feat.columns]

    X = df_feat[sensor_and_feat_cols].values
    X_scaled = scaler.transform(X)

    df_feat["predicted_anomaly"] = model.predict(X_scaled) == -1
    df_feat["anomaly_score"] = model.decision_function(X_scaled)
    return df_feat


def predict_single(features, model, scaler, feature_names):
    X = features[feature_names].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    score = model.decision_function(X_scaled)[0]
    return prediction == -1, score


def read_new_messages(last_position: int) -> tuple:
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
    font=dict(family="Inter, -apple-system, sans-serif", color="#1F2937"),
    paper_bgcolor="#FAFAFA",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=50, r=20, t=50, b=50),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(size=11, color="#1F2937")
    ),
)

AXIS_STYLE = dict(
    gridcolor="#E5E7EB",
    tickfont=dict(color="#1F2937"),
    linecolor="#D1D5DB",
)


def _extract_chart_data(display_data, value_col, df_col):
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
    if display_data is None or (isinstance(display_data, pd.DataFrame) and display_data.empty):
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, height=420, title="Chưa có dữ liệu...")
        return fig

    timestamps, powers, is_anomaly = _extract_chart_data(display_data, "power", "Global_active_power")
    scores = display_data["anomaly_score"] if isinstance(display_data, pd.DataFrame) else [d["anomaly_score"] for d in display_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=powers, mode="lines",
        name="Công suất (kW)",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(107, 76, 230, 0.06)",
        hovertemplate="<b>%{x}</b><br>Công suất: %{y:.3f} kW<extra></extra>"
    ))

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
            marker=dict(color=COLORS["accent"], size=10, symbol="circle", line=dict(width=2, color="white")),
            customdata=anom_sc,
            hovertemplate="<b>BẤT THƯỜNG</b><br>Thời điểm: %{x}<br>Công suất: %{y:.3f} kW<br>Score: %{customdata:.4f}<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT, height=420,
        title=dict(text=f"Biểu đồ công suất tiêu thụ điện năng {title_suffix}", font=dict(size=15, color=COLORS["text_primary"]), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Thời gian", font=dict(size=12, color="#1F2937")), showgrid=True, **AXIS_STYLE),
        yaxis=dict(title=dict(text="Global Active Power (kW)", font=dict(size=12, color="#1F2937")), showgrid=True, zeroline=True, zerolinecolor="#D1D5DB", **AXIS_STYLE),
    )
    return fig


def build_voltage_chart(display_data) -> go.Figure:
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
        name="Điện áp (V)", line=dict(color=COLORS["primary"], width=2),
    ))

    if isinstance(display_data, pd.DataFrame):
        anom_mask = display_data["predicted_anomaly"]
        anom_ts, anom_v = display_data.index[anom_mask], display_data.loc[anom_mask, "Voltage"]
    else:
        anom_ts = [t for t, a in zip(timestamps, is_anomaly) if a]
        anom_v = [v for v, a in zip(voltages, is_anomaly) if a]

    if len(anom_ts) > 0:
        fig.add_trace(go.Scatter(
            x=anom_ts, y=anom_v, mode="markers", name="Bất thường",
            marker=dict(color=COLORS["accent"], size=8, symbol="circle", line=dict(width=1.5, color="white")),
        ))

    fig.update_layout(
        **CHART_LAYOUT, height=320,
        title=dict(text="Điện áp (Voltage)", font=dict(size=14, color=COLORS["text_primary"]), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Thời gian", font=dict(size=12, color="#1F2937")), **AXIS_STYLE),
        yaxis=dict(title=dict(text="Voltage (V)", font=dict(size=12, color="#1F2937")), **AXIS_STYLE),
    )
    return fig


def build_anomaly_pie(df: pd.DataFrame) -> go.Figure:
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
        "power_surge": "Đột biến công suất",
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
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(size=10, color=COLORS["text_secondary"])),
    )
    return fig


def build_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
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
            colorscale=[[0, COLORS["primary_light"]], [0.5, COLORS["primary"]], [1, COLORS["primary_dark"]]],
            line=dict(width=0),
            cornerradius=4,
        ),
        hovertemplate="Giờ %{x}h<br>TB: %{y:.3f} kW<extra></extra>",
    )])

    fig.update_layout(
        **CHART_LAYOUT, height=320,
        title=dict(text="Trung bình công suất theo giờ", font=dict(size=14, color=COLORS["text_primary"]), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Giờ", font=dict(size=12, color="#1F2937")), dtick=2, **AXIS_STYLE),
        yaxis=dict(title=dict(text="Power (kW)", font=dict(size=12, color="#1F2937")), **AXIS_STYLE),
        bargap=0.15,
    )
    return fig


# --- UI Utilities & Components ---

def load_css():
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE, "r", encoding="utf-8") as f:
            st.markdown(f"<style>\n{f.read()}\n</style>", unsafe_allow_html=True)


def spacer(px: int = 12):
    st.markdown(f"<div style='height:{px}px;'></div>", unsafe_allow_html=True)


def render_metric_card(value: str, label: str, color: str = "#0F172A", delta: str = None, delta_color: str = "#10B981", variant: str = None):
    delta_html = f'<div class="metric-delta" style="color:{delta_color};">{delta}</div>' if delta else ""
    variant_class = f" {variant}" if variant else ""
    val_color = f' style="color:{color} !important;"' if color and not variant else ''

    st.markdown(f"""
    <div class="metric-card{variant_class}">
        <div class="metric-value"{val_color}>{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_alert_bar(anomaly_type: str, timestamp: str, power: float, voltage: float, score: float):
    type_labels = {
        "power_surge": "Đột biến công suất",
        "voltage_drop": "Sụt áp điện",
        "night_spike": "Bất thường ban đêm",
        "normal": "Bất thường phát hiện",
    }
    label = type_labels.get(anomaly_type, "Bất thường")

    st.markdown(f"""
    <div class="anomaly-alert-bar">
        <div class="alert-title">CẢNH BÁO — {label.upper()}</div>
        <div class="alert-detail">
            Thời điểm: <b>{timestamp}</b> &nbsp;·&nbsp;
            Công suất: <b>{power:.3f} kW</b> &nbsp;·&nbsp;
            Điện áp: <b>{voltage:.1f}V</b> &nbsp;·&nbsp;
            Score: <b>{score:.4f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Process Management ---

def _is_producer_pid_alive(pid) -> bool:
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


def _stop_producer_pid(pid):
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

def render_top_control_bar(df_full=None):
    st.markdown('<div class="top-control-card">', unsafe_allow_html=True)

    t_col1, t_col2, t_col3 = st.columns([2.5, 1, 1])

    with t_col1:
        badge_class = "title-badge" if st.session_state.active_mode == "history" else "title-badge-rt"
        badge_text = "HISTORY" if st.session_state.active_mode == "history" else "REAL-TIME"
        st.markdown(f"""
        <div class="dashboard-title" style="padding:0; margin:0;">
            <h1>Smart Meter Anomaly Detection</h1>
            <span class="{badge_class}">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        if st.button("Phân tích lịch sử", key="btn_history", use_container_width=True, type="primary" if st.session_state.active_mode == "history" else "secondary"):
            st.session_state.active_mode = "history"
            st.rerun()

    with t_col3:
        if st.button("Giám sát thời gian thực", key="btn_realtime", use_container_width=True, type="primary" if st.session_state.active_mode == "realtime" else "secondary"):
            st.session_state.active_mode = "realtime"
            st.rerun()

    spacer(12)

    if st.session_state.active_mode == "history":
        if df_full is not None and not df_full.empty:
            min_date = df_full.index.min().date()
            max_date = df_full.index.max().date()

            if "history_start" not in st.session_state:
                st.session_state["history_start"] = min_date
            if "history_end" not in st.session_state:
                st.session_state["history_end"] = max_date

            st.markdown('<p style="font-size:0.75em; font-weight:700; color:#6B7280; text-transform:uppercase; margin:0 0 6px 0;">Chọn nhanh mốc thời gian:</p>', unsafe_allow_html=True)

            presets = [
                ("Toàn bộ", "pre_all", min_date, max_date),
                ("T7/2010", "pre_jul", datetime(2010, 7, 1).date(), datetime(2010, 7, 31).date()),
                ("T8/2010", "pre_aug", datetime(2010, 8, 1).date(), datetime(2010, 8, 31).date()),
                ("T9/2010", "pre_sep", datetime(2010, 9, 1).date(), datetime(2010, 9, 30).date()),
                ("T10/2010", "pre_oct", datetime(2010, 10, 1).date(), datetime(2010, 10, 31).date()),
                ("T11/2010", "pre_nov", datetime(2010, 11, 1).date(), min(datetime(2010, 11, 30).date(), max_date)),
            ]

            p_cols = st.columns([1, 1, 1, 1, 1, 1, 2])
            for idx, (label, key_name, s_date, e_date) in enumerate(presets):
                with p_cols[idx]:
                    if st.button(label, key=key_name, use_container_width=True):
                        st.session_state["history_start"] = s_date
                        st.session_state["history_end"] = e_date
                        st.rerun()

            with p_cols[6]:
                st.markdown(f"""
                <div style="padding-top:8px; color:#4B5563; font-size:0.82em; font-weight:600; text-align:right;">
                    Khoảng thời gian: <span style="color:#6B4CE6;">{st.session_state['history_start'].strftime('%d/%m/%Y')}</span> → <span style="color:#6B4CE6;">{st.session_state['history_end'].strftime('%d/%m/%Y')}</span>
                </div>
                """, unsafe_allow_html=True)

            spacer(8)

            d_col1, d_col2, d_col3 = st.columns([1.3, 1.3, 2.1])
            with d_col1:
                st.date_input("Từ ngày:", min_value=min_date, max_value=max_date, format="DD/MM/YYYY", key="history_start")
            with d_col2:
                st.date_input("Đến ngày:", min_value=min_date, max_value=max_date, format="DD/MM/YYYY", key="history_end")
            with d_col3:
                st.markdown('<div style="padding-top:28px; color:#6B7280; font-size:0.85em; font-weight:500;">Chọn mốc thời gian xem dữ liệu quá khứ.</div>', unsafe_allow_html=True)
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
                st.markdown(f'<div style="display:flex; align-items:center; gap:6px; padding-top:10px;"><span class="status-dot online"></span><span style="color:#10B981; font-weight:600; font-size:0.85em;">Producer đang chạy (PID: {st.session_state.producer_pid})</span></div>', unsafe_allow_html=True)
            else:
                has_data = st.session_state.get("rt_total", 0) > 0
                sub_text = f"Dừng ({st.session_state.get('rt_total', 0):,} mẫu)" if has_data else "Chưa chạy"
                st.markdown(f'<div style="display:flex; align-items:center; gap:6px; padding-top:10px;"><span class="status-dot offline"></span><span style="color:#9CA3AF; font-size:0.85em; font-weight:500;">{sub_text}</span></div>', unsafe_allow_html=True)

        with r_col5:
            refresh_rate = st.number_input("Cập nhật (s):", min_value=1, max_value=10, value=st.session_state.get("refresh_rate", 2), step=1, key="num_rf")
            st.session_state.refresh_rate = refresh_rate

    st.markdown('</div>', unsafe_allow_html=True)


# --- View Renderers ---

def render_history_tab(model, scaler, feature_names):
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

    total = len(df_predict)
    n_pred_anomaly = df_predict["predicted_anomaly"].sum()
    n_gt_anomaly = int(df_predict.get("is_anomaly_gt", pd.Series(dtype=int)).sum())
    avg_power = df_predict["Global_active_power"].mean()
    anomaly_rate = (n_pred_anomaly / total * 100) if total > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card(f"{total:,}", "Tổng mẫu", variant="metric-card-primary")
    with c2:
        render_metric_card(f"{n_pred_anomaly:,}", "Model phát hiện", variant="metric-card-red")
    with c3:
        render_metric_card(f"{n_gt_anomaly}", "Nhãn thật (GT)", variant="metric-card-accent")
    with c4:
        render_metric_card(f"{avg_power:.2f} kW", "Công suất TB", variant="metric-card-primary")
    with c5:
        render_metric_card(f"{anomaly_rate:.1f}%", "Tỷ lệ bất thường", variant="metric-card-green")

    spacer(12)

    date_range_str = f"({start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')})"
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(build_main_power_chart(df_predict, title_suffix=date_range_str), width="stretch", key="hist_power")
    st.markdown('</div>', unsafe_allow_html=True)

    anomaly_rows = df_predict[df_predict["predicted_anomaly"]].copy()
    n_anomaly_display = len(anomaly_rows)

    st.markdown(f"""
    <div class="anomaly-table-header">
        <h4>Chi tiết các điểm bất thường</h4>
        <span class="table-count">{n_anomaly_display} điểm</span>
    </div>
    """, unsafe_allow_html=True)

    if not anomaly_rows.empty:
        display_cols = {
            "Global_active_power": "Công suất (kW)",
            "Voltage": "Điện áp (V)",
            "Global_intensity": "Dòng điện (A)",
            "anomaly_score": "Score",
        }
        if "anomaly_type" in anomaly_rows.columns:
            display_cols["anomaly_type"] = "Loại (GT)"

        existing_cols = {k: v for k, v in display_cols.items() if k in anomaly_rows.columns}
        table_df = anomaly_rows[list(existing_cols.keys())].copy()
        table_df = table_df.rename(columns=existing_cols)
        table_df.index = table_df.index.strftime("%d/%m/%Y %H:%M")
        table_df.index.name = "Thời gian"

        for col in ["Công suất (kW)", "Điện áp (V)", "Dòng điện (A)"]:
            if col in table_df.columns:
                table_df[col] = table_df[col].round(3)
        if "Score" in table_df.columns:
            table_df["Score"] = table_df["Score"].round(4)

        st.dataframe(table_df.head(100), width="stretch", height=450)
    else:
        st.success("Không phát hiện bất thường trong khoảng thời gian này.")

    spacer(12)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(build_voltage_chart(df_predict), width="stretch", key="hist_voltage")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(build_hourly_heatmap(df_predict), width="stretch", key="hist_hourly")
        st.markdown('</div>', unsafe_allow_html=True)

    if "anomaly_type" in df_predict.columns:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(build_anomaly_pie(df_predict), width="stretch", key="hist_pie")
        st.markdown('</div>', unsafe_allow_html=True)


def render_realtime_tab(model, scaler, feature_names):
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

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            f"{st.session_state.rt_total:,}", "Tổng nhận",
            variant="metric-card-primary",
            delta=f"+{len(new_messages)}" if new_messages else None,
            delta_color=COLORS["success"]
        )
    with c2:
        render_metric_card(f"{st.session_state.rt_anomalies}", "Anomaly", variant="metric-card-red")
    with c3:
        rate = (st.session_state.rt_anomalies / st.session_state.rt_total * 100 if st.session_state.rt_total > 0 else 0)
        render_metric_card(f"{rate:.1f}%", "Tỷ lệ anomaly", variant="metric-card-accent")
    with c4:
        latest_power = st.session_state.rt_display[-1]["power"] if st.session_state.rt_display else 0
        render_metric_card(f"{latest_power:.3f}", "Công suất (kW)", variant="metric-card-primary")

    spacer(8)

    if st.session_state.rt_last_anomaly:
        a = st.session_state.rt_last_anomaly
        render_alert_bar(a["anomaly_type"], a["timestamp"], a["power"], a["voltage"], a["anomaly_score"])

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(build_main_power_chart(st.session_state.rt_display, title_suffix="— Thời gian thực"), width="stretch", key="rt_power")
    st.markdown('</div>', unsafe_allow_html=True)

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
        log_df = log_df[["timestamp", "power", "voltage", "anomaly_score", "anomaly_type"]]
        log_df.columns = ["Thời gian", "Công suất (kW)", "Điện áp (V)", "Score", "Loại"]
        st.dataframe(log_df, width="stretch", hide_index=True, height=400)
    else:
        st.info("Chưa phát hiện anomaly. Đang giám sát...")

    if producer_is_running:
        time.sleep(refresh_rate)
        st.rerun()


# --- Main Application Entry ---

def main():
    st.set_page_config(
        page_title="Smart Meter Anomaly Detection",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    load_css()

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
        print("Please run this dashboard using Streamlit CLI: streamlit run src/05_dashboard.py")
        sys.exit(0)
    else:
        main()
