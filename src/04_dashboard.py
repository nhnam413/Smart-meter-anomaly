"""
04_dashboard.py - Dashboard giam sat va phan tich bat thuong dien nang.
Giao dien gom 2 che do: Phan tich lich su va Giam sat thoi gian thuc.
"""

import os
import time
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    MODEL_BUNDLE_PATH, DEMO_STREAM_PATH, CSS_FILE,
    COLORS, ANOMALY_TYPE_COLORS, TYPE_LABELS,
    MAX_DISPLAY_POINTS, TARGET_COL,
)
from features import (
    extract_features, extract_latest,
    calc_severity, get_severity_level,
    classify_type, explain_anomaly,
)

# Cau hinh giao dien bieu do
CHART_THEME = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, sans-serif", color=COLORS["text_primary"]),
    margin=dict(l=40, r=20, t=35, b=35),
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["surface"],
)

TABLE_COLS = [
    "Thời gian", "Công suất (kW)", "Điện áp (V)",
    "Mức độ (Severity)", "Phân loại lỗi (AI)", "Nguyên nhân chính (XAI)"
]


# 1. Nap mo hinh va du lieu

@st.cache_resource
def load_bundle():
    """Tai goi mo hinh da huan luyen."""
    if not os.path.exists(MODEL_BUNDLE_PATH):
        return None
    return joblib.load(MODEL_BUNDLE_PATH)


@st.cache_data
def load_historical_data():
    """Tai du lieu demo day du."""
    if not os.path.exists(DEMO_STREAM_PATH):
        return pd.DataFrame()
    return pd.read_csv(DEMO_STREAM_PATH, index_col="datetime", parse_dates=True)


def load_custom_css():
    """Nap file CSS style.css."""
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# 2. Tao bieu do Plotly

def build_power_line_chart(df: pd.DataFrame) -> go.Figure:
    """Bieu do duong cong suat tieu thu (kW) kem diem bat thuong."""
    fig = go.Figure()
    if df.empty or TARGET_COL not in df.columns:
        fig.update_layout(**CHART_THEME, title="Chua co du lieu...")
        return fig

    # Duong cong suat
    fig.add_trace(go.Scatter(
        x=df.index, y=df[TARGET_COL], mode="lines",
        name="Công suất (kW)", line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy", fillcolor=COLORS["primary_rgba_05"],
        hovertemplate="<b>%{x}</b><br>Công suất: %{y:.3f} kW<extra></extra>",
    ))

    # Diem bat thuong
    if "is_anomaly" in df.columns:
        anom_df = df[df["is_anomaly"] == 1]
        if not anom_df.empty:
            fig.add_trace(go.Scatter(
                x=anom_df.index, y=anom_df[TARGET_COL], mode="markers",
                name="Bất thường",
                marker=dict(color=COLORS["danger"], size=8.5, symbol="circle", line=dict(width=1.5, color=COLORS["surface"])),
                customdata=anom_df.get("anomaly_type", "Bất thường"),
                hovertemplate="<b>BẤT THƯỜNG</b><br>Thời gian: %{x}<br>Công suất: %{y:.3f} kW<br>Dạng lỗi: %{customdata}<extra></extra>",
            ))

    fig.update_layout(
        **CHART_THEME, height=310,
        title=dict(text="Biểu đồ đường công suất tiêu thụ điện năng (kW)", font=dict(size=13, color=COLORS["text_primary"]), x=0),
        xaxis=dict(title=dict(text="Thời gian", font=dict(size=10)), showgrid=True, gridcolor=COLORS["grid_line"]),
        yaxis=dict(title=dict(text="Global Active Power (kW)", font=dict(size=10)), showgrid=True, gridcolor=COLORS["grid_line"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_voltage_line_chart(df: pd.DataFrame) -> go.Figure:
    """Bieu do duong dien ap (V) kem dai an toan 220-250V."""
    fig = go.Figure()
    if df.empty or "Voltage" not in df.columns:
        fig.update_layout(**CHART_THEME, title="Chua co du lieu...")
        return fig

    # Dai an toan dien ap
    fig.add_hrect(
        y0=220, y1=250, fillcolor=COLORS["success_rgba_08"], line_width=0,
        annotation_text="Vùng an toàn (220–250V)", annotation_position="top left",
        annotation_font=dict(color=COLORS["success_dark"], size=10),
    )

    # Duong dien ap
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Voltage"], mode="lines",
        name="Điện áp (V)", line=dict(color=COLORS["primary_dark"], width=1.8),
        hovertemplate="<b>%{x}</b><br>Điện áp: %{y:.1f} V<extra></extra>",
    ))

    # Diem su co dien ap
    if "is_anomaly" in df.columns:
        anom_df = df[df["is_anomaly"] == 1]
        if not anom_df.empty:
            fig.add_trace(go.Scatter(
                x=anom_df.index, y=anom_df["Voltage"], mode="markers",
                name="Sự cố điện áp",
                marker=dict(color=COLORS["danger"], size=8.0, symbol="circle", line=dict(width=1.5, color=COLORS["surface"])),
                hovertemplate="<b>SỰ CỐ ĐIỆN ÁP</b><br>Thời gian: %{x}<br>Điện áp: %{y:.1f} V<extra></extra>",
            ))

    fig.update_layout(
        **CHART_THEME, height=270,
        title=dict(text="Biểu đồ đường điện áp lưới điện (Voltage)", font=dict(size=13, color=COLORS["text_primary"]), x=0),
        xaxis=dict(title=dict(text="Thời gian", font=dict(size=10)), showgrid=True, gridcolor=COLORS["grid_line"]),
        yaxis=dict(title=dict(text="Voltage (V)", font=dict(size=10)), showgrid=True, gridcolor=COLORS["grid_line"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_anomaly_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Bieu do Donut the hien ty le % cac loai bat thuong."""
    types = ["power_surge", "voltage_drop", "night_spike"]
    labels = [TYPE_LABELS.get(t, t) for t in types]
    counts = [0] * len(types)
    slice_colors = [ANOMALY_TYPE_COLORS.get(t, COLORS["primary"]) for t in types]

    if not df.empty and "is_anomaly" in df.columns and "anomaly_type" in df.columns:
        anom_df = df[df["is_anomaly"] == 1]
        if not anom_df.empty:
            val_counts = anom_df["anomaly_type"].value_counts()
            counts = [int(val_counts.get(t, 0)) for t in types]

    total_anom = sum(counts)

    fig = go.Figure()
    if total_anom == 0:
        fig.add_trace(go.Pie(
            labels=["Bình thường"],
            values=[1],
            hole=0.58,
            marker=dict(colors=[COLORS["border"]]),
            textinfo="none",
            hoverinfo="none",
        ))
        center_text = f"<b>0</b><br><span style='font-size:10px; color:{COLORS['text_muted']};'>sự cố</span>"
    else:
        fig.add_trace(go.Pie(
            labels=labels,
            values=counts,
            hole=0.58,
            marker=dict(colors=slice_colors, line=dict(color=COLORS["surface"], width=2)),
            textinfo="percent",
            textposition="inside",
            textfont=dict(size=11, color=COLORS["surface"], family="Inter, sans-serif"),
            hovertemplate="<b>%{label}</b><br>Số lượng: %{value:,} vụ<br>Tỷ lệ: %{percent}<extra></extra>",
        ))
        center_text = f"<b>{total_anom:,}</b><br><span style='font-size:10px; color:{COLORS['text_muted']};'>sự cố</span>"

    fig.update_layout(
        **{**CHART_THEME, "margin": dict(l=20, r=20, t=35, b=45)},
        height=310,
        title=dict(text="Tỷ lệ phân bố các loại bất thường", font=dict(size=13, color=COLORS["text_primary"]), x=0),
        annotations=[dict(
            text=center_text,
            x=0.5, y=0.5, font_size=15, font_color=COLORS["text_primary"], showarrow=False
        )],
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
    )
    return fig


def build_anomaly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Ban do nhiet (Heatmap) mat do bat thuong 24h x 7 ngay."""
    day_labels = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
    hours = [f"{h:02d}:00" for h in range(24)]
    z_matrix = np.zeros((24, 7), dtype=int)

    if not df.empty and "is_anomaly" in df.columns:
        anom_df = df[df["is_anomaly"] == 1]
        if not anom_df.empty:
            for dt in anom_df.index:
                z_matrix[dt.hour, dt.dayofweek] += 1

    heatmap_colorscale = [
        [0.0, COLORS["surface"]],
        [0.001, COLORS["primary_light"]],
        [0.35, "#93C5FD"],
        [0.70, COLORS["primary"]],
        [1.00, COLORS["danger"]],
    ]

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=day_labels,
        y=hours,
        colorscale=heatmap_colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text="Số lỗi", font=dict(size=10)),
            thickness=10,
            len=0.85,
            tickfont=dict(size=9),
            outlinewidth=0,
        ),
        hovertemplate="<b>%{x}</b> lúc <b>%{y}</b><br>Số điểm bất thường: <b>%{z}</b><extra></extra>",
    ))

    fig.update_layout(
        **{**CHART_THEME, "margin": dict(l=45, r=20, t=35, b=25)},
        height=270,
        title=dict(text="Bản đồ nhiệt bất thường (24 giờ × 7 ngày)", font=dict(size=13, color=COLORS["text_primary"]), x=0),
        xaxis=dict(title="", showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="", showgrid=False, dtick=3, tickfont=dict(size=10), autorange="reversed"),
    )
    return fig


def calc_peak_hour_stats(df: pd.DataFrame) -> tuple[int, int, list[int]]:
    """Tinh so luong bat thuong theo 24 gio va tra ve (peak_hour, peak_count, hour_counts)."""
    hour_counts = [0] * 24
    if not df.empty and "is_anomaly" in df.columns:
        anom_df = df[df["is_anomaly"] == 1]
        if not anom_df.empty:
            counts = anom_df.index.hour.value_counts()
            for h, c in counts.items():
                hour_counts[h] = int(c)
    peak_hour = int(np.argmax(hour_counts)) if sum(hour_counts) > 0 else 0
    peak_count = hour_counts[peak_hour]
    return peak_hour, peak_count, hour_counts


def build_hourly_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Bieu do cot phan phoi bat thuong theo 24 gio."""
    hours = list(range(24))
    peak_hour, peak_count, hour_counts = calc_peak_hour_stats(df)
    bar_colors = [COLORS["danger"] if h == peak_hour and peak_count > 0 else COLORS["primary"] for h in hours]

    fig = go.Figure(go.Bar(
        x=[f"{h:02d}h" for h in hours],
        y=hour_counts,
        marker=dict(color=bar_colors, opacity=0.85),
        hovertemplate="<b>Khung giờ: %{x}</b><br>Số lượng bất thường: %{y} điểm<extra></extra>",
    ))
    fig.update_layout(
        **CHART_THEME, height=270,
        title=dict(text="Phân bố bất thường theo 24 giờ (00:00 - 23:00)", font=dict(size=13, color=COLORS["text_primary"]), x=0),
        xaxis=dict(title=dict(text="Khung giờ", font=dict(size=10)), showgrid=False),
        yaxis=dict(title=dict(text="Số điểm bất thường", font=dict(size=10)), showgrid=True, gridcolor=COLORS["grid_line"]),
    )
    return fig


# 3. Thanh phan giao dien va bang bieu

def render_kpi(label: str, value: str, subtext: str = "", card_theme: str = "primary"):
    """Hien thi the chi so KPI."""
    border_class = {
        "primary": "metric-card-primary",
        "red": "metric-card-red",
        "accent": "metric-card-accent",
        "green": "metric-card-green",
    }.get(card_theme, "metric-card-primary")

    html = (
        f'<div class="metric-card {border_class}">'
        f'<div class="metric-card-header"><span class="metric-label">{label}</span></div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-subtext">{subtext}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_chart_grid(df: pd.DataFrame, key_prefix: str, is_realtime: bool = False):
    """Luoi bieu do 2 cot (Trai: Cong suat & Dien ap | Phai: Donut & Heatmap/Bieu do 24h)."""
    col_l, col_r = st.columns([1.5, 1.1])
    with col_l:
        st.plotly_chart(build_power_line_chart(df), key=f"{key_prefix}_power_chart")
        st.plotly_chart(build_voltage_line_chart(df), key=f"{key_prefix}_voltage_chart")
    with col_r:
        st.plotly_chart(build_anomaly_donut_chart(df), key=f"{key_prefix}_donut_chart")
        if is_realtime:
            st.plotly_chart(build_hourly_distribution_chart(df), key=f"{key_prefix}_hourly_chart")
        else:
            st.plotly_chart(build_anomaly_heatmap(df), key=f"{key_prefix}_heatmap_chart")


def format_anomaly_table_data(anom_df: pd.DataFrame) -> pd.DataFrame:
    """Chuan hoa DataFrame bat thuong de dua vao bang hien thi."""
    df_out = anom_df.copy()
    df_out["Thời gian"] = df_out.index.strftime("%Y-%m-%d %H:%M")
    df_out["Công suất (kW)"] = df_out[TARGET_COL].map(lambda x: f"{x:.3f}")
    df_out["Điện áp (V)"] = df_out["Voltage"].map(lambda x: f"{x:.1f}")
    df_out["Mức độ (Severity)"] = df_out["severity"].map(lambda x: f"{x*100:.1f}%")
    df_out["Phân loại lỗi (AI)"] = df_out["anomaly_type"].map(lambda x: TYPE_LABELS.get(x, x))
    df_out["Nguyên nhân chính (XAI)"] = df_out["explanation"]
    return df_out[TABLE_COLS]


def render_anomaly_table(df_anom: pd.DataFrame, height: int = 300):
    """Bang HTML/CSS hien thi danh sach su co bat thuong."""
    if df_anom.empty:
        st.info("Không phát hiện điểm bất thường nào trong khoảng thời gian này.")
        return

    rows_html = []
    for _, row in df_anom.iterrows():
        ts = row.get("Thời gian", "—")
        power = row.get("Công suất (kW)", "—")
        voltage = row.get("Điện áp (V)", "—")
        severity = row.get("Mức độ (Severity)", "—")
        anomaly_type = row.get("Phân loại lỗi (AI)", "—")
        explanation = row.get("Nguyên nhân chính (XAI)", "—")

        # Badge pill muc do nghiem trong
        try:
            sev_num = float(str(severity).replace("%", ""))
            if sev_num >= 60.0:
                sev_badge = f'<span class="badge-pill badge-red">{severity}</span>'
            elif sev_num >= 30.0:
                sev_badge = f'<span class="badge-pill badge-amber">{severity}</span>'
            else:
                sev_badge = f'<span class="badge-pill badge-blue">{severity}</span>'
        except Exception:
            sev_badge = f'<span class="badge-pill badge-neutral">{severity}</span>'

        # Badge pill loai loi
        type_str = str(anomaly_type).lower()
        if "công suất" in type_str:
            type_badge = f'<span class="badge-pill badge-amber">{anomaly_type}</span>'
        elif "sụt" in type_str or "điện áp" in type_str:
            type_badge = f'<span class="badge-pill badge-red">{anomaly_type}</span>'
        elif "đêm" in type_str:
            type_badge = f'<span class="badge-pill badge-blue">{anomaly_type}</span>'
        else:
            type_badge = f'<span class="badge-pill badge-neutral">{anomaly_type}</span>'

        rows_html.append(
            f'<tr><td class="col-time">{ts}</td><td class="col-num"><b>{power}</b></td><td class="col-num">{voltage}</td><td class="col-center">{sev_badge}</td><td class="col-center">{type_badge}</td><td class="col-xai">{explanation}</td></tr>'
        )

    tbody = "".join(rows_html)
    html_code = (
        f'<div class="custom-table-container" style="max-height:{height}px;">'
        f'<table class="custom-table">'
        f'<thead><tr><th style="width:16%;">Thời gian</th><th style="width:13%;">Công suất (kW)</th><th style="width:12%;">Điện áp (V)</th><th style="width:15%; text-align:center;">Mức độ (Severity)</th><th style="width:18%; text-align:center;">Phân loại lỗi (AI)</th><th style="width:26%;">Nguyên nhân chính (XAI)</th></tr></thead>'
        f'<tbody>{tbody}</tbody>'
        f'</table></div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


# 4. Che do phan tich lich su

def render_history_view(bundle, df_demo: pd.DataFrame):
    """Man hinh phan tich du lieu lich su theo khoang thoi gian."""
    if df_demo.empty:
        st.warning("Chưa tìm thấy tập dữ liệu demo. Hãy chạy lệnh python src/01_data_prep.py rồi python src/02_train.py để tạo dữ liệu!")
        return

    model, scaler, feat_names = bundle["model"], bundle["scaler"], bundle["features"]
    medians, iqrs = bundle["medians"], bundle["iqrs"]

    # 1. Bo loc ngay thang
    min_date, max_date = df_demo.index.min().date(), df_demo.index.max().date()
    c1, c2, c3 = st.columns([1.5, 1.5, 1.0])
    with c1:
        start_date = st.date_input("Từ ngày:", min_value=min_date, max_value=max_date, value=min_date, key="hist_start")
    with c2:
        end_date = st.date_input("Đến ngày:", min_value=min_date, max_value=max_date, value=max_date, key="hist_end")
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        reset_filter = st.button("Xem toàn bộ", key="btn_reset_dates")

    if reset_filter:
        start_date, end_date = min_date, max_date

    mask = (df_demo.index.date >= start_date) & (df_demo.index.date <= end_date)
    df_filtered = df_demo[mask].copy()

    if df_filtered.empty:
        st.info("Không có bản ghi nào trong khoảng thời gian đã chọn.")
        return

    # 2. Suy luan va gan nhan
    df_feat = extract_features(df_filtered)
    X = scaler.transform(df_feat[feat_names].values)
    scores = model.decision_function(X)

    df_eval = df_filtered.loc[df_feat.index].copy()
    df_eval["raw_score"] = scores
    df_eval["severity"] = [calc_severity(s) for s in scores]
    df_eval["severity_level"] = [get_severity_level(s) for s in df_eval["severity"]]
    df_eval["is_anomaly"] = (df_eval["raw_score"] < 0).astype(int)

    types, expls = [], []
    for is_anom, (_, row) in zip(df_eval["is_anomaly"], df_feat.iterrows()):
        if is_anom:
            types.append(classify_type(row))
            expls.append(explain_anomaly(row, medians, iqrs))
        else:
            types.append("normal")
            expls.append("—")
    df_eval["anomaly_type"] = types
    df_eval["explanation"] = expls

    # 3. The chi so KPI
    total_pts = len(df_eval)
    anom_pts = int(df_eval["is_anomaly"].sum())
    anom_rate = (anom_pts / total_pts * 100) if total_pts > 0 else 0
    avg_power = df_eval[TARGET_COL].mean()
    peak_hour, peak_count, _ = calc_peak_hour_stats(df_eval)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Tổng Số Mẫu", f"{total_pts:,}", f"Từ {start_date} đến {end_date}", "primary")
    with k2:
        render_kpi("Tổng Bất Thường", f"{anom_pts:,}", f"Tỷ lệ: {anom_rate:.1f}% tổng tải", "red")
    with k3:
        render_kpi("Khung Giờ Đỉnh Lỗi", f"{peak_hour:02d}:00", f"{peak_count} sự cố phát hiện", "accent")
    with k4:
        render_kpi("Công Suất Trung Bình", f"{avg_power:.3f} kW", f"Cao nhất: {df_eval[TARGET_COL].max():.2f} kW", "green")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # 4. Luoi bieu do
    render_chart_grid(df_eval, key_prefix="hist", is_realtime=False)

    # 5. Bang danh sach su co bat thuong
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    anom_logs = df_eval[df_eval["is_anomaly"] == 1].copy()

    st.markdown(
        f'<div class="anomaly-table-header">'
        f'<h4>Danh sách chi tiết các điểm bất thường phát hiện</h4>'
        f'<span class="table-count">{len(anom_logs):,} điểm sự cố</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if not anom_logs.empty:
        display_df = format_anomaly_table_data(anom_logs[::-1])
        render_anomaly_table(display_df, height=350)
    else:
        st.info("Không phát hiện điểm bất thường nào trong khoảng thời gian này.")


# 5. Dong co va giao dien giam sat thoi gian thuc (Real-Time)

def _step_stream_engine(df_demo: pd.DataFrame, bundle, n_steps: int = 1):
    """Nap n_steps ban ghi tu df_demo vao session state."""
    model, scaler, feat_names = bundle["model"], bundle["scaler"], bundle["features"]
    medians, iqrs = bundle["medians"], bundle["iqrs"]

    cursor = st.session_state.rt_cursor
    total_len = len(df_demo)

    for _ in range(n_steps):
        if cursor >= total_len:
            st.session_state.rt_is_playing = False
            break

        row = df_demo.iloc[cursor]
        dt = df_demo.index[cursor]
        cursor += 1

        # Buffer trich xuat dac trung
        st.session_state.rt_buffer.append({"datetime": dt, **row.to_dict()})
        if len(st.session_state.rt_buffer) > 50:
            st.session_state.rt_buffer.pop(0)

        # Suy luan diem moi nhat
        is_anom, severity, anom_type, expl = 0, 0.0, "normal", "—"
        if len(st.session_state.rt_buffer) >= 25:
            buf_df = pd.DataFrame(st.session_state.rt_buffer).set_index("datetime")
            latest_feat = extract_latest(buf_df)
            if latest_feat is not None:
                X = scaler.transform(latest_feat[feat_names].values.reshape(1, -1))
                raw_score = float(model.decision_function(X)[0])
                severity = calc_severity(raw_score)
                is_anom = 1 if raw_score < 0 else 0
                if is_anom:
                    anom_type = classify_type(latest_feat)
                    expl = explain_anomaly(latest_feat, medians, iqrs)

        # Cap nhat sliding window bieu do
        record = {
            "datetime": dt,
            TARGET_COL: float(row.get(TARGET_COL, 0.0)),
            "Voltage": float(row.get("Voltage", 0.0)),
            "severity": severity,
            "is_anomaly": is_anom,
            "anomaly_type": anom_type,
            "explanation": expl,
        }
        st.session_state.rt_display.append(record)
        if len(st.session_state.rt_display) > MAX_DISPLAY_POINTS:
            st.session_state.rt_display.pop(0)

        # Ghi nhan vao bang su co (chong trung lap)
        ts_key = dt.strftime("%Y-%m-%d %H:%M:%S")
        if is_anom == 1 and ts_key not in st.session_state.rt_event_keys:
            st.session_state.rt_event_keys.add(ts_key)
            st.session_state.rt_anomaly_events.append({
                "Thời gian": ts_key,
                "Công suất (kW)": f"{record[TARGET_COL]:.3f}",
                "Điện áp (V)": f"{record['Voltage']:.1f}",
                "Mức độ (Severity)": f"{severity*100:.1f}%",
                "Phân loại lỗi (AI)": TYPE_LABELS.get(anom_type, anom_type),
                "Nguyên nhân chính (XAI)": expl,
            })

    st.session_state.rt_cursor = cursor


def _reset_stream(df_demo: pd.DataFrame, bundle):
    """Dat lai trang thai phat luong ve diem bat dau."""
    st.session_state.rt_is_playing = False
    st.session_state.rt_cursor = 0
    st.session_state.rt_buffer = []
    st.session_state.rt_display = []
    st.session_state.rt_anomaly_events = []
    st.session_state.rt_event_keys = set()
    _step_stream_engine(df_demo, bundle, n_steps=30)


def render_realtime_view(bundle, df_demo: pd.DataFrame):
    """Man hinh giam sat luong du lieu thoi gian thuc."""
    if df_demo.empty:
        st.warning("Chưa tìm thấy tập dữ liệu demo. Hãy chạy python src/01_data_prep.py rồi python src/02_train.py trước!")
        return

    defaults = {
        "rt_is_playing": False,
        "rt_cursor": 0,
        "rt_buffer": [],
        "rt_display": [],
        "rt_anomaly_events": [],
        "rt_event_keys": set(),
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

    if len(st.session_state.rt_display) == 0 and len(df_demo) >= 30:
        _step_stream_engine(df_demo, bundle, n_steps=30)

    # 1. Thanh cong cu dieu khien
    st.markdown("##### Bảng điều khiển phát luồng trực tiếp")
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5 = st.columns([1.3, 1.2, 1.2, 1.2, 1.8])
    is_playing = st.session_state.rt_is_playing

    with ctrl_c1:
        if not is_playing:
            if st.button("Play (Tiếp tục)", type="primary", key="btn_play"):
                st.session_state.rt_is_playing = True
                st.rerun()
        else:
            if st.button("Pause (Tạm dừng)", type="secondary", key="btn_pause"):
                st.session_state.rt_is_playing = False
                st.rerun()

    with ctrl_c2:
        if st.button("Bước tiếp +1", key="btn_step_next"):
            st.session_state.rt_is_playing = False
            _step_stream_engine(df_demo, bundle, n_steps=1)
            st.rerun()

    with ctrl_c3:
        if st.button("Khởi động lại", key="btn_reset_stream"):
            _reset_stream(df_demo, bundle)
            st.success("Đã reset về điểm khởi đầu!")
            st.rerun()

    with ctrl_c4:
        speed_option = st.selectbox(
            "Tốc độ phát:",
            options=[0.1, 0.5, 1.0, 2.0],
            index=2,
            format_func=lambda x: f"{x}s / mẫu",
            key="stream_speed_select"
        )

    with ctrl_c5:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        cur, total = st.session_state.rt_cursor, len(df_demo)
        if is_playing:
            st.markdown(f'<div class="stream-status-playing">ĐANG PHÁT LUỒNG (Vị trí: {cur:,}/{total:,})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="stream-status-paused">TẠM DỪNG (Vị trí: {cur:,}/{total:,})</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # 2. Nap ban ghi moi khi dang Play
    if is_playing:
        _step_stream_engine(df_demo, bundle, n_steps=1)

    disp_df = pd.DataFrame(st.session_state.rt_display).set_index("datetime") if st.session_state.rt_display else pd.DataFrame()

    # 3. The KPI thoi gian thuc
    total_streamed = st.session_state.rt_cursor
    total_anoms = len(st.session_state.rt_anomaly_events)
    anom_rate = (total_anoms / max(1, total_streamed) * 100)
    latest_power = float(disp_df[TARGET_COL].iloc[-1]) if (not disp_df.empty and TARGET_COL in disp_df.columns) else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Số Mẫu Đã Phát", f"{total_streamed:,}", f"Tiến độ: {total_streamed:,}/{len(df_demo):,} mẫu", "primary")
    with k2:
        render_kpi("Bất Thường Đã Bắt", f"{total_anoms:,}", f"Tỷ lệ: {anom_rate:.1f}% luồng", "red")
    with k3:
        render_kpi("Công Suất Hiện Tại", f"{latest_power:.3f} kW", "Tải tiêu thụ tức thời", "green")
    with k4:
        render_kpi("Trạng Thái Stream", "PLAYING" if is_playing else "PAUSED", f"Tốc độ: {speed_option}s / mẫu", "accent" if is_playing else "primary")

    # 4. Thanh thong bao trang thai tuc thoi
    if not disp_df.empty and "is_anomaly" in disp_df.columns and disp_df["is_anomaly"].iloc[-1] == 1:
        last = disp_df.iloc[-1]
        type_str = TYPE_LABELS.get(last['anomaly_type'], last['anomaly_type']).upper()
        alert_html = (
            f'<div class="alert-bar-danger">'
            f'<div><span class="alert-tag">CẢNH BÁO: {type_str}</span>'
            f'<span>Thời gian: <b>{disp_df.index[-1].strftime("%H:%M:%S")}</b> | Công suất: <b>{last[TARGET_COL]:.3f} kW</b> | Điện áp: <b>{last["Voltage"]:.1f}V</b> | Mức độ: <b>{last["severity"]*100:.1f}%</b></span></div>'
            f'<div class="alert-xai">{last["explanation"]}</div>'
            f'</div>'
        )
        st.markdown(alert_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="alert-bar-success"><span class="alert-tag">TRẠNG THÁI LƯỚI ĐIỆN:</span> Vận hành bình thường — Điện áp và công suất tiêu thụ trong ngưỡng an toàn ổn định.</div>',
            unsafe_allow_html=True
        )

    # 5. Luoi bieu do
    render_chart_grid(disp_df, key_prefix="rt", is_realtime=True)

    # 6. Bang nhat ky su co bat thuong
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="anomaly-table-header">'
        f'<h4>Bảng nhật ký sự cố bất thường (Chỉ ghi nhận lỗi)</h4>'
        f'<span class="table-count">{total_anoms:,} sự cố ghi nhận</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.session_state.rt_anomaly_events:
        event_df = pd.DataFrame(st.session_state.rt_anomaly_events[::-1])
        render_anomaly_table(event_df[TABLE_COLS], height=280)
    else:
        st.info("Chưa ghi nhận sự cố bất thường nào trong phiên phát luồng này.")

    # 7. Tu dong lap lai khi Play
    if is_playing:
        time.sleep(speed_option)
        st.rerun()


# 6. Diem khoi chay chinh cua ung dung

def main():
    """Ham dieu phoi ung dung chinh."""
    st.set_page_config(page_title="Smart Meter Anomaly Detection", layout="wide")
    load_custom_css()

    st.markdown('<div class="dashboard-main-title">Smart Meter Anomaly Detection System</div>', unsafe_allow_html=True)

    bundle = load_bundle()
    if bundle is None:
        st.error("Chưa tìm thấy mô hình. Hãy chạy lệnh python src/01_data_prep.py rồi python src/02_train.py để huấn luyện!")
        return

    df_demo = load_historical_data()

    mode = st.radio(
        "Chế độ phân tích:",
        ["Lịch sử (Historical Analysis)", "Thời gian thực (Real-Time Monitoring)"],
        horizontal=True,
        key="main_mode_select"
    )

    if "Lịch sử" in mode:
        render_history_view(bundle, df_demo)
    else:
        render_realtime_view(bundle, df_demo)


if __name__ == "__main__":
    main()
