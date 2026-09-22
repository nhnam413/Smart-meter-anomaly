

import os
import sqlite3
import time
from contextlib import closing
from datetime import date, datetime
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    ANOMALY_TYPE_COLORS,
    COLORS,
    CSS_FILE,
    DATE_FORMAT,
    DATETIME_FORMAT,
    DATETIME_MINUTE_FORMAT,
    DEMO_STREAM_PATH,
    MAX_DISPLAY_POINTS,
    MODEL_BUNDLE_PATH,
    TARGET_COL,
    TYPE_LABELS,
)
from features import (
    calc_severity,
    classify_type,
    explain_anomaly,
    extract_features,
    extract_latest,
    get_severity_level,
)

CHART_THEME = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, sans-serif", color="#0F172A", size=11),
    margin=dict(l=40, r=20, t=35, b=35),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
)

TABLE_COLS = [
    "Thời gian",
    "Công suất (kW)",
    "Điện áp (V)",
    "Điểm cảnh báo",
    "Dạng gợi ý",
    "Dấu hiệu nổi bật",
]

ALERT_DB_PATH = os.path.join(os.path.dirname(DEMO_STREAM_PATH), "alert_history.sqlite3")
STATUS_LABELS = {"new": "Mới", "acknowledged": "Đã tiếp nhận", "closed": "Đã đóng"}
ALERT_TYPE_VALUES = [key for key in TYPE_LABELS if key not in {"normal", "unknown"}]
STREAM_STATE_VERSION = 2
ALERT_WIDGET_KEYS = ("alert_status_filter", "alert_type_filter", "selected_alert_id")


# Khởi tạo bảng lưu cảnh báo nếu chưa tồn tại.
def init_alert_store() -> None:
    with closing(sqlite3.connect(ALERT_DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY, data_time TEXT NOT NULL,
                power REAL NOT NULL, voltage REAL NOT NULL,
                severity REAL NOT NULL, severity_level TEXT NOT NULL,
                anomaly_type TEXT NOT NULL, explanation TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'new', note TEXT NOT NULL DEFAULT '',
                acknowledged_at TEXT, closed_at TEXT, updated_at TEXT NOT NULL
            )
        """)
        conn.commit()


# Lưu mới hoặc cập nhật một cảnh báo.
def save_alert(event: dict) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(ALERT_DB_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO alerts (
                alert_id, data_time, power, voltage, severity, severity_level,
                anomaly_type, explanation, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alert_id) DO UPDATE SET
                power=excluded.power, voltage=excluded.voltage,
                severity=excluded.severity, severity_level=excluded.severity_level,
                anomaly_type=excluded.anomaly_type, explanation=excluded.explanation,
                updated_at=excluded.updated_at
        """,
            (
                event["alert_id"],
                event["data_time"],
                event["power"],
                event["voltage"],
                event["severity"],
                event["severity_level"],
                event["anomaly_type"],
                event["explanation"],
                now,
            ),
        )
        conn.commit()


# Đọc toàn bộ cảnh báo hoặc một nhóm theo mã.
def load_alerts(alert_ids: list[str] | None = None) -> pd.DataFrame:
    with closing(sqlite3.connect(ALERT_DB_PATH)) as conn:
        if alert_ids is None:
            return pd.read_sql_query(
                "SELECT * FROM alerts ORDER BY data_time DESC", conn
            )

        unique_ids = list(dict.fromkeys(alert_ids))
        if not unique_ids:
            return pd.read_sql_query(
                "SELECT * FROM alerts WHERE 0 ORDER BY data_time DESC", conn
            )

        placeholders = ",".join("?" for _ in unique_ids)
        return pd.read_sql_query(
            f"SELECT * FROM alerts WHERE alert_id IN ({placeholders}) ORDER BY data_time DESC",
            conn,
            params=unique_ids,
        )


# Cập nhật trạng thái và ghi chú xử lý cảnh báo.
def update_alert(alert_id: str, status: str, note: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    acknowledged_at = now if status == "acknowledged" else None
    closed_at = now if status == "closed" else None
    with closing(sqlite3.connect(ALERT_DB_PATH)) as conn:
        conn.execute(
            """
            UPDATE alerts SET status=?, note=?,
                acknowledged_at=COALESCE(?, acknowledged_at),
                closed_at=COALESCE(?, closed_at), updated_at=?
            WHERE alert_id=?
        """,
            (status, note, acknowledged_at, closed_at, now, alert_id),
        )
        conn.commit()


# Tải model bundle đã huấn luyện.
@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_BUNDLE_PATH):
        return None
    return joblib.load(MODEL_BUNDLE_PATH)


# Tải tập dữ liệu Demo.
@st.cache_data
def load_historical_data():
    if not os.path.exists(DEMO_STREAM_PATH):
        return pd.DataFrame()
    return pd.read_csv(DEMO_STREAM_PATH, index_col="datetime", parse_dates=True)


# Nạp CSS giao diện từ tệp cấu hình.
def load_custom_css() -> None:
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Tạo biểu đồ công suất và đánh dấu điểm bất thường.
def build_power_line_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or TARGET_COL not in df.columns:
        fig.update_layout(**CHART_THEME, title="Chưa có dữ liệu")
        return fig

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[TARGET_COL],
            mode="lines",
            name="Công suất (kW)",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor=COLORS["primary_rgba_05"],
            hovertemplate="<b>%{x}</b><br>Công suất: %{y:.3f} kW<extra></extra>",
        )
    )

    if "is_anomaly" in df.columns:
        anom_df = df[df["is_anomaly"] == 1]
        if not anom_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_df.index,
                    y=anom_df[TARGET_COL],
                    mode="markers",
                    name="Bất thường",
                    marker=dict(
                        color=COLORS["danger"],
                        size=8.5,
                        symbol="circle",
                        line=dict(width=1.5, color=COLORS["surface"]),
                    ),
                    customdata=anom_df.get("anomaly_type", "Bất thường"),
                    hovertemplate="<b>BẤT THƯỜNG</b><br>Thời gian: %{x}<br>Công suất: %{y:.3f} kW<br>Dạng lỗi: %{customdata}<extra></extra>",
                )
            )

    fig.update_layout(
        **CHART_THEME,
        height=310,
        title=dict(
            text="Biểu đồ đường công suất tiêu thụ điện năng (kW)",
            font=dict(size=13, color=COLORS["text_primary"]),
            x=0,
        ),
        xaxis=dict(
            title=dict(
                text="Thời gian", font=dict(size=10, color=COLORS["text_primary"])
            ),
            showgrid=True,
            gridcolor=COLORS["grid_line"],
            hoverformat="%d/%m/%Y %H:%M",
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            title=dict(
                text="Global Active Power (kW)",
                font=dict(size=10, color=COLORS["text_primary"]),
            ),
            showgrid=True,
            gridcolor=COLORS["grid_line"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# Tạo biểu đồ điện áp cùng dải tham chiếu.
def build_voltage_line_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or "Voltage" not in df.columns:
        fig.update_layout(**CHART_THEME, title="Chưa có dữ liệu")
        return fig

    fig.add_hrect(
        y0=220,
        y1=250,
        fillcolor=COLORS["success_rgba_08"],
        line_width=0,
        annotation_text="Dải tham chiếu (220–250V)",
        annotation_position="top left",
        annotation_font=dict(color=COLORS["success_dark"], size=10),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Voltage"],
            mode="lines",
            name="Điện áp (V)",
            line=dict(color=COLORS["primary_dark"], width=1.8),
            hovertemplate="<b>%{x}</b><br>Điện áp: %{y:.1f} V<extra></extra>",
        )
    )

    if "is_anomaly" in df.columns:
        anom_df = df[
            (df["is_anomaly"] == 1) & (df.get("anomaly_type", "") == "voltage_drop")
        ]
        if not anom_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_df.index,
                    y=anom_df["Voltage"],
                    mode="markers",
                    name="Sự cố điện áp",
                    marker=dict(
                        color=COLORS["danger"],
                        size=8.0,
                        symbol="circle",
                        line=dict(width=1.5, color=COLORS["surface"]),
                    ),
                    hovertemplate="<b>SỰ CỐ ĐIỆN ÁP</b><br>Thời gian: %{x}<br>Điện áp: %{y:.1f} V<extra></extra>",
                )
            )

    fig.update_layout(
        **CHART_THEME,
        height=270,
        title=dict(
            text="Biểu đồ đường điện áp lưới điện (Voltage)",
            font=dict(size=13, color=COLORS["text_primary"]),
            x=0,
        ),
        xaxis=dict(
            title=dict(
                text="Thời gian", font=dict(size=10, color=COLORS["text_primary"])
            ),
            showgrid=True,
            gridcolor=COLORS["grid_line"],
            hoverformat="%d/%m/%Y %H:%M",
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            title=dict(
                text="Voltage (V)", font=dict(size=10, color=COLORS["text_primary"])
            ),
            showgrid=True,
            gridcolor=COLORS["grid_line"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# Tạo biểu đồ tỷ lệ các dạng bất thường.
def build_anomaly_donut_chart(df: pd.DataFrame) -> go.Figure:
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
        fig.add_trace(
            go.Pie(
                labels=["Bình thường"],
                values=[1],
                hole=0.58,
                marker=dict(colors=[COLORS["border"]]),
                textinfo="none",
                hoverinfo="none",
            )
        )
        center_text = f"<b>0</b><br><span style='font-size:10px; color:{COLORS['text_muted']};'>sự cố</span>"
    else:
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.58,
                marker=dict(
                    colors=slice_colors, line=dict(color=COLORS["surface"], width=2)
                ),
                textinfo="percent",
                textposition="inside",
                textfont=dict(
                    size=11, color=COLORS["surface"], family="Inter, sans-serif"
                ),
                hovertemplate="<b>%{label}</b><br>Số lượng: %{value:,} vụ<br>Tỷ lệ: %{percent}<extra></extra>",
            )
        )
        center_text = f"<b>{total_anom:,}</b><br><span style='font-size:10px; color:{COLORS['text_muted']};'>sự cố</span>"

    fig.update_layout(
        **{**CHART_THEME, "margin": dict(l=20, r=20, t=35, b=45)},
        height=310,
        title=dict(
            text="Tỷ lệ phân bố các loại bất thường",
            font=dict(size=13, color=COLORS["text_primary"]),
            x=0,
        ),
        annotations=[
            dict(
                text=center_text,
                x=0.5,
                y=0.5,
                font_size=15,
                font_color=COLORS["text_primary"],
                showarrow=False,
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
    )
    return fig


# Tạo bản đồ nhiệt bất thường theo giờ và thứ.
def build_anomaly_heatmap(df: pd.DataFrame) -> go.Figure:
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

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            x=day_labels,
            y=hours,
            colorscale=heatmap_colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Số lỗi", font=dict(size=10, color=COLORS["text_primary"])
                ),
                thickness=10,
                len=0.85,
                tickfont=dict(size=9, color=COLORS["text_secondary"]),
                outlinewidth=0,
            ),
            hovertemplate="<b>%{x}</b> lúc <b>%{y}</b><br>Số điểm bất thường: <b>%{z}</b><extra></extra>",
        )
    )

    fig.update_layout(
        **{**CHART_THEME, "margin": dict(l=45, r=20, t=35, b=25)},
        height=270,
        title=dict(
            text="Bản đồ nhiệt bất thường (24 giờ × 7 ngày)",
            font=dict(size=13, color=COLORS["text_primary"]),
            x=0,
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=10, color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            dtick=3,
            tickfont=dict(size=10, color=COLORS["text_secondary"]),
            autorange="reversed",
        ),
    )
    return fig


# Thống kê số bất thường theo 24 giờ.
def calc_peak_hour_stats(df: pd.DataFrame) -> tuple[int, int, list[int]]:
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


# Tạo biểu đồ phân bố bất thường theo giờ.
def build_hourly_distribution_chart(df: pd.DataFrame) -> go.Figure:
    hours = list(range(24))
    peak_hour, peak_count, hour_counts = calc_peak_hour_stats(df)
    bar_colors = [
        COLORS["danger"] if h == peak_hour and peak_count > 0 else COLORS["primary"]
        for h in hours
    ]

    fig = go.Figure(
        go.Bar(
            x=[f"{h:02d}h" for h in hours],
            y=hour_counts,
            marker=dict(color=bar_colors, opacity=0.85),
            hovertemplate="<b>Khung giờ: %{x}</b><br>Số lượng bất thường: %{y} điểm<extra></extra>",
        )
    )
    fig.update_layout(
        **CHART_THEME,
        height=270,
        title=dict(
            text="Phân bố bất thường theo 24 giờ (00:00 - 23:00)",
            font=dict(size=13, color=COLORS["text_primary"]),
            x=0,
        ),
        xaxis=dict(
            title=dict(
                text="Khung giờ", font=dict(size=10, color=COLORS["text_primary"])
            ),
            showgrid=False,
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            title=dict(
                text="Số điểm bất thường",
                font=dict(size=10, color=COLORS["text_primary"]),
            ),
            showgrid=True,
            gridcolor=COLORS["grid_line"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
    )
    return fig


# Hiển thị một thẻ KPI.
def render_kpi(
    label: str, value: str, subtext: str = "", card_theme: str = "primary"
) -> None:
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
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# Hiển thị lưới biểu đồ theo chế độ.
def render_chart_grid(
    df: pd.DataFrame, key_prefix: str, is_realtime: bool = False
) -> None:
    col_l, col_r = st.columns(2 if is_realtime else [1.5, 1.1])
    with col_l:
        st.plotly_chart(
            build_power_line_chart(df), theme=None, key=f"{key_prefix}_power_chart"
        )
    with col_r:
        if is_realtime:
            st.plotly_chart(
                build_voltage_line_chart(df),
                theme=None,
                key=f"{key_prefix}_voltage_chart",
            )
        else:
            st.plotly_chart(
                build_anomaly_donut_chart(df),
                theme=None,
                key=f"{key_prefix}_donut_chart",
            )
    if not is_realtime:
        col_l, col_r = st.columns([1.5, 1.1])
        with col_l:
            st.plotly_chart(
                build_voltage_line_chart(df),
                theme=None,
                key=f"{key_prefix}_voltage_chart",
            )
        with col_r:
            st.plotly_chart(
                build_anomaly_heatmap(df), theme=None, key=f"{key_prefix}_heatmap_chart"
            )


# Định dạng dữ liệu cho bảng bất thường.
def format_anomaly_table_data(anom_df: pd.DataFrame) -> pd.DataFrame:
    df_out = anom_df.copy()
    df_out["Thời gian"] = df_out.index.strftime(DATETIME_MINUTE_FORMAT)
    df_out["Công suất (kW)"] = df_out[TARGET_COL].map(lambda x: f"{x:.3f}")
    df_out["Điện áp (V)"] = df_out["Voltage"].map(lambda x: f"{x:.1f}")
    df_out["Điểm cảnh báo"] = df_out["severity"].map(lambda x: f"{x * 100:.1f}/100")
    df_out["Dạng gợi ý"] = df_out["anomaly_type"].map(lambda x: TYPE_LABELS.get(x, x))
    df_out["Dấu hiệu nổi bật"] = df_out["explanation"]
    return df_out[TABLE_COLS]


# Hiển thị bảng bất thường dạng HTML.
def render_anomaly_table(df_anom: pd.DataFrame, height: int = 300) -> None:
    if df_anom.empty:
        st.info("Không phát hiện điểm bất thường nào trong khoảng thời gian này.")
        return

    rows_html = []
    for _, row in df_anom.iterrows():
        ts = row.get("Thời gian", "—")
        power = row.get("Công suất (kW)", "—")
        voltage = row.get("Điện áp (V)", "—")
        severity = row.get("Điểm cảnh báo", "—")
        anomaly_type = row.get("Dạng gợi ý", "—")
        explanation = row.get("Dấu hiệu nổi bật", "—")

        try:
            sev_num = float(str(severity).split("/")[0])
            if sev_num >= 70.0:
                sev_badge = f'<span class="badge-pill badge-red">{severity}</span>'
            elif sev_num >= 50.0:
                sev_badge = f'<span class="badge-pill badge-amber">{severity}</span>'
            else:
                sev_badge = f'<span class="badge-pill badge-blue">{severity}</span>'
        except Exception:
            sev_badge = f'<span class="badge-pill badge-neutral">{severity}</span>'

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
        f'<thead><tr><th style="width:16%;">Thời gian</th><th style="width:13%;">Công suất (kW)</th><th style="width:12%;">Điện áp (V)</th><th style="width:15%; text-align:center;">Điểm cảnh báo</th><th style="width:18%; text-align:center;">Dạng gợi ý</th><th style="width:26%;">Dấu hiệu nổi bật</th></tr></thead>'
        f"<tbody>{tbody}</tbody>"
        f"</table></div>"
    )
    st.markdown(html_code, unsafe_allow_html=True)


# Hiển thị bộ lọc ngày và trả về khoảng đã chọn.
def render_history_filters(df_demo: pd.DataFrame) -> tuple[date, date]:
    min_date, max_date = df_demo.index.min().date(), df_demo.index.max().date()
    c1, c2, c3 = st.columns([1.5, 1.5, 1.0])
    with c1:
        start_date = st.date_input(
            "Từ ngày:",
            min_value=min_date,
            max_value=max_date,
            value=min_date,
            format="DD/MM/YYYY",
            key="hist_start",
        )
    with c2:
        end_date = st.date_input(
            "Đến ngày:",
            min_value=min_date,
            max_value=max_date,
            value=max_date,
            format="DD/MM/YYYY",
            key="hist_end",
        )
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        reset_filter = st.button("Xem toàn bộ", key="btn_reset_dates")

    if reset_filter:
        start_date, end_date = min_date, max_date
    return start_date, end_date


# Suy luận toàn bộ chuỗi Demo trước khi áp dụng bộ lọc ngày.
def evaluate_history(bundle, df_demo: pd.DataFrame) -> pd.DataFrame:
    model, scaler, feat_names = bundle["model"], bundle["scaler"], bundle["features"]
    medians, iqrs = bundle["medians"], bundle["iqrs"]

    df_feat = extract_features(df_demo)
    if df_feat.empty:
        return pd.DataFrame()

    feature_matrix = scaler.transform(df_feat[feat_names].values)
    scores = model.decision_function(feature_matrix)

    df_eval = df_demo.loc[df_feat.index].copy()
    df_eval["raw_score"] = scores
    df_eval["severity"] = [calc_severity(score) for score in scores]
    df_eval["severity_level"] = [
        get_severity_level(severity) for severity in df_eval["severity"]
    ]
    df_eval["is_anomaly"] = (df_eval["raw_score"] < 0).astype(int)

    anomaly_types = []
    explanations = []
    for is_anomaly, (_, row) in zip(df_eval["is_anomaly"], df_feat.iterrows()):
        if is_anomaly:
            anomaly_types.append(classify_type(row))
            explanations.append(explain_anomaly(row, medians, iqrs))
        else:
            anomaly_types.append("normal")
            explanations.append("—")

    df_eval["anomaly_type"] = anomaly_types
    df_eval["explanation"] = explanations
    return df_eval


# Hiển thị các KPI của khoảng lịch sử đã chọn.
def render_history_kpis(
    df_eval: pd.DataFrame, start_date: date, end_date: date
) -> None:
    total_points = len(df_eval)
    anomaly_points = int(df_eval["is_anomaly"].sum())
    anomaly_rate = anomaly_points / total_points * 100 if total_points else 0
    average_power = df_eval[TARGET_COL].mean()
    peak_hour, peak_count, _ = calc_peak_hour_stats(df_eval)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi(
            "Tổng Số Mẫu",
            f"{total_points:,}",
            f"Từ {start_date.strftime(DATE_FORMAT)} đến {end_date.strftime(DATE_FORMAT)}",
            "primary",
        )
    with k2:
        render_kpi(
            "Tổng Bất Thường",
            f"{anomaly_points:,}",
            f"Tỷ lệ: {anomaly_rate:.1f}% mẫu đã đánh giá",
            "red",
        )
    with k3:
        render_kpi(
            "Khung Giờ Đỉnh Lỗi",
            f"{peak_hour:02d}:00",
            f"{peak_count} sự cố phát hiện",
            "accent",
        )
    with k4:
        render_kpi(
            "Công Suất Trung Bình",
            f"{average_power:.3f} kW",
            f"Cao nhất: {df_eval[TARGET_COL].max():.2f} kW",
            "green",
        )


# Hiển thị chế độ phân tích lịch sử.
def render_history_view(bundle, df_demo: pd.DataFrame) -> None:
    if df_demo.empty:
        st.warning(
            "Chưa tìm thấy tập dữ liệu demo. Hãy chạy lệnh python src/01_data_prep.py rồi python src/02_train.py!"
        )
        return

    start_date, end_date = render_history_filters(df_demo)

    display_mask = (df_demo.index.date >= start_date) & (df_demo.index.date <= end_date)
    if not display_mask.any():
        st.info("Không có bản ghi nào trong khoảng thời gian đã chọn.")
        return

    df_eval = evaluate_history(bundle, df_demo)
    if df_eval.empty:
        st.info("Chưa đủ dữ liệu lịch sử để trích xuất đặc trưng.")
        return
    df_eval = df_eval[
        (df_eval.index.date >= start_date) & (df_eval.index.date <= end_date)
    ]

    if df_eval.empty:
        st.info("Khoảng đã chọn chưa có mẫu nào đủ ngữ cảnh để đánh giá.")
        return

    render_history_kpis(df_eval, start_date, end_date)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    render_chart_grid(df_eval, key_prefix="hist", is_realtime=False)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    anom_logs = df_eval[df_eval["is_anomaly"] == 1].copy()

    st.markdown(
        f'<div class="anomaly-table-header">'
        f"<h4>Danh sách điểm bất thường được mô hình phát hiện</h4>"
        f'<span class="table-count">{len(anom_logs):,} điểm cảnh báo</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    if not anom_logs.empty:
        display_df = format_anomaly_table_data(anom_logs[::-1])
        render_anomaly_table(display_df, height=350)
    else:
        st.info("Không phát hiện điểm bất thường nào trong khoảng thời gian này.")


# Tạo phiên phát mới mà không xóa lịch sử SQLite.
def _reset_stream() -> None:
    st.session_state.rt_state_version = STREAM_STATE_VERSION
    st.session_state.rt_run_id = uuid4().hex[:8].upper()
    st.session_state.rt_is_playing = False
    st.session_state.rt_cursor = 0
    st.session_state.rt_buffer = []
    st.session_state.rt_display = []
    st.session_state.rt_anomaly_events = []
    st.session_state.rt_event_keys = set()
    for key in ALERT_WIDGET_KEYS:
        st.session_state.pop(key, None)


# Khởi tạo trạng thái của luồng mô phỏng.
def _init_stream_state() -> None:
    if st.session_state.get("rt_state_version") != STREAM_STATE_VERSION:
        _reset_stream()
        return

    defaults = {
        "rt_is_playing": False,
        "rt_cursor": 0,
        "rt_buffer": [],
        "rt_display": [],
        "rt_anomaly_events": [],
        "rt_event_keys": set(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("rt_run_id", uuid4().hex[:8].upper())


# Thêm một mẫu và giữ tối đa 50 mẫu trong bộ đệm.
def _append_stream_buffer(timestamp: pd.Timestamp, row: pd.Series) -> None:
    st.session_state.rt_buffer.append({"datetime": timestamp, **row.to_dict()})
    if len(st.session_state.rt_buffer) > 50:
        st.session_state.rt_buffer.pop(0)


# Suy luận mẫu cuối bộ đệm hoặc trả về trạng thái warm-up.
def _evaluate_latest_buffer(bundle) -> dict:
    result = {
        "severity": None,
        "raw_score": None,
        "evaluation_status": "warming_up",
        "is_anomaly": None,
        "anomaly_type": "unknown",
        "explanation": "—",
    }
    if len(st.session_state.rt_buffer) < 25:
        return result

    buffer_df = pd.DataFrame(st.session_state.rt_buffer).set_index("datetime")
    latest_features = extract_latest(buffer_df)
    if latest_features is None:
        return result

    model, scaler, feature_names = (
        bundle["model"],
        bundle["scaler"],
        bundle["features"],
    )
    feature_matrix = scaler.transform(
        latest_features[feature_names].values.reshape(1, -1)
    )
    raw_score = float(model.decision_function(feature_matrix)[0])
    is_anomaly = 1 if raw_score < 0 else 0

    result.update(
        severity=calc_severity(raw_score),
        raw_score=raw_score,
        evaluation_status="evaluated",
        is_anomaly=is_anomaly,
        anomaly_type="normal",
    )
    if is_anomaly:
        result["anomaly_type"] = classify_type(latest_features)
        result["explanation"] = explain_anomaly(
            latest_features, bundle["medians"], bundle["iqrs"]
        )
    return result


# Lưu kết quả suy luận vào danh sách hiển thị.
def _append_stream_record(
    timestamp: pd.Timestamp, row: pd.Series, evaluation: dict
) -> dict:
    record = {
        "datetime": timestamp,
        TARGET_COL: float(row.get(TARGET_COL, 0.0)),
        "Voltage": float(row.get("Voltage", 0.0)),
        **evaluation,
    }
    st.session_state.rt_display.append(record)
    if len(st.session_state.rt_display) > MAX_DISPLAY_POINTS:
        st.session_state.rt_display.pop(0)
    return record


# Tạo cảnh báo mới cho mẫu bất thường chưa được ghi nhận.
def _save_stream_alert(timestamp: pd.Timestamp, record: dict) -> None:
    timestamp_key = timestamp.strftime(DATETIME_FORMAT)
    if record["is_anomaly"] != 1 or timestamp_key in st.session_state.rt_event_keys:
        return

    st.session_state.rt_event_keys.add(timestamp_key)
    event = {
        "alert_id": (
            f"ALT-{st.session_state.rt_run_id}-{timestamp.strftime('%Y%m%d%H%M%S')}"
        ),
        "data_time": timestamp.isoformat(),
        "power": record[TARGET_COL],
        "voltage": record["Voltage"],
        "severity": record["severity"],
        "severity_level": get_severity_level(record["severity"]),
        "anomaly_type": record["anomaly_type"],
        "explanation": record["explanation"],
    }
    st.session_state.rt_anomaly_events.append(event)
    save_alert(event)


# Nạp và suy luận một số bản ghi tiếp theo.
def _step_stream_engine(df_demo: pd.DataFrame, bundle, n_steps: int = 1) -> None:
    cursor = st.session_state.rt_cursor
    total_len = len(df_demo)

    for _ in range(n_steps):
        if cursor >= total_len:
            st.session_state.rt_is_playing = False
            break

        row = df_demo.iloc[cursor]
        dt = df_demo.index[cursor]
        cursor += 1

        _append_stream_buffer(dt, row)
        evaluation = _evaluate_latest_buffer(bundle)
        record = _append_stream_record(dt, row, evaluation)
        _save_stream_alert(dt, record)

    st.session_state.rt_cursor = cursor


# Hiển thị hàng đợi và thao tác xử lý cảnh báo.
def render_alert_queue(alert_ids: list[str]) -> None:
    alerts = load_alerts(alert_ids)
    st.markdown("#### Cảnh báo cần xử lý")
    if alerts.empty:
        st.info("Chưa ghi nhận cảnh báo nào trong phiên hiện tại.")
        return

    f1, f2 = st.columns(2)
    with f1:
        status_filter = st.multiselect(
            "Trạng thái",
            list(STATUS_LABELS),
            default=["new", "acknowledged"],
            format_func=lambda x: STATUS_LABELS[x],
            key="alert_status_filter",
        )
    with f2:
        type_filter = st.multiselect(
            "Dạng gợi ý",
            ALERT_TYPE_VALUES,
            default=ALERT_TYPE_VALUES,
            format_func=lambda x: TYPE_LABELS.get(x, x),
            key="alert_type_filter",
        )

    filtered = alerts[
        alerts["status"].isin(status_filter) & alerts["anomaly_type"].isin(type_filter)
    ].copy()
    filtered["data_time_sort"] = pd.to_datetime(filtered["data_time"], errors="coerce")
    filtered = filtered.sort_values(
        ["data_time_sort", "severity"],
        ascending=[False, False],
        na_position="last",
    )

    if filtered.empty:
        st.info("Không có cảnh báo phù hợp với bộ lọc.")
        return

    table_df = pd.DataFrame(
        {
            "Mã cảnh báo": filtered["alert_id"],
            "Thời điểm dữ liệu": pd.to_datetime(filtered["data_time"]).dt.strftime(
                DATETIME_MINUTE_FORMAT
            ),
            "Trạng thái": filtered["status"].map(STATUS_LABELS),
            "Điểm cảnh báo": (filtered["severity"] * 100).round(1),
            "Dạng gợi ý": filtered["anomaly_type"].map(lambda x: TYPE_LABELS.get(x, x)),
            "Công suất (kW)": filtered["power"].round(3),
            "Điện áp (V)": filtered["voltage"].round(1),
            "Dấu hiệu nổi bật": filtered["explanation"],
        }
    )
    st.dataframe(
        table_df, width="stretch", hide_index=True, height=280, key="alert_queue_table"
    )
    st.download_button(
        "Tải CSV theo bộ lọc",
        table_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="bao_cao_canh_bao.csv",
        mime="text/csv",
        key="download_alerts",
    )

    selected_id = st.selectbox(
        "Chọn cảnh báo để xem và xử lý",
        filtered["alert_id"].tolist(),
        key="selected_alert_id",
    )
    selected = alerts.loc[alerts["alert_id"] == selected_id].iloc[0]
    st.markdown("##### Chi tiết cảnh báo")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Công suất", f"{selected['power']:.3f} kW")
    d2.metric("Điện áp", f"{selected['voltage']:.1f} V")
    d3.metric("Điểm quy đổi", f"{selected['severity'] * 100:.1f}/100")
    d4.metric("Điểm quyết định", "< 0 (bất thường)")
    st.caption(f"Dấu hiệu nổi bật: {selected['explanation']}")

    note = st.text_area(
        "Ghi chú xử lý", value=selected["note"] or "", key=f"note_{selected_id}"
    )
    a1, a2 = st.columns(2)
    with a1:
        if st.button(
            "Tiếp nhận cảnh báo",
            key=f"ack_{selected_id}",
            disabled=selected["status"] == "closed",
        ):
            update_alert(selected_id, "acknowledged", note)
            st.rerun()
    with a2:
        if st.button("Đóng cảnh báo", key=f"close_{selected_id}"):
            update_alert(selected_id, "closed", note)
            st.rerun()


# Hiển thị điều khiển luồng và trả về trạng thái phát.
def render_stream_controls(bundle, df_demo: pd.DataFrame) -> tuple[bool, float]:
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
            _reset_stream()
            st.success("Đã reset về điểm khởi đầu!")
            st.rerun()

    with ctrl_c4:
        speed_option = st.selectbox(
            "Tốc độ phát:",
            options=[0.1, 0.5, 1.0, 2.0],
            index=2,
            format_func=lambda x: f"{x}s / mẫu",
            key="stream_speed_select",
        )

    with ctrl_c5:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        cur, total = st.session_state.rt_cursor, len(df_demo)
        if is_playing:
            st.markdown(
                f'<div class="stream-status-playing">ĐANG PHÁT LUỒNG ({cur:,}/{total:,})</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="stream-status-paused">TẠM DỪNG ({cur:,}/{total:,})</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    return is_playing, speed_option


# Chuyển dữ liệu hiển thị trong session thành DataFrame.
def get_realtime_display() -> pd.DataFrame:
    if not st.session_state.rt_display:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.rt_display).set_index("datetime")


# Trả về mã cảnh báo của phiên và số cảnh báo đang mở.
def get_session_alert_summary() -> tuple[list[str], int]:
    alert_ids = [event["alert_id"] for event in st.session_state.rt_anomaly_events]
    alerts = load_alerts(alert_ids)
    if alerts.empty:
        return alert_ids, 0
    open_count = int(alerts["status"].isin(["new", "acknowledged"]).sum())
    return alert_ids, open_count


# Hiển thị KPI của phiên mô phỏng hiện tại.
def render_realtime_kpis(
    df_demo: pd.DataFrame, display_df: pd.DataFrame, open_alerts: int
) -> None:
    total_streamed = st.session_state.rt_cursor
    total_anomalies = len(st.session_state.rt_anomaly_events)
    evaluated_count = max(0, total_streamed - 24)
    anomaly_rate = total_anomalies / max(1, evaluated_count) * 100
    latest_power = (
        f"{float(display_df[TARGET_COL].iloc[-1]):.3f} kW"
        if not display_df.empty and TARGET_COL in display_df.columns
        else "—"
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi(
            "Mẫu Đã Đánh Giá",
            f"{evaluated_count:,}",
            f"Đã nhận {total_streamed:,}/{len(df_demo):,} mẫu",
            "primary",
        )
    with k2:
        render_kpi(
            "Cảnh Báo Trong Phiên",
            f"{total_anomalies:,}",
            f"Tỷ lệ: {anomaly_rate:.1f}% mẫu đã đánh giá",
            "red",
        )
    with k3:
        render_kpi("Công Suất Hiện Tại", latest_power, "Tải tiêu thụ tức thời", "green")
    with k4:
        render_kpi(
            "Cần Xử Lý",
            f"{open_alerts:,}",
            "Cảnh báo mới hoặc đã tiếp nhận",
            "accent" if open_alerts else "green",
        )


# Hiển thị trạng thái của mẫu mới nhất.
def render_latest_stream_status(display_df: pd.DataFrame) -> None:
    if display_df.empty:
        st.info(
            "Chưa nhận dữ liệu. Nhấn Play hoặc Bước tiếp +1 để bắt đầu luồng mô phỏng."
        )
        return
    if display_df["evaluation_status"].iloc[-1] == "warming_up":
        st.info(
            "Đang tích lũy đủ 24 mẫu lịch sử; mẫu mới nhất chưa được mô hình đánh giá."
        )
        return
    if display_df["is_anomaly"].iloc[-1] != 1:
        st.markdown(
            '<div class="alert-bar-success"><span class="alert-tag">KẾT QUẢ MẪU MỚI NHẤT:</span> Mô hình không đánh dấu bất thường.</div>',
            unsafe_allow_html=True,
        )
        return

    latest = display_df.iloc[-1]
    type_label = TYPE_LABELS.get(latest["anomaly_type"], latest["anomaly_type"]).upper()
    alert_html = (
        '<div class="alert-bar-danger">'
        f'<div><span class="alert-tag">CẢNH BÁO: {type_label}</span>'
        f"<span>Thời gian: <b>{display_df.index[-1].strftime('%H:%M:%S')}</b> | "
        f"Công suất: <b>{latest[TARGET_COL]:.3f} kW</b> | "
        f"Điện áp: <b>{latest['Voltage']:.1f}V</b> | "
        f"Mức độ: <b>{latest['severity'] * 100:.1f}%</b></span></div>"
        f'<div class="alert-xai">{latest["explanation"]}</div>'
        "</div>"
    )
    st.markdown(alert_html, unsafe_allow_html=True)


# Hiển thị chế độ mô phỏng giám sát tuần tự.
def render_realtime_view(bundle, df_demo: pd.DataFrame) -> None:
    if df_demo.empty:
        st.warning(
            "Chưa tìm thấy tập dữ liệu demo. Hãy chạy python src/01_data_prep.py rồi python src/02_train.py trước!"
        )
        return

    _init_stream_state()
    is_playing, speed_option = render_stream_controls(bundle, df_demo)

    if is_playing:
        _step_stream_engine(df_demo, bundle, n_steps=1)

    display_df = get_realtime_display()
    session_alert_ids, open_alerts = get_session_alert_summary()
    render_realtime_kpis(df_demo, display_df, open_alerts)
    render_latest_stream_status(display_df)
    render_chart_grid(display_df, key_prefix="rt", is_realtime=True)

    st.caption(
        f"Hai biểu đồ trên hiển thị tối đa {MAX_DISPLAY_POINTS} mẫu gần nhất; KPI tính trên toàn phiên mô phỏng."
    )
    render_alert_queue(session_alert_ids)

    if is_playing:
        time.sleep(speed_option)
        st.rerun()


# Nạp dữ liệu và hiển thị chế độ dashboard đã chọn.
def render_application() -> None:
    load_custom_css()
    init_alert_store()

    st.markdown(
        '<div class="dashboard-main-title">Smart Meter Anomaly Detection System</div>',
        unsafe_allow_html=True,
    )

    bundle = load_bundle()
    if bundle is None:
        st.error(
            "Chưa tìm thấy mô hình. Hãy chạy lệnh python src/01_data_prep.py rồi python src/02_train.py để huấn luyện!"
        )
    else:
        df_demo = load_historical_data()
        mode = st.radio(
            "Chế độ phân tích:",
            [
                "Lịch sử (Historical Analysis)",
                "Thời gian thực (Real-Time Monitoring)",
            ],
            horizontal=True,
            key="main_mode_select",
        )

        if "Lịch sử" in mode:
            render_history_view(bundle, df_demo)
        else:
            render_realtime_view(bundle, df_demo)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Smart Meter Anomaly Detection",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_application()
