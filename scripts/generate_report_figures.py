"""Generate reproducible report figures from the project's current data and model bundle."""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

import joblib
PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smart-meter-anomaly-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import ENGINEERED_FEATURE_NAMES  # noqa: E402
from features import extract_features  # noqa: E402
from report_diagrams import generate_diagrams  # noqa: E402


DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
BUNDLE_PATH = PROJECT_DIR / "models" / "model_bundle.pkl"
DPI = 300
COLORS = {
    "blue": "#2563EB",
    "orange": "#F59E0B",
    "red": "#DC2626",
    "green": "#059669",
    "purple": "#7C3AED",
    "slate": "#334155",
    "light": "#E2E8F0",
}
EXPECTED_VALID_TYPES = {
    "normal": 6265,
    "power_surge": 205,
    "voltage_drop": 205,
    "night_spike": 135,
}


def format_int_vi(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def format_float_vi(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#CBD5E1",
        "grid.alpha": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, filename: str, *, use_tight_layout: bool = True) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    train = pd.read_csv(DATA_DIR / "train_hourly.csv", index_col="datetime", parse_dates=True)
    demo = pd.read_csv(DATA_DIR / "demo_stream.csv", index_col="datetime", parse_dates=True)
    bundle = joblib.load(BUNDLE_PATH)
    demo_features = extract_features(demo)
    matrix = bundle["scaler"].transform(demo_features[bundle["features"]].to_numpy())
    decision = bundle["model"].decision_function(matrix)
    predicted = (decision < 0).astype(int)
    labels = demo.loc[demo_features.index, "is_anomaly"].astype(int)
    types = demo.loc[demo_features.index, "anomaly_type"]
    return train, demo, bundle, demo_features, decision, predicted, labels, types


def validate_inputs(
    train: pd.DataFrame,
    demo: pd.DataFrame,
    bundle: dict,
    train_features: pd.DataFrame,
    demo_features: pd.DataFrame,
    decision: np.ndarray,
    predicted: np.ndarray,
    labels: pd.Series,
    types: pd.Series,
) -> None:
    """Fail clearly when report figures no longer match the current project artifacts."""
    model = bundle["model"]
    checks = {
        "Train rows": (len(train), 27334),
        "Train valid features": (len(train_features), 27310),
        "Demo rows": (len(demo), 6834),
        "Demo valid features": (len(demo_features), 6810),
        "Injected valid labels": (int(labels.sum()), 545),
        "Model alerts": (int(predicted.sum()), 764),
        "Feature count": (int(model.n_features_in_), 9),
        "Tree count": (int(model.n_estimators), 200),
        "Samples per tree": (int(model.max_samples_), 512),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{name}: expected {expected}, got {actual}")
    if list(bundle["features"]) != list(ENGINEERED_FEATURE_NAMES):
        raise ValueError("Feature order in model bundle does not match config.py")
    if types.value_counts().to_dict() != EXPECTED_VALID_TYPES:
        raise ValueError(f"Unexpected valid Demo labels: {types.value_counts().to_dict()}")
    if not np.isclose(float(model.offset_), -0.5271250921882412, atol=1e-12):
        raise ValueError(f"Unexpected Isolation Forest offset_: {model.offset_}")
    matrix = confusion_matrix(labels, predicted)
    if matrix.tolist() != [[5898, 367], [148, 397]]:
        raise ValueError(f"Unexpected confusion matrix: {matrix.tolist()}")
    auc = roc_auc_score(labels, -decision)
    if not np.isclose(auc, 0.9231314203709263, atol=1e-12):
        raise ValueError(f"Unexpected ROC-AUC: {auc}")


def figure_1_1(demo: pd.DataFrame) -> None:
    specs = [
        ("power_surge", "Đột biến công suất ban ngày", "Global_active_power", "Công suất tác dụng (kW)", COLORS["orange"]),
        ("voltage_drop", "Sụt điện áp", "Voltage", "Điện áp (V)", COLORS["red"]),
        ("night_spike", "Đột biến công suất ban đêm", "Global_active_power", "Công suất tác dụng (kW)", COLORS["purple"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    for ax, (kind, title, column, ylabel, color) in zip(axes, specs):
        timestamp = demo.index[demo["anomaly_type"] == kind][0]
        window = demo.loc[timestamp - pd.Timedelta(hours=12): timestamp + pd.Timedelta(hours=12)]
        value = float(demo.loc[timestamp, column])
        decimals = 1 if column == "Voltage" else 3
        unit = "V" if column == "Voltage" else "kW"
        ax.plot(window.index, window[column], color=COLORS["blue"], lw=1.8, label="Chuỗi Demo sau khi tiêm")
        ax.scatter([timestamp], [demo.loc[timestamp, column]], color=color, s=56, zorder=4, label="Điểm tiêm")
        ax.axvline(timestamp, color=color, ls="--", lw=1)
        ax.annotate(
            f"{timestamp.strftime('%d/%m/%Y %H:%M')}\n{format_float_vi(value, decimals)} {unit}",
            xy=(timestamp, value), xytext=(8, 10), textcoords="offset points",
            fontsize=8.5, color=color,
        )
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Thời điểm")
    save(fig, "Hinh_1.1_MinhHoa_3_Loai_SuCo.png")


def figure_2_1(train: pd.DataFrame) -> None:
    night = train.index.hour.isin(range(1, 6))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bins = np.linspace(0, train["Global_active_power"].max(), 90)
    ax.hist(train.loc[~night, "Global_active_power"], bins=bins, density=True, alpha=0.58, color=COLORS["orange"], label=f"Các giờ khác (n={format_int_vi((~night).sum())})")
    ax.hist(train.loc[night, "Global_active_power"], bins=bins, density=True, alpha=0.62, color=COLORS["blue"], label=f"Ban đêm 01:00–05:00 (n={format_int_vi(night.sum())})")
    ax.set(
        title="Phân phối công suất tác dụng theo khung giờ trên tập huấn luyện",
        xlabel="Công suất tác dụng (kW)",
        ylabel="Mật độ xác suất (diện tích = 1)",
    )
    ax.legend()
    save(fig, "Hinh_2.1_PhanPhoi_CongSuat_Theo_KhungGio.png")


def figure_2_2(demo: pd.DataFrame, bundle: dict) -> None:
    """Plot three meaningful 2D projections after the Train-fitted scaler."""
    demo_features = extract_features(demo)
    features = bundle["features"]
    scaled = pd.DataFrame(
        bundle["scaler"].transform(demo_features[features].to_numpy()),
        index=demo_features.index,
        columns=features,
    )
    manual_scaled = (
        demo_features[features].to_numpy() - bundle["scaler"].center_
    ) / bundle["scaler"].scale_
    np.testing.assert_allclose(
        scaled[features].to_numpy(), manual_scaled, rtol=1e-12, atol=1e-12,
        err_msg="RobustScaler coordinates do not match (x - center_) / scale_",
    )
    scaled["anomaly_type"] = demo.loc[scaled.index, "anomaly_type"]

    actual_counts = scaled["anomaly_type"].value_counts().to_dict()
    if len(scaled) != 6810 or actual_counts != EXPECTED_VALID_TYPES:
        raise ValueError(
            f"Unexpected valid Demo distribution: rows={len(scaled)}, counts={actual_counts}"
        )

    scaled["hour"] = scaled.index.hour
    panels = [
        {
            "type": "power_surge", "title": "Đột biến công suất ban ngày",
            "x": "power_diff_1h", "y": "power_dev_24h",
            "xlabel": "Biến động công suất so với mẫu trước",
            "ylabel": "Lệch công suất so với 24 mẫu trước",
            "background": (scaled["anomaly_type"] == "normal") & scaled["hour"].between(8, 22),
            "color": COLORS["orange"], "marker": "o",
        },
        {
            "type": "voltage_drop", "title": "Sụt điện áp",
            "x": "voltage_diff_1h", "y": "voltage_zscore_6h",
            "xlabel": "Biến động điện áp so với mẫu trước",
            "ylabel": "Độ lệch điện áp trong 6 mẫu",
            "background": scaled["anomaly_type"] == "normal",
            "color": COLORS["red"], "marker": "v",
        },
        {
            "type": "night_spike", "title": "Đột biến công suất ban đêm",
            "x": "power_dev_24h", "y": "power_zscore_6h",
            "xlabel": "Lệch công suất so với 24 mẫu trước",
            "ylabel": "Độ lệch công suất trong 6 mẫu",
            "background": (scaled["anomaly_type"] == "normal") & scaled["hour"].between(1, 5),
            "color": COLORS["purple"], "marker": "s",
        },
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 12.5))
    for ax, panel in zip(axes, panels):
        background = scaled[panel["background"]]
        points = scaled[scaled["anomaly_type"] == panel["type"]]
        ax.scatter(
            background[panel["x"]], background[panel["y"]],
            s=9, c="#CBD5E1", alpha=0.25, edgecolors="none",
            label=f"Điểm nền cùng khung giờ (n={format_int_vi(len(background))})", rasterized=True,
        )
        ax.scatter(
            points[panel["x"]], points[panel["y"]],
            s=28, c=panel["color"], marker=panel["marker"], alpha=0.82,
            edgecolors="white", linewidths=0.35,
            label=f"Điểm được tiêm (n={format_int_vi(len(points))})", rasterized=True,
        )
        ax.axvline(0, color=COLORS["slate"], ls="--", lw=1, alpha=0.8)
        ax.axhline(0, color=COLORS["slate"], ls="--", lw=1, alpha=0.8)
        ax.set_title(panel["title"])
        ax.set_xlabel(f"{panel['xlabel']}\n(số IQR so với trung vị Train)")
        ax.set_ylabel(f"{panel['ylabel']}\n(số IQR so với trung vị Train)")
        ax.legend(loc="best", frameon=True, fontsize=8)
    fig.suptitle("Ba phép chiếu đặc trưng sau RobustScaler", fontsize=14)
    save(fig, "Hinh_2.2_DacTrung_Sau_RobustScaler.png")


def _box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 0.19, height: float = 0.13) -> tuple[float, float]:
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012", ec=color, fc="white", lw=1.8))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9, wrap=True)
    return x + width / 2, y + height / 2


def figure_2_3(bundle: dict) -> None:
    model = bundle["model"]
    feature_count = len(bundle["features"])
    tree_count = int(model.n_estimators)
    samples_per_tree = int(model.max_samples_)
    offset = float(model.offset_)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        _box(ax, (0.02, 0.69), f"Train đã scale\n{feature_count} đặc trưng", COLORS["blue"], width=0.25),
        _box(ax, (0.37, 0.69), f"Lấy tối đa\n{samples_per_tree} mẫu/cây", COLORS["purple"], width=0.25),
        _box(ax, (0.72, 0.69), f"Học {tree_count} cây\nđặc trưng / ngưỡng ngẫu nhiên", COLORS["orange"], width=0.25),
        _box(ax, (0.02, 0.30), f"Mẫu mới đã scale\n{feature_count} đặc trưng", COLORS["blue"], width=0.25),
        _box(ax, (0.37, 0.30), "Trung bình đường đi\nđã hiệu chỉnh qua các cây", COLORS["green"], width=0.25),
        _box(ax, (0.72, 0.30), "Tính score_samples\nrồi trừ offset_", COLORS["red"], width=0.25),
    ]
    for i, j in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        left, right = boxes[i], boxes[j]
        ax.add_patch(FancyArrowPatch(
            (left[0] + 0.14, left[1]), (right[0] - 0.14, right[1]),
            arrowstyle="->", mutation_scale=14, color=COLORS["slate"], lw=1.5,
        ))
    ax.text(0.02, 0.91, "HUẤN LUYỆN", color=COLORS["blue"], fontweight="bold")
    ax.text(0.02, 0.52, "SUY LUẬN BẰNG RỪNG ĐÃ HỌC", color=COLORS["green"], fontweight="bold")
    ax.add_patch(FancyArrowPatch((0.84, 0.66), (0.50, 0.45), arrowstyle="->", mutation_scale=14, color=COLORS["slate"]))
    ax.text(
        0.50, 0.10,
        f"offset_ = {format_float_vi(offset, 6)}; decision_function = score_samples − offset_\n"
        "decision_function < 0: dự báo bất thường; ≥ 0: dự báo bình thường",
        ha="center", va="center", color=COLORS["slate"], fontsize=9,
    )
    ax.set_title("Huấn luyện và suy luận Isolation Forest", pad=16)
    save(fig, "Hinh_2.3_IsolationForest.png")


def figure_2_4(decision: np.ndarray, labels: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bins = np.linspace(decision.min(), decision.max(), 55)
    injected = labels.to_numpy() == 1
    ax.hist(decision[~injected], bins=bins, density=True, color="#94A3B8", alpha=0.62, label=f"Không được tiêm lỗi (n={format_int_vi((~injected).sum())})")
    ax.hist(decision[injected], bins=bins, density=True, color=COLORS["red"], alpha=0.66, label=f"Được tiêm lỗi (n={format_int_vi(injected.sum())})")
    ax.axvline(0, color="black", ls="--", lw=1.5, label="Ngưỡng quyết định = 0")
    ax.set(
        title="Phân phối điểm quyết định theo nhãn kiểm định tổng hợp",
        xlabel="Điểm quyết định (thấp hơn là bất thường hơn)",
        ylabel="Mật độ xác suất (mỗi nhóm có diện tích = 1)",
    )
    ax.legend()
    save(fig, "Hinh_2.4_PhanPhoi_Diem_QuyetDinh.png")


def figure_3_2(train_features: pd.DataFrame, features: list[str]) -> None:
    corr = train_features[features].corr(method="pearson")
    labels = [
        "Chu kỳ giờ (sin)", "Chu kỳ giờ (cos)", "Giờ đêm",
        "Δ công suất 1 mẫu", "Lệch công suất 24 mẫu", "Độ lệch CS 6 mẫu",
        "Δ điện áp 1 mẫu", "Độ lệch ĐA 6 mẫu", "Hệ số công suất",
    ]
    fig, ax = plt.subplots(figsize=(8.1, 7.1))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    values = np.ma.array(corr.to_numpy(), mask=mask)
    cmap = plt.get_cmap("coolwarm").with_extremes(bad="white")
    image = ax.imshow(values, vmin=-1, vmax=1, cmap=cmap)
    ax.grid(False)
    ax.set_xticks(range(len(features)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(features)), labels)
    for i in range(len(features)):
        for j in range(i + 1):
            value = corr.iloc[i, j]
            ax.text(j, i, format_float_vi(value), ha="center", va="center", fontsize=8, color="white" if abs(value) > .5 else "#0F172A")
    fig.colorbar(image, ax=ax, shrink=.88, label="Hệ số tương quan Pearson")
    ax.set_title("Ma trận tương quan của 9 đặc trưng trên tập huấn luyện")
    save(fig, "Hinh_3.2_MaTran_TuongQuan_9_DacTrung.png")


def figure_4_1(train: pd.DataFrame, demo: pd.DataFrame) -> None:
    train_valid = len(extract_features(train))
    demo_valid_index = extract_features(demo).index
    demo_valid = len(demo_valid_index)
    injected_before = int(demo["is_anomaly"].sum())
    injected_after = int(demo.loc[demo_valid_index, "is_anomaly"].sum())
    train_warmup = len(train) - train_valid
    demo_warmup = len(demo) - demo_valid
    total_rows = len(train) + len(demo)
    train_share = 100 * len(train) / total_rows
    demo_share = 100 * len(demo) / total_rows

    fig = plt.figure(figsize=(11.5, 6.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.55], hspace=0.35)
    timeline = fig.add_subplot(grid[0])
    flow = fig.add_subplot(grid[1])

    timeline.hlines(1, train.index.min(), train.index.max(), color=COLORS["blue"], lw=18)
    timeline.hlines(1, demo.index.min(), demo.index.max(), color=COLORS["orange"], lw=18)
    timeline.axvline(train.index.max(), color=COLORS["slate"], ls="--", lw=1.2)
    train_mid = train.index.min() + (train.index.max() - train.index.min()) / 2
    demo_mid = demo.index.min() + (demo.index.max() - demo.index.min()) / 2
    timeline.text(train_mid, 1.03, f"HUẤN LUYỆN — {format_float_vi(train_share, 0)}% — {format_int_vi(len(train))} dòng", ha="center", va="bottom", color=COLORS["blue"], fontweight="bold")
    timeline.text(demo_mid, 1.03, f"KIỂM ĐỊNH/DEMO — {format_float_vi(demo_share, 0)}% — {format_int_vi(len(demo))} dòng", ha="center", va="bottom", color=COLORS["orange"], fontweight="bold")
    timeline.text(train.index.min(), .94, train.index.min().strftime("%d/%m/%Y %H:%M"), ha="left", va="top", fontsize=8.5)
    timeline.text(train.index.max(), .94, train.index.max().strftime("%d/%m/%Y %H:%M"), ha="right", va="top", fontsize=8.5)
    timeline.text(demo.index.min(), .86, demo.index.min().strftime("%d/%m/%Y %H:%M"), ha="left", va="top", fontsize=8.5)
    timeline.text(demo.index.max(), .94, demo.index.max().strftime("%d/%m/%Y %H:%M"), ha="right", va="top", fontsize=8.5)
    timeline.set(ylim=(.72, 1.22), yticks=[], xlabel="Thời gian", title="Chia dữ liệu theo đúng thứ tự thời gian")

    flow.set_xlim(0, 1); flow.set_ylim(0, 1); flow.axis("off")
    rows = [
        (
            0.62, COLORS["blue"],
            [
                f"Train\n{format_int_vi(len(train))} dòng",
                f"Tính 9 đặc trưng\nloại {format_int_vi(train_warmup)} dòng warm-up",
                f"{format_int_vi(train_valid)} mẫu\nhợp lệ",
                "Fit RobustScaler\nvà Isolation Forest",
            ],
        ),
        (
            0.18, COLORS["orange"],
            [
                f"Demo\n{format_int_vi(len(demo))} dòng",
                f"Tiêm {format_int_vi(injected_before)} điểm\nseed 42",
                f"Tính 9 đặc trưng\nloại {format_int_vi(demo_warmup)} dòng warm-up",
                f"{format_int_vi(demo_valid)} mẫu hợp lệ\n{format_int_vi(injected_after)} nhãn tiêm",
            ],
        ),
    ]
    for y, color, texts in rows:
        centers = []
        for x, text in zip([0.02, 0.27, 0.52, 0.77], texts):
            centers.append(_box(flow, (x, y), text, color, width=0.19, height=0.17))
        for left, right in zip(centers, centers[1:]):
            flow.add_patch(FancyArrowPatch(
                (left[0] + 0.105, left[1]), (right[0] - 0.105, right[1]),
                arrowstyle="->", mutation_scale=14, color=COLORS["slate"], lw=1.5,
            ))
    flow.set_title("Luồng tạo dữ liệu huấn luyện và kiểm định", pad=8)
    save(fig, "Hinh_4.1_PhanBo_DuLieu.png", use_tight_layout=False)


def figure_5_1(labels: pd.Series, decision: np.ndarray) -> None:
    auc = roc_auc_score(labels, -decision)
    fpr, tpr, _ = roc_curve(labels, -decision)
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    ax.plot(fpr, tpr, color=COLORS["orange"], lw=2.2, label=f"ROC-AUC = {format_float_vi(auc, 4)}")
    ax.plot([0, 1], [0, 1], color=COLORS["slate"], ls="--", label="Ngẫu nhiên")
    ax.set(title="Đường cong ROC trên nhãn kiểm định tổng hợp", xlabel="Tỷ lệ dương tính giả", ylabel="Tỷ lệ dương tính đúng")
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    ax.text(.03, .97, f"{format_int_vi(positive_count)} điểm được tiêm / {format_int_vi(negative_count)} điểm không được tiêm", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=COLORS["slate"])
    ax.legend(loc="lower right")
    save(fig, "Hinh_5.1_ROC_Curve.png")


def figure_5_2(labels: pd.Series, predicted: np.ndarray) -> None:
    matrix = confusion_matrix(labels, predicted)
    row_percent = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    image = ax.imshow(row_percent, vmin=0, vmax=1, cmap="Blues")
    ax.grid(False)
    cell_labels = [
        ["Đúng bình thường\n(TN)", "Cảnh báo ngoài\nnhãn tiêm (FP)"],
        ["Bỏ sót (FN)", "Phát hiện đúng (TP)"],
    ]
    for i in range(2):
        for j in range(2):
            percent = row_percent[i, j] * 100
            text_color = "white" if row_percent[i, j] >= .55 else "#0F172A"
            ax.text(
                j, i, f"{cell_labels[i][j]}\n{format_int_vi(matrix[i, j])} điểm\n{format_float_vi(percent, 2)}% theo hàng",
                ha="center", va="center", fontsize=9.2, color=text_color,
                fontweight="bold" if i == j else "normal",
            )
    ax.set_xticks([0, 1], ["Dự đoán bình thường", "Dự đoán bất thường"])
    ax.set_yticks([0, 1], ["Không được tiêm lỗi", "Được tiêm lỗi"])
    ax.set_title("Ma trận nhầm lẫn trên nhãn kiểm định tổng hợp")
    ax.set_xlabel("Kết quả của Isolation Forest")
    ax.set_ylabel("Nhãn kiểm định tổng hợp")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Tỷ lệ trong từng hàng")
    ax.text(
        0.5, -0.17,
        f"Phát hiện {format_int_vi(matrix[1, 1])}/{format_int_vi(matrix[1].sum())} điểm tiêm; "
        f"{format_int_vi(matrix[0, 1])} cảnh báo không trùng thời điểm tiêm.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=COLORS["slate"],
    )
    save(fig, "Hinh_5.2_Confusion_Matrix.png")


def figure_5_3(types: pd.Series, predicted: np.ndarray) -> None:
    order = ["power_surge", "voltage_drop", "night_spike"]
    predicted_s = pd.Series(predicted, index=types.index)
    totals = types.value_counts()
    caught = predicted_s.groupby(types).sum()
    injected = types != "normal"
    counts = [(int(predicted_s[injected].sum()), int(injected.sum()))] + [
        (int(caught[name]), int(totals[name])) for name in order
    ]
    recalls = [caught_count / total * 100 for caught_count, total in counts]
    display_labels = ["Tổng thể", "Đột biến công suất\nban ngày", "Sụt điện áp", "Đột biến công suất\nban đêm"]
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    bars = ax.bar(display_labels, recalls, color=[COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["purple"]])
    for bar, (caught_count, total), rate in zip(bars, counts, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2.2, f"{caught_count}/{total}\n{format_float_vi(rate, 2)}%", ha="center", va="bottom", fontweight="bold")
    ax.set(ylim=(0, 108), ylabel="Recall (%)", title="Recall tổng thể và theo từng kịch bản tiêm")
    save(fig, "Hinh_5.3_Recall_Theo_Tung_Loai_Loi.png")


def main() -> None:
    setup_style()
    train, demo, bundle, demo_features, decision, predicted, labels, types = load_inputs()
    train_features = extract_features(train)
    validate_inputs(train, demo, bundle, train_features, demo_features, decision, predicted, labels, types)
    figure_1_1(demo)
    figure_2_1(train)
    figure_2_2(demo, bundle)
    figure_2_3(bundle)
    figure_2_4(decision, labels)
    figure_3_2(train_features, bundle["features"])
    figure_4_1(train, demo)
    figure_5_1(labels, decision)
    figure_5_2(labels, predicted)
    figure_5_3(types, predicted)
    generate_diagrams(FIGURES_DIR)
    print(f"Generated 10 PNG figures and 4 SVG diagrams in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
