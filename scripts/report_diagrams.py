"""Tạo sơ đồ vector mô tả pipeline đã triển khai."""

from html import escape
from pathlib import Path


# Xây dựng sơ đồ SVG từ các phần tử cơ bản.
class Diagram:
    # Khởi tạo khung SVG và tiêu đề.
    def __init__(self, title, height=640):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}" viewBox="0 0 1080 {height}" role="img">',
            f"<title>{escape(title)}</title>",
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/></marker></defs>',
            f'<rect width="1080" height="{height}" fill="white"/>',
            '<g font-family="DejaVu Sans, Arial, sans-serif" font-size="17" fill="#0f172a">',
        ]
        self.text(540, 34, title, size=25, bold=True)

    # Thêm một khối văn bản nhiều dòng.
    def text(self, x, y, label, size=17, bold=False, anchor="middle", color="#0f172a"):
        lines = label.split("\n")
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{"bold" if bold else "normal"}">'
        )
        for i, line in enumerate(lines):
            self.parts.append(
                f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.45}">{escape(line)}</tspan>'
            )
        self.parts.append("</text>")

    # Thêm hộp bo góc có nhãn.
    def box(self, x, y, w, h, label, color="#2563eb", fill="#eff6ff", size=17):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{color}" stroke-width="2"/>'
        )
        lines = len(label.split("\n"))
        self.text(
            x + w / 2,
            y + h / 2 - (lines - 1) * size * 0.725 + size * 0.32,
            label,
            size=size,
        )

    # Thêm mũi tên nối các thành phần.
    def arrow(self, points, label=None, label_xy=None):
        self.parts.append(
            f'<polyline points="{points}" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>'
        )
        if label:
            self.text(*label_xy, label, size=15, color="#475569")

    # Ghi sơ đồ hoàn chỉnh ra tệp SVG.
    def save(self, path):
        path.write_text("\n".join(self.parts + ["</g></svg>"]) + "\n", encoding="utf-8")


# Tạo toàn bộ sơ đồ kiến trúc dùng trong báo cáo.
def generate_diagrams(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    d = Diagram("Kiến trúc và luồng dữ liệu của hệ thống", 760)
    d.box(
        350,
        65,
        380,
        65,
        "Dữ liệu UCI theo phút\n01_data_prep.py: làm sạch, gom giờ",
        size=16,
    )
    d.box(45, 180, 410, 65, "train_hourly.csv\n80% đầu theo thời gian")
    d.box(625, 180, 410, 65, "demo_stream.csv\n20% cuối + bất thường tổng hợp")
    d.arrow("440,130 440,155 250,155 250,180")
    d.arrow("640,130 640,155 830,155 830,180")
    d.box(
        45,
        295,
        410,
        85,
        "features.py → 9 đặc trưng\n02_train.py: fit RobustScaler\nvà Isolation Forest trên Train",
        size=16,
    )
    d.arrow("250,245 250,295")
    d.box(
        45,
        425,
        410,
        75,
        "model_bundle.pkl\nmodel, scaler, features, medians, iqrs",
        size=16,
    )
    d.arrow("250,380 250,425")
    d.box(
        625,
        300,
        410,
        110,
        "04_dashboard.py + features.py\nLịch sử / mô phỏng tuần tự\nTrích đặc trưng → transform → suy luận",
        size=16,
    )
    d.arrow("830,245 830,300")
    d.arrow("455,462 540,462 540,355 625,355")
    d.box(
        625,
        475,
        410,
        75,
        "SQLite: alerts\nLưu cảnh báo từ mô phỏng; đọc hàng đợi",
        size=16,
    )
    d.arrow("830,410 830,475")
    d.box(
        625,
        610,
        410,
        85,
        "Người vận hành qua dashboard\nTiếp nhận / đóng / ghi chú\nXuất hàng đợi đã lọc ra CSV",
        size=16,
    )
    d.arrow("830,550 830,610")
    d.box(
        45,
        600,
        410,
        95,
        "Tiện ích độc lập: 03_producer.py\ndemo_stream.csv → stream_buffer.jsonl\nDashboard không đọc JSONL",
        size=16,
    )
    d.save(folder / "Hinh_3.1_KienTruc_HeThong.svg")

    d = Diagram("Luồng suy luận lịch sử và mô phỏng tuần tự", 1030)
    for x, title in [(35, "PHÂN TÍCH LỊCH SỬ"), (565, "MÔ PHỎNG TUẦN TỰ")]:
        d.text(x + 240, 78, title, 21, True)
    left = [
        "Lọc khoảng ngày từ Demo",
        "extract_features trên phần đã chọn\nKhông có vector hợp lệ: thông báo, dừng",
        "Scaler.transform → decision_function\nDùng bundle đã học trên Train",
        "Căn kết quả với index đặc trưng\nd(x) < 0: bất thường; còn lại: bình thường",
        "Tính severity; gợi ý loại và dấu hiệu\ncho các điểm bất thường",
        "Hiển thị KPI, biểu đồ và bảng\nKhông ghi cảnh báo vào SQLite",
    ]
    right = [
        "Đọc hàng Demo theo con trỏ\nThêm vào buffer, giữ tối đa 50 mẫu",
        "extract_latest: cần ít nhất 25 mẫu\nChưa đủ: warming_up, chưa chấm điểm",
        "Có vector: transform → decision_function\nDùng bundle đã học trên Train",
        "d(x) < 0: bất thường\nNgược lại: bình thường, không tạo cảnh báo",
        "Tính severity; gợi ý loại và dấu hiệu\ncho điểm bất thường",
        "Lưu cảnh báo theo ID thời điểm\nUpsert vào SQLite",
    ]
    for x, labels in [(35, left), (565, right)]:
        for i, label in enumerate(labels):
            y = 110 + i * 140
            d.box(x, y, 480, 95, label, size=16)
            if i < 5:
                d.arrow(f"{x + 240},{y + 95} {x + 240},{y + 140}")
    d.box(565, 940, 480, 65, "Cập nhật hiển thị / chuyển mẫu tiếp", size=16)
    d.arrow("805,905 805,940")
    d.arrow("1045,297 1065,297 1065,972 1045,972")
    d.text(963, 370, "Chưa đủ: bỏ qua", 14)
    d.arrow("1045,577 1065,577")
    d.text(966, 650, "Bình thường: bỏ qua", 14)
    d.text(430, 370, "Có vector", 14)
    d.text(935, 370, "", 14)
    d.text(715, 370, "Có vector", 14)
    d.text(695, 650, "Bất thường", 14)
    d.save(folder / "Hinh_3.3_Luong_SuyLuan.svg")

    d = Diagram("Vòng đời cảnh báo qua thao tác trên dashboard", 560)
    for x, w, label in [
        (35, 240, "Mới\nnew"),
        (405, 270, "Đã tiếp nhận\nacknowledged"),
        (805, 240, "Đã đóng\nclosed"),
    ]:
        d.box(x, 170, w, 85, label)
    d.arrow("275,212 405,212", "Tiếp nhận", (340, 150))
    d.arrow("675,212 805,212", "Đóng", (740, 150))
    d.arrow("155,255 155,355 925,355 925,255", "Đóng trực tiếp", (540, 340))
    d.text(540, 90, "Cảnh báo mô phỏng mới → lưu SQLite với trạng thái new", 18)
    d.text(
        540,
        420,
        "Tiếp nhận: cập nhật acknowledged_at. Đóng: cập nhật closed_at.\nCả hai lưu ghi chú và updated_at; giao diện không có thao tác mở lại.",
        17,
    )
    d.text(
        540,
        505,
        "Upsert cùng alert_id giữ trạng thái, ghi chú và thời điểm xử lý đã lưu.",
        17,
    )
    d.save(folder / "Hinh_3.4_VongDoi_CanhBao.svg")

    d = Diagram("Bố cục chức năng của hai chế độ dashboard", 1010)
    left = [
        "Bộ lọc ngày / Xem toàn bộ",
        "4 KPI theo dữ liệu đã đánh giá",
        "Công suất  |  Phân bố dạng bất thường",
        "Điện áp  |  Phân bố theo giờ",
        "Bảng các điểm bất thường\nSố đo, điểm cảnh báo, loại, dấu hiệu",
    ]
    right = [
        "Play / Pause / Bước tiếp / Khởi động lại\nTốc độ và tiến độ phát luồng",
        "4 KPI của phiên mô phỏng",
        "Thông báo warm-up / cảnh báo mới nhất",
        "Biểu đồ công suất  |  Biểu đồ điện áp\nTối đa 100 mẫu gần nhất",
        "Hàng đợi: lọc trạng thái và dạng gợi ý\nBảng cảnh báo / Tải CSV",
        "Chọn ID → chi tiết → ghi chú\nTiếp nhận / Đóng cảnh báo",
    ]
    for x, title, labels in [
        (35, "PHÂN TÍCH LỊCH SỬ", left),
        (565, "GIÁM SÁT MÔ PHỎNG", right),
    ]:
        d.text(x + 240, 88, title, 21, True)
        for i, label in enumerate(labels):
            d.box(x, 125 + i * 130, 480, 95, label, size=16)
    d.text(
        540,
        970,
        "Sơ đồ vùng chức năng theo mã giao diện hiện tại; không biểu diễn số liệu đo.",
        16,
    )
    d.save(folder / "Hinh_3.5_Wireframe.svg")
    generate_general_model_diagrams(folder)


# Tạo ba sơ đồ cho tài liệu mô hình tổng quát.
def generate_general_model_diagrams(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    d = Diagram("Mô hình kiến trúc tổng thể của hệ thống", 930)
    d.text(70, 86, "TUYẾN NGOẠI TUYẾN", 18, True, anchor="start", color="#2563eb")
    d.box(55, 115, 260, 82, "Dữ liệu UCI theo phút\nDữ liệu đo hộ gia đình", size=16)
    d.box(
        410,
        115,
        260,
        82,
        "Tầng 1 — Chuẩn bị dữ liệu\nLàm sạch, gom giờ, chia 80/20",
        color="#0f766e",
        fill="#f0fdfa",
        size=16,
    )
    d.box(
        765,
        115,
        260,
        82,
        "Hai tập dữ liệu\nTrain và Demo có nhãn tổng hợp",
        color="#c2410c",
        fill="#fff7ed",
        size=16,
    )
    d.arrow("315,156 410,156")
    d.arrow("670,156 765,156")

    d.box(
        55,
        260,
        260,
        100,
        "Tầng 2 — Đặc trưng\nfeatures.py\n9 đặc trưng thời gian và điện",
        color="#7c3aed",
        fill="#faf5ff",
        size=16,
    )
    d.box(
        410,
        260,
        260,
        100,
        "Huấn luyện\nRobustScaler + Isolation Forest\nchỉ fit trên Train",
        color="#7c3aed",
        fill="#faf5ff",
        size=16,
    )
    d.box(
        765,
        260,
        260,
        100,
        "model_bundle.pkl\nmodel, scaler, features,\nmedians và iqrs",
        color="#7c3aed",
        fill="#faf5ff",
        size=16,
    )
    d.arrow("895,197 895,230 185,230 185,260")
    d.arrow("315,310 410,310")
    d.arrow("670,310 765,310")

    d.text(70, 425, "TUYẾN VẬN HÀNH", 18, True, anchor="start", color="#0f766e")
    d.box(
        55,
        455,
        260,
        100,
        "demo_stream.csv\nNguồn cho lịch sử và\nmô phỏng tuần tự",
        color="#0f766e",
        fill="#f0fdfa",
        size=16,
    )
    d.box(
        410,
        455,
        260,
        100,
        "Tầng 3 — Suy luận\nBuffer → 9 đặc trưng → scale\n→ Isolation Forest → hậu xử lý",
        color="#0f766e",
        fill="#f0fdfa",
        size=16,
    )
    d.box(
        765,
        455,
        260,
        100,
        "Tầng 4 — Trình bày\nDashboard lịch sử / trực tiếp\nKPI, biểu đồ và bảng",
        color="#0f766e",
        fill="#f0fdfa",
        size=16,
    )
    d.arrow("315,505 410,505")
    d.arrow("670,505 765,505")
    d.arrow("895,360 895,405 540,405 540,455", "Nạp bundle đã học", (725, 397))

    d.box(
        55,
        640,
        260,
        92,
        "Tiện ích độc lập\n03_producer.py xuất JSONL\nDashboard không đọc JSONL",
        color="#64748b",
        fill="#f8fafc",
        size=15,
    )
    d.box(
        410,
        640,
        260,
        92,
        "Hậu xử lý kết quả\nSeverity, dạng gợi ý,\nTop-3 dấu hiệu XAI",
        color="#d97706",
        fill="#fffbeb",
        size=16,
    )
    d.box(
        765,
        640,
        260,
        92,
        "SQLite alerts\nChỉ nhận cảnh báo từ\nmô phỏng tuần tự",
        color="#dc2626",
        fill="#fef2f2",
        size=16,
    )
    d.arrow("185,555 185,640", "Tùy chọn xuất luồng", (270, 605))
    d.arrow("540,555 540,640")
    d.arrow("670,686 765,686", "Nếu bất thường", (708, 670))
    d.arrow("895,732 895,780")
    d.box(
        765,
        780,
        260,
        92,
        "Người vận hành\nLọc, tiếp nhận, ghi chú, đóng\nvà xuất danh sách cảnh báo",
        color="#2563eb",
        fill="#eff6ff",
        size=16,
    )
    d.text(
        55,
        900,
        "Ranh giới nguyên mẫu: chạy cục bộ bằng CSV, model bundle, Streamlit và SQLite.",
        15,
        anchor="start",
        color="#475569",
    )
    d.save(folder / "Hinh_MHTQ_01_KienTruc_TongThe.svg")

    d = Diagram("Luồng xử lý suy luận của hai chế độ", 1040)
    d.text(285, 78, "PHÂN TÍCH LỊCH SỬ", 20, True, color="#2563eb")
    d.text(795, 78, "MÔ PHỎNG TUẦN TỰ", 20, True, color="#0f766e")
    history = [
        "Đọc toàn bộ Demo và bộ lọc ngày",
        "Tính 9 đặc trưng trên toàn chuỗi\nđể giữ ngữ cảnh trước khoảng lọc",
        "RobustScaler.transform\n→ IsolationForest.decision_function",
        "Căn theo index hợp lệ\nrồi mới áp dụng khoảng ngày",
        "d(x) < 0: hậu xử lý bất thường\nd(x) ≥ 0: bình thường",
        "Hiển thị KPI, biểu đồ và bảng\nKhông ghi vào SQLite",
    ]
    realtime = [
        "Đọc hàng Demo tiếp theo theo con trỏ\nBuffer phiên giữ tối đa 50 mẫu",
        "Buffer có ít nhất 25 mẫu?\n24 mẫu đầu là warm-up",
        "extract_latest → transform\n→ decision_function cho mẫu mới nhất",
        "d(x) < 0?",
        "Có: severity, dạng gợi ý, XAI\nKhông: ghi nhận kết quả bình thường",
        "Lưu cảnh báo vào SQLite nếu có\nCập nhật dashboard; tối đa 100 mẫu",
    ]
    for x, labels, color, fill in [
        (45, history, "#2563eb", "#eff6ff"),
        (555, realtime, "#0f766e", "#f0fdfa"),
    ]:
        for i, label in enumerate(labels):
            y = 115 + i * 142
            d.box(x, y, 480, 96, label, color=color, fill=fill, size=16)
            if i < len(labels) - 1:
                d.arrow(f"{x + 240},{y + 96} {x + 240},{y + 142}")
    d.text(920, 374, "Không đủ: warm-up", 14, color="#b45309")
    d.text(920, 658, "Có bất thường: tạo alert_id", 14, color="#b91c1c")
    d.text(
        540,
        1000,
        "Cả hai chế độ dùng chung model bundle, công thức đặc trưng và hậu xử lý.",
        16,
        color="#475569",
    )
    d.save(folder / "Hinh_MHTQ_02_Luong_XuLy_SuyLuan.svg")

    d = Diagram("Vòng đời và dữ liệu của một cảnh báo", 660)
    d.text(
        540,
        90,
        "Mẫu được Isolation Forest đánh dấu bất thường trong mô phỏng tuần tự",
        17,
    )
    d.box(
        55, 145, 250, 92, "MỚI\nstatus = new", color="#dc2626", fill="#fef2f2", size=18
    )
    d.box(
        415,
        145,
        250,
        92,
        "ĐÃ TIẾP NHẬN\nstatus = acknowledged",
        color="#d97706",
        fill="#fffbeb",
        size=17,
    )
    d.box(
        775,
        145,
        250,
        92,
        "ĐÃ ĐÓNG\nstatus = closed",
        color="#0f766e",
        fill="#f0fdfa",
        size=18,
    )
    d.arrow("305,191 415,191", "Tiếp nhận", (360, 168))
    d.arrow("665,191 775,191", "Đóng", (720, 168))
    d.arrow("180,237 180,315 900,315 900,237", "Cho phép đóng trực tiếp", (540, 300))

    d.box(
        55,
        390,
        300,
        120,
        "Định danh và dữ liệu mô hình\nALT-{run_id}-{timestamp}\ndata_time, power, voltage\nseverity, level, type, explanation",
        color="#2563eb",
        fill="#eff6ff",
        size=15,
    )
    d.box(
        390,
        390,
        300,
        120,
        "Dữ liệu xử lý vận hành\nstatus, note\nacknowledged_at, closed_at\nupdated_at",
        color="#7c3aed",
        fill="#faf5ff",
        size=15,
    )
    d.box(
        725,
        390,
        300,
        120,
        "Lưu trữ SQLite\nalert_id là khóa chính\nUpsert giữ trạng thái và ghi chú\nqua các lần rerun Streamlit",
        color="#64748b",
        fill="#f8fafc",
        size=15,
    )
    d.arrow("355,450 390,450")
    d.arrow("690,450 725,450")
    d.text(
        540,
        575,
        "Dashboard hỗ trợ lọc theo trạng thái và dạng gợi ý, xem chi tiết, ghi chú và tải CSV.",
        16,
    )
    d.text(
        540,
        615,
        "Hiện tại không có thao tác mở lại cảnh báo đã đóng.",
        15,
        color="#475569",
    )
    d.save(folder / "Hinh_MHTQ_03_VongDoi_CanhBao.svg")
