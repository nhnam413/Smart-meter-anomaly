"""Vector diagrams of the implemented pipeline, without runtime app changes."""
from html import escape
from pathlib import Path


class Diagram:
    def __init__(self, title, height=640):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}" viewBox="0 0 1080 {height}" role="img">',
            f'<title>{escape(title)}</title>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/></marker></defs>',
            f'<rect width="1080" height="{height}" fill="white"/>',
            '<g font-family="DejaVu Sans, Arial, sans-serif" font-size="17" fill="#0f172a">',
        ]
        self.text(540, 34, title, size=25, bold=True)

    def text(self, x, y, label, size=17, bold=False, anchor="middle", color="#0f172a"):
        lines = label.split("\n")
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{"bold" if bold else "normal"}">')
        for i, line in enumerate(lines):
            self.parts.append(f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.45}">{escape(line)}</tspan>')
        self.parts.append('</text>')

    def box(self, x, y, w, h, label, color="#2563eb", fill="#eff6ff", size=17):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{color}" stroke-width="2"/>')
        lines = len(label.split("\n"))
        self.text(x+w/2, y+h/2-(lines-1)*size*.725+size*.32, label, size=size)

    def arrow(self, points, label=None, label_xy=None):
        self.parts.append(f'<polyline points="{points}" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>')
        if label:
            self.text(*label_xy, label, size=15, color="#475569")

    def save(self, path):
        path.write_text("\n".join(self.parts+['</g></svg>'])+"\n", encoding="utf-8")


def generate_diagrams(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    d = Diagram("Kiến trúc mô-đun và các tệp trao đổi", 700)
    d.box(40, 75, 230, 65, "UCI: số đo theo phút")
    d.box(365, 75, 350, 65, "01_data_prep.py\nLàm sạch → tổng hợp → chia thời gian")
    d.arrow("270,108 365,108")
    d.box(100, 195, 340, 70, "train_hourly.csv\nĐầu vào huấn luyện")
    d.box(640, 195, 340, 70, "Demo sau tiêm tổng hợp\ndemo_stream.csv", "#d97706", "#fff7ed")
    d.arrow("470,140 470,166 270,166 270,195")
    d.arrow("610,140 610,166 810,166 810,195")
    d.box(100, 315, 340, 80, "02_train.py + features.py\nFit scaler → fit Isolation Forest")
    d.arrow("270,265 270,315")
    d.box(100, 445, 340, 80, "model_bundle.pkl\nmodel, scaler, features, medians, iqrs", size=16)
    d.arrow("270,395 270,445")
    d.box(640, 335, 340, 100, "04_dashboard.py + features.py\nLịch sử / mô phỏng tuần tự\nTransform → suy luận → hậu xử lý", "#059669", "#ecfdf5", 16)
    d.arrow("810,265 810,335", "đọc CSV", (870,305))
    d.arrow("440,485 555,485 555,385 640,385", "nạp bundle", (525,465))
    d.box(640, 505, 340, 70, "SQLite: alerts\nLưu và cập nhật trạng thái", "#7c3aed", "#f5f3ff")
    d.arrow("810,435 810,505", "cảnh báo mô phỏng", (902,475))
    d.box(640, 615, 340, 55, "Người vận hành / CSV đã lọc", "#475569", "#f8fafc")
    d.arrow("810,575 810,615")
    d.text(270, 595, "03_producer.py xuất JSONL tùy chọn.\nDashboard hiện chưa đọc JSONL.", 16, color="#475569")
    d.save(folder / "Hinh_3.1_KienTruc_HeThong.svg")

    d = Diagram("Luồng suy luận tuần tự của một mẫu", 690)
    d.box(330, 70, 420, 55, "Nhận dòng Demo theo con trỏ")
    d.box(330, 165, 420, 60, "Thêm vào bộ đệm FIFO, tối đa 50 mẫu")
    d.arrow("540,125 540,165")
    d.box(330, 265, 420, 60, "Đủ 25 mẫu và trích được đặc trưng?", "#7c3aed", "#f5f3ff")
    d.arrow("540,225 540,265")
    d.box(25, 365, 245, 75, "warming_up\nis_anomaly = None", "#64748b", "#f8fafc")
    d.arrow("330,295 147,295 147,365", "Chưa", (245,284))
    d.box(330, 365, 420, 75, "Scaler.transform → decision_function\nDùng bundle đã học từ Train")
    d.arrow("540,325 540,365", "Có", (571,350))
    d.box(330, 485, 420, 55, "Điểm quyết định < 0?", "#7c3aed", "#f5f3ff")
    d.arrow("540,440 540,485")
    d.box(790, 480, 265, 70, "Dự báo bình thường\nKhông tạo cảnh báo", "#059669", "#ecfdf5")
    d.arrow("750,512 790,512", "Không", (902,463))
    d.box(330, 590, 420, 75, "Tính severity, gợi ý loại và dấu hiệu\nUpsert cảnh báo theo alert_id", "#dc2626", "#fef2f2")
    d.arrow("540,540 540,590", "Có", (570,575))
    d.text(145, 515, "Cập nhật hiển thị;\nchờ dòng tiếp theo.", 16, color="#475569")
    d.save(folder / "Hinh_3.3_Luong_SuyLuan.svg")

    d = Diagram("Vòng đời xử lý cảnh báo trong giao diện", 420)
    d.box(40, 155, 230, 75, "Mới\nnew")
    d.box(415, 155, 250, 75, "Đã tiếp nhận\nacknowledged", "#d97706", "#fff7ed")
    d.box(810, 155, 230, 75, "Đã đóng\nclosed", "#059669", "#ecfdf5")
    d.arrow("270,192 415,192", "Tiếp nhận + ghi chú", (340,139))
    d.arrow("665,192 810,192", "Đóng + ghi chú", (737,139))
    d.arrow("155,230 155,295 925,295 925,230", "Đóng trực tiếp", (540,285))
    d.text(540, 85, "Mô hình phát hiện → tạo mã theo thời điểm dữ liệu → lưu SQLite", 18)
    d.text(540, 357, "Lưu lại cùng ID cập nhật số đo, giữ trạng thái và ghi chú.\nMẫu mới bình thường không tự đóng cảnh báo đang chờ.", 18, color="#475569")
    d.save(folder / "Hinh_3.4_VongDoi_CanhBao.svg")

    d = Diagram("Phác thảo bố cục hai chế độ đang hiện thực", 680)
    for x, title, mode in [(35,"PHÂN TÍCH LỊCH SỬ","history"),(565,"GIÁM SÁT MÔ PHỎNG","stream")]:
        d.box(x,70,480,565,"", "#cbd5e1", "#f8fafc")
        d.text(x+240,108,title,21,True)
        d.box(x+20,130,440,50,"Chọn khoảng ngày" if mode=="history" else "Play / Pause / Bước tiếp / Khởi động lại",size=15)
        d.box(x+20,200,440,55,"KPI theo phạm vi đã chọn" if mode=="history" else "KPI phiên và cảnh báo cần xử lý",size=16)
        d.box(x+20,280,440,105,"Biểu đồ công suất và điện áp",size=18)
        if mode=="history":
            d.box(x+20,410,440,75,"Phân bố theo dạng và khung giờ",size=17)
            d.box(x+20,510,440,95,"Danh sách điểm bất thường\ntrong khoảng lịch sử",size=17)
        else:
            d.box(x+20,410,440,75,"Lọc trạng thái / dạng gợi ý\nHàng đợi cảnh báo + tải CSV",size=17)
            d.box(x+20,510,440,95,"Chọn ID → chi tiết → ghi chú\nTiếp nhận / Đóng cảnh báo",size=17)
    d.text(540,661,"Wireframe diễn tả vùng chức năng; ảnh chụp giao diện thật nằm ở Chương 4.",16,color="#475569")
    d.save(folder / "Hinh_3.5_Wireframe.svg")
