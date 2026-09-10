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
    d = Diagram("Kiến trúc và luồng dữ liệu của hệ thống", 760)
    d.box(350,65,380,65,"Dữ liệu UCI theo phút\n01_data_prep.py: làm sạch, gom giờ",size=16)
    d.box(45,180,410,65,"train_hourly.csv\n80% đầu theo thời gian")
    d.box(625,180,410,65,"demo_stream.csv\n20% cuối + bất thường tổng hợp")
    d.arrow("440,130 440,155 250,155 250,180")
    d.arrow("640,130 640,155 830,155 830,180")
    d.box(45,295,410,85,"features.py → 9 đặc trưng\n02_train.py: fit RobustScaler\nvà Isolation Forest trên Train",size=16)
    d.arrow("250,245 250,295")
    d.box(45,425,410,75,"model_bundle.pkl\nmodel, scaler, features, medians, iqrs",size=16)
    d.arrow("250,380 250,425")
    d.box(625,300,410,110,"04_dashboard.py + features.py\nLịch sử / mô phỏng tuần tự\nTrích đặc trưng → transform → suy luận",size=16)
    d.arrow("830,245 830,300")
    d.arrow("455,462 540,462 540,355 625,355")
    d.box(625,475,410,75,"SQLite: alerts\nLưu cảnh báo từ mô phỏng; đọc hàng đợi",size=16)
    d.arrow("830,410 830,475")
    d.box(625,610,410,85,"Người vận hành qua dashboard\nTiếp nhận / đóng / ghi chú\nXuất hàng đợi đã lọc ra CSV",size=16)
    d.arrow("830,550 830,610")
    d.box(45,600,410,95,"Tiện ích độc lập: 03_producer.py\ndemo_stream.csv → stream_buffer.jsonl\nDashboard không đọc JSONL",size=16)
    d.save(folder / "Hinh_3.1_KienTruc_HeThong.svg")

    d = Diagram("Luồng suy luận lịch sử và mô phỏng tuần tự", 1030)
    for x,title in [(35,"PHÂN TÍCH LỊCH SỬ"),(565,"MÔ PHỎNG TUẦN TỰ")]:
        d.text(x+240,78,title,21,True)
    left=[
        "Lọc khoảng ngày từ Demo",
        "extract_features trên phần đã chọn\nKhông có vector hợp lệ: thông báo, dừng",
        "Scaler.transform → decision_function\nDùng bundle đã học trên Train",
        "Căn kết quả với index đặc trưng\nd(x) < 0: bất thường; còn lại: bình thường",
        "Tính severity; gợi ý loại và dấu hiệu\ncho các điểm bất thường",
        "Hiển thị KPI, biểu đồ và bảng\nKhông ghi cảnh báo vào SQLite"]
    right=[
        "Đọc hàng Demo theo con trỏ\nThêm vào buffer, giữ tối đa 50 mẫu",
        "extract_latest: cần ít nhất 25 mẫu\nChưa đủ: warming_up, chưa chấm điểm",
        "Có vector: transform → decision_function\nDùng bundle đã học trên Train",
        "d(x) < 0: bất thường\nNgược lại: bình thường, không tạo cảnh báo",
        "Tính severity; gợi ý loại và dấu hiệu\ncho điểm bất thường",
        "Lưu cảnh báo theo ID thời điểm\nUpsert vào SQLite"]
    for x,labels in [(35,left),(565,right)]:
        for i,label in enumerate(labels):
            y=110+i*140
            d.box(x,y,480,95,label,size=16)
            if i<5: d.arrow(f"{x+240},{y+95} {x+240},{y+140}")
    d.box(565,940,480,65,"Cập nhật hiển thị / chuyển mẫu tiếp",size=16)
    d.arrow("805,905 805,940")
    d.arrow("1045,297 1065,297 1065,972 1045,972")
    d.text(963,370,"Chưa đủ: bỏ qua",14)
    d.arrow("1045,577 1065,577")
    d.text(966,650,"Bình thường: bỏ qua",14)
    d.text(430,370,"Có vector",14)
    d.text(935,370,"",14)
    d.text(715,370,"Có vector",14)
    d.text(695,650,"Bất thường",14)
    d.save(folder / "Hinh_3.3_Luong_SuyLuan.svg")

    d=Diagram("Vòng đời cảnh báo qua thao tác trên dashboard",560)
    for x,w,label in [(35,240,"Mới\nnew"),(405,270,"Đã tiếp nhận\nacknowledged"),(805,240,"Đã đóng\nclosed")]:
        d.box(x,170,w,85,label)
    d.arrow("275,212 405,212","Tiếp nhận",(340,150))
    d.arrow("675,212 805,212","Đóng",(740,150))
    d.arrow("155,255 155,355 925,355 925,255","Đóng trực tiếp",(540,340))
    d.text(540,90,"Cảnh báo mô phỏng mới → lưu SQLite với trạng thái new",18)
    d.text(540,420,"Tiếp nhận: cập nhật acknowledged_at. Đóng: cập nhật closed_at.\nCả hai lưu ghi chú và updated_at; giao diện không có thao tác mở lại.",17)
    d.text(540,505,"Upsert cùng alert_id giữ trạng thái, ghi chú và thời điểm xử lý đã lưu.",17)
    d.save(folder / "Hinh_3.4_VongDoi_CanhBao.svg")

    d=Diagram("Bố cục chức năng của hai chế độ dashboard",1010)
    left=[
        "Bộ lọc ngày / Xem toàn bộ",
        "4 KPI theo dữ liệu đã đánh giá",
        "Công suất  |  Phân bố dạng bất thường",
        "Điện áp  |  Phân bố theo giờ",
        "Bảng các điểm bất thường\nSố đo, điểm cảnh báo, loại, dấu hiệu"]
    right=[
        "Play / Pause / Bước tiếp / Khởi động lại\nTốc độ và tiến độ phát luồng",
        "4 KPI của phiên mô phỏng",
        "Thông báo warm-up / cảnh báo mới nhất",
        "Biểu đồ công suất  |  Biểu đồ điện áp\nTối đa 100 mẫu gần nhất",
        "Hàng đợi: lọc trạng thái và dạng gợi ý\nBảng cảnh báo / Tải CSV",
        "Chọn ID → chi tiết → ghi chú\nTiếp nhận / Đóng cảnh báo"]
    for x,title,labels in [(35,"PHÂN TÍCH LỊCH SỬ",left),(565,"GIÁM SÁT MÔ PHỎNG",right)]:
        d.text(x+240,88,title,21,True)
        for i,label in enumerate(labels):
            d.box(x,125+i*130,480,95,label,size=16)
    d.text(540,970,"Sơ đồ vùng chức năng theo mã giao diện hiện tại; không biểu diễn số liệu đo.",16)
    d.save(folder / "Hinh_3.5_Wireframe.svg")
