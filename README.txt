TỔNG QUAN DỰ ÁN: PHÁT HIỆN BẤT THƯỜNG ĐỒNG HỒ ĐIỆN THÔNG MINH (Smart
Meter Anomaly Detection System)

1.  Dự án giả lập một đồng hồ điện thông minh phát hiện lượng điện
 năng tiêu thụ bất thường hoạt động 24/7 : ví dụ khi điện năng tiêu 
thụ đạt ngưỡng cao bất thường, ban đêm 2-3h điện áp tang cao thì hệ 
thống sẽ báo đỏ 


2.  CÔNG NGHỆ SỬ DỤNG 

-   Ngôn ngữ lập trình: Python
-   Xử lý dữ liệu:
-   Pandas & NumPy: Dùng để đọc file, lọc làm sạch dữ liệu và tính
    toán các chỉ số điện năng
-   Mô hình AI / Học máy:
-   Scikit-learn: Isolation Forest và StandardScaler
-   Joblib: Dùng để đóng gói và lưu trữ “bộ não” AI sau khi học xong ra
    file .pkl.
-   Giao diện & Biểu đồ (Dashboard):
-   Streamlit: dựng nhanh một trang web ứng dụng 
-   Plotly: Vẽ các biểu đồ thông minh, cho phép rê chuột xem chi tiết
    từng điểm dữ liệu, phóng to thu nhỏ
-   Giả lập luồng dữ liệu (Streaming):
-   Sử dụng định dạng JSON Lines (.jsonl) kết hợp Python File I/O để
    đóng vai đồng hồ điện thật, liên tục gửi dữ liệu về màn hình theo
    thời gian thực (mỗi giây 1 mẫu).

3.  Cách mô hình hoạt động

-   Thuật toán cốt lõi: Isolation Forest (Rừng cô lập - Học không giám
    sát / Unsupervised).

-   Giải thích cách hoạt động

-   Dữ liệu dùng điện bình thường hay nằm tập trung thành từng nhóm
   lớn
-   Dữ liệu bất thường thì nằm rải rác

-   Thuật toán Isolation Forest sẽ liên tục cắt ngẫu nhiên dữ liệu ra
    thành các phần nhỏ. Điểm nào mà chỉ cần cắt vài lần đã bị cô lập
   thì thuật toán kết luận ngay là bất thường

-   Kỹ thuật tạo đặc trưng cho AI

-   Mã hóa giờ sin/cos: Giúp AI hiểu giờ 23:00 đêm và 00:00 sáng nằm
    ngay sát nhau trên vòng tròn thời gian (chứ không phải cách nhau 23
    đơn vị).

-   Lag features: Nhìn lại lượng điện tiêu thụ 1 giờ trước, 2
    giờ trước, và 24 giờ trước cùng giờ hôm qua

-   Rolling stats: Tính công suất trung bình và độ biến
    động trong 6 giờ gần nhất để làm mốc so sánh.

CÁC CHỨC NĂNG CHÍNH CỦA HỆ THỐNG Hệ thống được chia thành 5 bước
    
1.  Tiền xử lý dữ liệu :

-   Đọc dữ liệu gốc từ bộ dataset chuẩn của UCI (hơn 2 triệu dòng ghi
    nhận theo phút).
-   Tự động điền các khoảng trống dữ liệu bị thiếu.
-   Gom dữ liệu từ 1 phút thành 1 giờ để làm mượt đường biểu đồ và giảm
    tải tính toán.

2.  Huấn luyện bộ não AI:

-   Tạo ra các đặc trưng thông minh (giờ sin/cos, công suất quá khứ).
-   Cho mô hình Isolation Forest học thói quen dùng điện bình thường.
-   Lưu bộ não AI đã hoàn thiện vào thư mục models

3.  Bơm lỗi giả lập để test:

-   Do dữ liệu thật ít khi có nhãn lỗi, hệ thống tự động giả lập 3 loại
    sự cố thực tế:
-   Power Surge: Đột biến công suất gấp 3-5 lần (quá tải/chập điện).
-   Voltage Drop: Sụt áp điện ngẫu nhiên từ 20V - 40V (điện yếu).
-   Night Spike: Tiêu thụ điện cao bất thường vào ban đêm (từ 1h - 5h
    sáng).

4.  Giả lập thiết bị phát dữ liệu:

-   Đóng vai đồng hồ thông minh tại nhà dân, cứ 1 giây lại bắn 1 dòng dữ
    liệu điện vào hệ thống để màn hình theo dõi.

5.  Màn hình giám sát trung tâm:

-   Chế độ 1 - Phân tích lịch sử: Cho phép chọn khoảng ngày từ quá khứ
    để xem lại tổng quan, soi biểu đồ công suất, biểu đồ điện áp và
    thống kê loại lỗi.
-   Chế độ 2 - Giám sát thời gian thực: Có nút bấm Khởi động - Dừng -
    Tiếp tục. Dữ liệu chạy nhảy liên tục theo thời gian thực, khi phát
    hiện lỗi sẽ có thanh cảnh báo màu đỏ nhấp nháy ngay lập tức.
