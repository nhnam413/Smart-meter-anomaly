# Hệ Thống Phát Hiện Bất Thường Điện Năng (Smart Meter Anomaly Detection System)

Hệ thống phát hiện bất thường tiêu thụ điện năng theo thời gian thực sử dụng học máy không giám sát Isolation Forest, kết hợp trích xuất 9 đặc trưng chuỗi thời gian vật lý , cơ chế phân loại và giải thích nguyên nhân XAI, cùng giao diện dashboard trực quan bằng Streamlit và Plotly.

---

## 1. Tổng quan hệ thống

Hệ thống hoạt động như một trung tâm giám sát chỉ số đồng hồ điện thông minh theo chuỗi thời gian, phân tích hành vi phụ tải và cảnh báo sớm 3 dạng bất thường chính:

- Đột biến công suất : Công suất tiêu thụ tăng đột biến gấp 3 đến 5 lần mức tiêu thụ nền thông thường.
- Sụt điện áp : Điện áp lưới giảm đột ngột từ 20V đến 40V (dưới ngưỡng an toàn 220V).
- Bất thường ban đêm : Công suất tăng đột biến trong khung giờ thấp điểm từ 1h đến 5h sáng, cảnh báo nguy cơ rò rỉ điện hoặc tiêu thụ ngoài giờ sinh hoạt.

---

## 2. Công nghệ sử dụng

- Ngôn ngữ lập trình: Python 3.10 trở lên.
- Xử lý và biến đổi dữ liệu: Pandas (resample 1 giờ, rolling window), NumPy.
- Machine learning: Scikit-Learn (mô hình Isolation Forest, chuẩn hóa RobustScaler), Joblib .
- Trực quan hóa và Giao diện: Streamlit, Plotly.
- Định kiểu giao diện: CSS tùy chỉnh .

---

## 3. Cấu trúc dự án và Cấu hình hệ thống

### Cấu trúc thư mục

```
smart-meter-anomaly/
|-- data/
|   |-- raw/
|   |   `-- household_power_consumption.txt
|   |-- train_hourly.csv
|   |-- demo_stream.csv
|   `-- stream_buffer.jsonl
|-- models/
|   `-- model_bundle.pkl
|-- src/
|   |-- config.py
|   |-- features.py
|   |-- 01_data_prep.py
|   |-- 02_train.py
|   |-- 03_producer.py
|   |-- 04_dashboard.py
|   `-- style.css
|-- requirements.txt
`-- README.md
```

### Các tham số cấu hình chính config.py

- Cột mục tiêu TARGET_COL: Global_active_power (kW).
- Ngưỡng sụt điện áp VOLTAGE_DROP_THRESHOLD: -15.0V.
- Khoảng thời gian phát stream SEND_INTERVAL: 1.0 giây mỗi bản ghi.
- Số điểm hiển thị tối đa trên biểu đồ stream MAX_DISPLAY_POINTS: 100 điểm.

### 9 đặc trưng chuỗi thời gian

1. hour_sin, hour_cos: Mã hóa chu kỳ lượng giác 24 giờ trong ngày.
2. is_night: Biến nhị phân xác định khung giờ đêm (1h - 5h sáng).
3. power_diff_1h: Chênh lệch công suất so với 1 giờ trước.
4. power_dev_24h: Độ lệch tương đối so với cùng giờ ngày hôm trước.
5. power_zscore_6h: Z-Score công suất trong cửa sổ trượt 6 giờ.
6. voltage_diff_1h: Chênh lệch điện áp so với 1 giờ trước.
7. voltage_zscore_6h: Z-Score điện áp trong cửa sổ trượt 6 giờ.
8. power_factor: Hệ số công suất cos(phi) phản ánh hiệu quả sử dụng tải điện.

---

## 4. Ý nghĩa các biểu đồ và Thành phần giao diện

Dashboard gồm 2 chế độ chính: Phân tích lịch sử (Historical Analysis) và Giám sát thời gian thực (Real-Time Monitoring).

### Các biểu đồ trực quan

- Biểu đồ đường công suất tiêu thụ (kW): Hiển thị diễn biến công suất theo thời gian. Các điểm bất thường được đánh dấu rõ ràng bằng chấm tròn đỏ để người dùng nhận diện ngay thời điểm xảy ra sự cố.
- Biểu đồ đường điện áp (V): Hiển thị điện áp lưới điện kèm dải an toàn từ 220V đến 250V. Các điểm sụt điện áp ngoài ngưỡng an toàn được đánh dấu bằng chấm tròn đỏ.
- Biểu đồ Donut tỷ lệ loại bất thường: Thể hiện tỷ lệ phần trăm giữa các dạng sự cố Đột biến công suất, Sụt điện áp, Đột biến đêm. Tâm biểu đồ thể hiện tổng số vụ lỗi đã phát hiện.
- Bản đồ nhiệt Heatmap 24 giờ x 7 ngày: Phân bố mật độ bất thường theo 24 giờ trong ngày và 7 ngày trong tuần, giúp phát hiện các khung giờ và thứ trong tuần có mật độ sự cố tập trung cao nhất (sử dụng trong chế độ Lịch sử).
- Biểu đồ cột phân bố theo 24 giờ: Thống kê số lượng điểm bất thường theo từng giờ từ 00h đến 23h, làm nổi bật giờ có nhiều sự cố nhất sử dụng trong chế độ Thời gian thực.

### Các thành phần giao diện khác

- Thẻ chỉ số KPI: Tổng hợp 4 chỉ số nhanh gồm Tổng số mẫu, Tổng bất thường kèm tỷ lệ %, Khung giờ đỉnh lỗi và Công suất trung bình.
- Thanh trạng thái và Cảnh báo: Hiển thị trạng thái vận hành của lưới điện hoặc thông tin cảnh báo sự cố kèm giải thích nguyên nhân XAI
- Bảng nhật ký sự cố bất thường: Danh sách chi tiết các điểm sự cố gồm Thời gian, Công suất, Điện áp, Mức độ nghiêm trọng , Huy hiệu phân loại lỗi và Nguyên nhân chính được giải thích bởi XAI.

---

## 5. Kết quả đánh giá mô hình

Mô hình Isolation Forest được huấn luyện trên tập dữ liệu Train và đánh giá trên tập dữ liệu kiểm thử (demo_stream.csv gồm 6,834 mẫu):

- ROC-AUC Score: 0.9231 (khả năng phân tách xuất sắc giữa mẫu bình thường và bất thường).
- Precision: 74.43% (tỷ lệ cảnh báo đúng cao, hạn chế tối đa báo động giả).
- Recall: 72.84% (độ bao phủ tốt, bắt được hầu hết các điểm bất thường).
- F1-Score: 0.7363 (cân bằng giữa Precision và Recall).

Tỷ lệ phát hiện theo từng dạng lỗi:

- Đột biến công suất (Power Surge): 94.1%
- Sụt điện áp (Voltage Drop): 92.7%
- Đột biến ban đêm (Night Spike): 83.6%

---

## 6. Hướng dẫn cài đặt và Chạy hệ thống

### Bước 1: Cài đặt môi trường

```bash
cd smart-meter-anomaly
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu

```bash
python src/01_data_prep.py
```

Lệnh này sẽ đọc dữ liệu thô, làm sạch, resample 1 giờ, chia tập Train/Demo (80/20), giả lập tạo các dạng lỗi và xuất ra 2 file: data/train_hourly.csv và data/demo_stream.csv.

### Bước 3: Huấn luyện mô hình

```bash
python src/02_train.py
```

Lệnh này sẽ trích xuất 9 đặc trưng từ tập Train, huấn luyện mô hình Isolation Forest, tính thống kê baseline phục vụ XAI và lưu gói mô hình vào models/model_bundle.pkl.

### Bước 4: Khởi chạy Dashboard

```bash
python -m streamlit run smart-meter-anomaly/src/04_dashboard.py
```

Sau khi chạy lệnh, mở trình duyệt tại địa chỉ: `http://localhost:8501`.
