# 🔌 Smart Meter Anomaly Detection System

Hệ thống phát hiện điện năng tiêu thụ bất thường thời gian thực ứng dụng học máy không giám sát **Isolation Forest (Unsupervised Machine Learning)**, kết hợp giao diện giám sát trực quan bằng **Streamlit** & **Plotly**.

---

## 🌟 Tổng Quan Dự Án

Hệ thống đóng vai trò như một bộ giám sát đồng hồ điện thông minh 24/7. Hệ thống tự động phân tích hành vi tiêu thụ điện, tính toán các điểm đặc trưng vi phân công suất & sụt áp, từ đó cảnh báo tức thời các rủi ro nguy hiểm:
- **Đột biến công suất (Power Surge)**: Tiêu thụ vượt ngưỡng bình thường từ 3x đến 5x.
- **Sụt áp điện đột ngột (Voltage Drop)**: Điện áp giảm mạnh từ 20V đến 40V, đe dọa thiết bị điện.
- **Tăng tải ban đêm (Night Spike)**: Điện năng tăng đột biến vào khung giờ thấp điểm (1h - 5h sáng).

---

## 🛠 Công Nghệ Sử Dụng

| Thành Phần | Công Nghệ / Thư Viện | Mô Tả |
|---|---|---|
| **Ngôn ngữ** | Python 3.10+ | Môi trường lập trình chính |
| **Xử lý dữ liệu** | Pandas, NumPy | Tiền xử lý, resample 1h & tính toán đặc trưng |
| **Machine Learning** | Scikit-Learn, Joblib | Isolation Forest, RobustScaler, lưu trữ artifacts |
| **Dashboard UI** | Streamlit, Plotly | Giao diện F-Z Pattern, biểu đồ tương tác cao |
| **Styling** | Vanilla CSS (`style.css`) | Design system tông sáng (Inter font, Purple `#6B4CE6` & Orange `#EF7D32`) |
| **Real-time Stream** | JSON Lines (`.jsonl`) + File I/O | Giả lập luồng truyền dữ liệu IoT thời gian thực |

---

## 📐 Kiến Trúc Pipeline Hệ Thống

```
+---------------------+     +--------------------+     +------------------------+
| 01_prepare_data.py  | --> | 02_train_model.py  | --> | 03_inject_anomalies.py |
| (Clean & Resample)  |     | (Train Model AI)   |     | (Giả lập lỗi Demo)     |
+---------------------+     +--------------------+     +------------------------+
                                                                   |
                                          +------------------------+------------------------+
                                          |                                                 |
                                          v                                                 v
                               +--------------------+                            +----------------------+
                               | 04_producer.py     |                            | 06_evaluate_model.py |
                               | (Terminal Stream)  |                            | (Đánh giá Metrics)   |
                               +--------------------+                            +----------------------+
                                          |
                                          v (stream_buffer.jsonl)
                               +--------------------+
                               | 05_dashboard.py    |
                               | (Streamlit App)    |
                               +--------------------+
```

---

## 🚀 Hướng Dẫn Cài Đặt & Vận Hành

### 1. Yêu cầu môi trường
Đảm bảo đã cài đặt Python 3.10 trở lên. Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu gốc
- Tải dataset UCI: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235)
- Giải nén file `household_power_consumption.txt` vào thư mục: `data/raw/`

### 3. Thực thi Pipeline xử lý & Huấn luyện mô hình

```bash
# Bước 1: Tiền xử lý dữ liệu thô (Resample 1h & EDA outlier)
python src/01_prepare_data.py

# Bước 2: Trích xuất đặc trưng & Huấn luyện Isolation Forest
python src/02_train_model.py

# Bước 3: Giả lập bơm lỗi bất thường vào tập dữ liệu Demo
python src/03_inject_anomalies.py

# Bước 4: Đánh giá hiệu năng mô hình (Precision, Recall, F1, ROC-AUC)
python src/06_evaluate_model.py
```

### 4. Khởi chạy Dashboard giám sát

Bạn có thể chạy trực tiếp **Dashboard** hoặc chạy kết hợp luồng **Producer giả lập**:

```bash
# Khởi chạy giao diện chính Streamlit Dashboard
streamlit run src/05_dashboard.py
```

*(Tùy chọn) Để phát luồng thời gian thực từ Terminal riêng:*
```bash
python src/04_producer.py --speed 0.5
```

---

## ⚙️ Thiết Kế Kỹ Thuật (Feature Engineering)

Mô hình **Isolation Forest** được huấn luyện trên các đặc trưng được thiết kế chuyên biệt (nằm trong `src/features.py` — Single Source of Truth):

1. **Chu kỳ thời gian**: `hour_sin`, `hour_cos` (Mã hóa vòng tròn 24h).
2. **Lag Features & Vi phân**:
   - `power_lag_1h`, `power_diff_1h` (Vi phân công suất tức thời giúp bắt lỗi *Power Surge*).
   - `voltage_lag_1h`, `voltage_diff_1h` (Vi phân sụt áp giúp bắt lỗi *Voltage Drop*).
   - `power_lag_24h` (So sánh công suất cùng giờ ngày hôm trước).
3. **Thống kê cuộn (Rolling 6h)**: `power_rolling_mean_6h`, `power_rolling_std_6h`, `voltage_rolling_mean_6h`, `voltage_rolling_std_6h`.
4. **Ngữ cảnh giờ đêm**: `night_power_spike` (Độ lệch công suất giờ đêm từ 1h - 5h AM).
5. **Loại bỏ đa cộng tuyến**: Loại bỏ `Global_intensity` ($r = +0.9992$ với `Global_active_power`) để tránh làm nhiễu cây quyết định của Isolation Forest.

---

## 🎨 Điểm Nổi Bật Trên Dashboard (`05_dashboard.py`)

- **Bố cục F-Z Pattern hiện đại**: Màn hình trải dài không sử dụng sidebar che khuất, tập trung vào thanh điều khiển trên cùng (Top Control Bar).
- **Phân tích lịch sử mượt mà**:
  - Tự động load đầy đủ dữ liệu demo từ **28/06/2010 đến 26/11/2010** (3,417 mẫu).
  - Hỗ trợ các nút **Chọn nhanh mốc thời gian 1-click**: `[Toàn bộ]`, `[T7/2010]`, `[T8/2010]`, `[T9/2010]`, `[T10/2010]`, `[T11/2010]`.
  - Ô chọn lịch tùy chỉnh định dạng `DD/MM/YYYY` với giao diện sáng (Light Mode), hiển thị số ngày rõ nét 100%.
- **Giám sát thời gian thực (Real-time)**:
  - Tích hợp bộ điều khiển Producer trực tiếp trên giao diện (`Khởi động`, `Dừng`, `Tiếp tục`).
  - Ô chỉnh thời gian cập nhật giây gọn gàng với thiết kế phẳng viền trắng tinh tế.
  - Thanh cảnh báo đỏ nhấp nháy ngay khi phát hiện điểm bất thường mới nhất.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
smart-meter-anomaly/
├── src/
│   ├── config.py               # Cấu hình tập trung (Đường dẫn, Hyperparameters, Bảng màu)
│   ├── features.py             # Feature Engineering (Single Source of Truth)
│   ├── 01_prepare_data.py      # Tiền xử lý dữ liệu UCI & Phân tích EDA
│   ├── 02_train_model.py       # Huấn luyện Isolation Forest & RobustScaler
│   ├── 03_inject_anomalies.py  # Giả lập bơm lỗi (Power Surge, Voltage Drop, Night Spike)
│   ├── 04_producer.py          # Trình phát luồng dữ liệu giả lập JSONL
│   ├── 05_dashboard.py         # Dashboard chính Streamlit
│   ├── 06_evaluate_model.py    # Đánh giá chỉ số mô hình (Precision, Recall, F1, ROC-AUC)
│   └── style.css               # Design System CSS tùy chỉnh
├── data/
│   ├── raw/                    # Chứa dataset UCI gốc (household_power_consumption.txt)
│   ├── processed/              # Chứa train.csv, test.csv, contamination.txt
│   └── demo/                   # Chứa demo.csv, demo_with_anomalies.csv, stream_buffer.jsonl
├── models/                     # Thư mục chứa artifacts (.pkl)
├── reports/                    # Báo cáo đánh giá hiệu năng mô hình
├── requirements.txt            # Thư viện phụ thuộc
├── AGENTS.md                   # Hướng dẫn quy chuẩn cho AI Coding Agents
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

© 2026 Smart Meter Anomaly Detection System.