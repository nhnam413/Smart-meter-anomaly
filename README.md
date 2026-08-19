# 🔌 Smart Meter Anomaly Detection System

Hệ thống phát hiện điện năng tiêu thụ bất thường thời gian thực ứng dụng học máy không giám sát **Isolation Forest (Unsupervised Machine Learning)**, kết hợp trích xuất đặc trưng chuỗi thời gian chuyên sâu (Time-series Feature Engineering), cơ chế phân loại & giải thích nguyên nhân AI (Explainable AI), cùng giao diện giám sát trực quan hiện đại bằng **Streamlit & Plotly**.

---

## 🌟 Tổng Quan Dự Án

Hệ thống hoạt động như một trung tâm giám sát đồng hồ điện thông minh 24/7. Tự động thu thập dữ liệu tiêu thụ điện theo chuỗi thời gian, phân tích hành vi phụ tải, tính toán vi phân công suất và sụt áp, từ đó phát hiện sớm các nguy cơ rủi ro và bất thường về điện:

- ⚡ **Đột biến công suất (Power Surge)**: Công suất tiêu thụ tăng vọt từ 3x đến 5x so với mức nền bình thường do thiết bị công suất lớn hoặc quá tải đột ngột.
- 📉 **Sụt áp điện đột ngột (Voltage Drop)**: Điện áp giảm đột ngột từ 20V đến 40V (dưới ngưỡng 220V), đe dọa độ bền thiết bị và an toàn lưới điện.
- 🌙 **Bất thường ban đêm (Night Spike)**: Điện năng tăng đột biến (2x đến 3x) vào khung giờ thấp điểm (1h - 5h sáng), cảnh báo rò rỉ điện hoặc hành vi bất thường ngoài giờ sinh hoạt.

---

## 🛠 Công Nghệ Sử Dụng

| Thành Phần | Công Nghệ / Thư Viện | Vai Trò Trong Hệ Thống |
|---|---|---|
| **Ngôn ngữ** | Python 3.10+ | Môi trường lập trình chính, hỗ trợ xử lý luồng UTF-8 đa nền tảng |
| **Xử lý dữ liệu** | Pandas (≥2.0), NumPy (≥1.24) | Làm sạch, resample 1 giờ, rolling window & trích xuất đặc trưng |
| **Machine Learning** | Scikit-Learn (≥1.3), Joblib (≥1.3) | Mô hình `IsolationForest`, chuẩn hóa `RobustScaler`, serialization artifacts |
| **Dashboard UI** | Streamlit (≥1.30), Plotly (≥5.18) | Giao diện điều khiển tương tác cao, thiết kế F-Z Pattern, biểu đồ tương tác đa chiều |
| **Styling System** | Vanilla CSS (`src/style.css`) | Design system tùy chỉnh (Font Inter, tông màu tím `#6B4CE6`, cam `#EF7D32`, đỏ `#EF4444`) |
| **Real-time Stream** | JSON Lines (`.jsonl`) + File Pointer I/O | Giả lập phát luồng IoT thời gian thực, quản lý tiến trình ngầm qua `subprocess.Popen` |

---

## 📐 Kiến Trúc Pipeline Hệ Thống

```
+---------------------------------------------------------------------------------------------------+
|                                      DATA & ML PIPELINE FLOW                                      |
+---------------------------------------------------------------------------------------------------+

 [ Dữ liệu thô UCI ] 
  (1-minute interval)
          │
          ▼
 ┌──────────────────────┐
 │ 01_prepare_data.py   │ ──> Làm sạch, xử lý Missing Value & Resample 1h
 └──────────────────────┘     Phân tích EDA & tính ngưỡng Contamination qua IQR
          │                   Chia tập dữ liệu theo thứ tự thời gian: Train (70%), Test (20%), Demo (10%)
          ├───────────────────────────────┬───────────────────────────────┐
          ▼                               ▼                               ▼
    [ train.csv ]                   [ test.csv ]                    [ demo.csv ]
    [ contamination.txt ]                                                 │
          │                                                               │
          ▼                                                               ▼
 ┌──────────────────────┐                                       ┌────────────────────────┐
 │ 02_train_model.py    │                                       │ 03_inject_anomalies.py │
 └──────────────────────┘                                       └────────────────────────┘
          │                                                               │
          ▼ (Lưu Artifacts)                                               ▼ (Bơm nhãn Ground Truth)
 ┌──────────────────────────────────────┐                       ┌────────────────────────────┐
 │ models/                              │                       │ data/demo/                 │
 │  ├── isolation_forest_model.pkl      │                       │  └── demo_with_anomalies.csv│
 │  ├── scaler.pkl                      │                       └────────────────────────────┘
 │  └── feature_names.pkl               │                                     │
 └──────────────────────────────────────┘                                     ├────────────────────────┐
                                                                              ▼                        ▼
                                                                ┌────────────────────────┐  ┌──────────────────────┐
                                                                │ 04_producer.py         │  │ 06_evaluate_model.py │
                                                                │ (Real-time Streamer)   │  │ (Đánh giá hiệu năng) │
                                                                └────────────────────────┘  └──────────────────────┘
                                                                              │                        │
                                                                              ▼                        ▼
                                                                  [ stream_buffer.jsonl ]    [ reports/*.html, *.csv ]
                                                                              │
                                                                              ▼
                                                                ┌────────────────────────┐
                                                                │ 05_dashboard.py        │
                                                                │ (Streamlit Monitoring) │
                                                                └────────────────────────┘
```

### 📋 Ma Trận Hợp Đồng Dữ Liệu (Data Contract Matrix)

| Script | Input Artifacts | Output Artifacts | Nội dung xử lý chính |
|---|---|---|---|
| `01_prepare_data.py` | `data/raw/household_power_consumption.txt` | `data/processed/train.csv`<br>`data/processed/test.csv`<br>`data/demo/demo.csv`<br>`data/processed/contamination.txt` | Đọc dữ liệu phân cách `;`, điền khuyết `ffill`/`bfill`, resample 1h (mean), phân chia thời gian 70-20-10, tính Contamination qua IQR ($Q3 + 1.5 \times IQR$). |
| `02_train_model.py` | `data/processed/train.csv`<br>`data/processed/contamination.txt` | `models/isolation_forest_model.pkl`<br>`models/scaler.pkl`<br>`models/feature_names.pkl` | Tạo 13 đặc trưng qua `features.py`, fit `RobustScaler`, train `IsolationForest` (200 cây, `max_samples=512`, `max_features=0.5`). |
| `03_inject_anomalies.py` | `data/demo/demo.csv` | `data/demo/demo_with_anomalies.csv` | Bơm lỗi giả lập: Power Surge (3%), Voltage Drop (3%), Night Spike (2%) với `RANDOM_STATE=42`. |
| `04_producer.py` | `data/demo/demo_with_anomalies.csv` | `data/demo/stream_buffer.jsonl` | Đọc từng dòng CSV, phát tin nhắn JSONL tuần tự kèm timestamp và trường `sent_at`. Hỗ trợ tham số `--speed`, `--limit`, `--skip`, `--append`. |
| `06_evaluate_model.py` | `models/*.pkl`<br>`data/demo/demo_with_anomalies.csv` | `reports/evaluation_report.html`<br>`reports/metrics_summary.csv`<br>`reports/metrics_per_type.csv` | Đánh giá Precision, Recall, F1, ROC-AUC, PR-AUC và tỷ lệ phát hiện theo từng loại bất thường. Xuất báo cáo HTML tương tác và file CSV. |
| `05_dashboard.py` | `models/*.pkl`<br>`data/demo/demo_with_anomalies.csv`<br>`data/demo/stream_buffer.jsonl` | Giao diện Streamlit Dashboard | Giám sát 2 chế độ: Khám phá lịch sử (3,417 mẫu) & Giám sát luồng Realtime, tích hợp giải thích AI và phân loại theo luật. |

---

## ⚙️ Thiết Kế Kỹ Thuật (Feature Engineering)

Module `src/features.py` là **Single Source of Truth** duy nhất cho toàn bộ hệ thống (dùng chung cho cả huấn luyện offline, đánh giá và luồng dự đoán online realtime):

```
Raw Cảm Biến ──> Loại bỏ cột đa cộng tuyến / dư thừa (Global_intensity, Sub_metering_1..3)
             ──> Chu kỳ thời gian 24h: hour_sin, hour_cos
             ──> Độ trễ & Vi phân: power_lag_1h, power_diff_1h, power_lag_24h, voltage_lag_1h, voltage_diff_1h
             ──> Chênh lệch tương đối 24h: power_deviation_24h = (P - P_lag24) / (|P_lag24| + eps)
             ──> Z-Score chuẩn hóa (Rolling 6h): power_zscore_6h, voltage_zscore_6h = (val - mean_6h) / (std_6h + eps)
             ──> Công suất phản kháng: reactive_ratio = clip(Reactive / (|Active| + eps), -10, 10)
             ──> Ngữ cảnh giờ đêm: power_hourly_diff (lệch TB giờ), is_night (1 nếu 1h <= hour <= 5h)
```

### Danh sách 13 đặc trưng tối ưu:
1. `hour_sin`, `hour_cos`: Mã hóa chu kỳ vòng tròn 24h bằng hàm lượng giác.
2. `power_lag_1h`, `power_diff_1h`: Độ trễ và vi phân công suất 1 giờ trước — bắt nhanh các xung nhọn công suất (*Power Surge*).
3. `power_lag_24h`: Công suất cùng giờ của ngày hôm trước để nắm bắt chu kỳ ngày đêm.
4. `power_deviation_24h`: Tỷ lệ thay đổi tương đối so với cùng giờ hôm trước, giúp loại bỏ báo động giả do thói quen sinh hoạt.
5. `power_zscore_6h`: Z-Score công suất cục bộ trong cửa sổ trượt 6 giờ, phát hiện bất thường độc lập với biên độ tuyệt đối.
6. `voltage_lag_1h`, `voltage_diff_1h`: Độ trễ và vi phân điện áp 1 giờ trước — bắt nhanh sự cố sụt áp (*Voltage Drop*).
7. `voltage_zscore_6h`: Z-Score điện áp cục bộ trong cửa sổ trượt 6 giờ.
8. `reactive_ratio`: Tỷ lệ giữa công suất phản kháng và công suất thực (được cắt giới hạn $\pm 10$) — phát hiện tải cảm kháng bất thường.
9. `power_hourly_diff`: Độ lệch công suất so với trung bình của chính khung giờ đó trong ngày.
10. `is_night`: Biến nhị phân đánh dấu khung giờ đêm (1h – 5h sáng) — giúp mô hình nhạy bén hơn với các bất thường ban đêm (*Night Spike*).

> [!NOTE]
> **Loại bỏ Đa cộng tuyến**: Cột `Global_intensity` có hệ số tương quan tuyến tính $r = +0.9992$ với `Global_active_power`, do đó đã được loại bỏ khỏi không gian đặc trưng để tránh làm phân mảnh và suy giảm hiệu quả phân tách của cây quyết định trong Isolation Forest.

> [!IMPORTANT]
> **Nguyên tắc Real-time Buffer**: Hàm `create_features_realtime(buffer_df)` yêu cầu tối thiểu **25 mẫu** dữ liệu trong buffer để tính toán chính xác `power_lag_24h` và thống kê rolling 6h, đảm bảo tính nhất quán tuyệt đối giữa môi trường online và offline.

---

## 🏷️ Phân Loại Bất Thường & Giải Thích AI (Explainable AI)

Sau khi **Isolation Forest** phát hiện một điểm là bất thường (`decision_function < 0` / nhãn `-1`), module `src/classify.py` thực hiện 2 bước hậu xử lý chuyên sâu:

### 1. Phân loại theo luật chuyên ngành (Rule-based Classification)
- **Sụt áp điện (`voltage_drop`)**: Khi vi phân điện áp $\text{voltage\_diff\_1h} < -15.0\text{V}$.
- **Đột biến đêm (`night_spike`)**: Khi xảy ra trong giờ đêm ($\text{is\_night} = 1$) VÀ ($\text{power\_zscore\_6h} > 1.0$ HOẶC $\text{power\_deviation\_24h} > 1.5$).
- **Đột biến công suất (`power_surge`)**: Khi xảy ra ngoài giờ đêm ($\text{is\_night} = 0$) VÀ ($\text{power\_zscore\_6h} > 1.5$ HOẶC $\text{power\_deviation\_24h} > 1.5$).
- **Chưa xác định (`unknown`)**: Điểm dị biệt nằm ở vùng biên không gian đặc trưng không thuộc 3 nhóm trên.

### 2. Giải thích đóng góp đặc trưng (Feature Attribution)
- Tính độ lệch của từng đặc trưng của điểm bất thường so với phân phối của tập dữ liệu bình thường:
  $$\text{score} = \frac{|x - \text{median}|}{\text{IQR}}$$
- Trích xuất **Top 3 đặc trưng** có mức độ sai lệch lớn nhất (score > 0.5) và hiển thị trực tiếp nguyên nhân trên bảng cảnh báo của Dashboard (ví dụ: `Z-Score công suất 6h=+3.42, Vi phân công suất=+2.81 kW`).

---

## 📊 Kết Quả Đánh Giá Hiệu Năng Mô Hình

Đánh giá thực nghiệm độc lập trên tập Demo (3,417 mẫu — bao gồm 270 mẫu bất thường giả lập) được tạo ra bởi script `06_evaluate_model.py`:

### 1. Bảng chỉ số tổng quan (Overall Metrics)

| Chỉ Số Đánh Giá | Giá Trị Thực Nghiệm | Ý Nghĩa / Nhận Xét |
|---|---|---|
| **ROC-AUC** | **0.90** | Khả năng phân tách tổng thể giữa mẫu bình thường và bất thường đạt mức xuất sắc |
| **PR-AUC (Average Precision)** | **0.58** | Hiệu năng phân định trên tập dữ liệu mất cân bằng cao |
| **Precision** | **0.47** | Tỷ lệ cảnh báo thực tế chính xác trong số các điểm model gán nhãn bất thường |
| **Recall** | **0.66** | Tỷ lệ phát hiện được 179 / 270 điểm bất thường thực tế |
| **F1-Score** | **0.55** | Trung hòa hài hòa giữa Precision và Recall |
| **True Positives (TP)** | 179 mẫu | Số điểm bất thường phát hiện chính xác |
| **False Positives (FP)** | 201 mẫu | Điểm bình thường nhưng có biến động lớn bị cảnh báo |
| **True Negatives (TN)** | 2,922 mẫu | Điểm bình thường nhận diện chính xác |
| **False Negatives (FN)** | 91 mẫu | Điểm bất thường bị bỏ sót |

### 2. Tỷ lệ phát hiện theo loại bất thường (Detection Rate by Type)

| Loại Bất Thường | Ground Truth | Đã Phát Hiện | Bỏ Sót | Tỷ Lệ Phát Hiện (Detection Rate) |
|---|---|---|---|---|
| 📉 **Sụt áp điện (Voltage Drop)** | 101 mẫu | 100 mẫu | 1 mẫu | **99.0%** |
| ⚡ **Đột biến công suất (Power Surge)** | 102 mẫu | 55 mẫu | 47 mẫu | **53.9%** |
| 🌙 **Bất thường ban đêm (Night Spike)** | 67 mẫu | 24 mẫu | 43 mẫu | **35.8%** |

> Báo cáo đánh giá trực quan đầy đủ bao gồm Confusion Matrix, Đường cong Precision-Recall, Phân phối Score và Biểu đồ phân nhóm được lưu tại `reports/evaluation_report.html`.

---

## 🎨 Điểm Nổi Bật Trên Dashboard Giám Sát (`05_dashboard.py`)

Dashboard được xây dựng dựa trên triết lý **Modern Light Design System** với cấu trúc **F-Z Pattern Layout** không sử dụng sidebar, tối đa hóa không gian hiển thị thông tin:

### 1. Chế độ Phân tích Lịch sử (Historical Analysis Mode)
- **Tải toàn diện**: Khám phá toàn bộ 3,417 mẫu dữ liệu demo (từ tháng 07/2010 đến 11/2010).
- **Bộ lọc thời gian 1-Click nhanh**: Các nút chọn tức thời `[Toàn bộ]`, `[T7/2010]`, `[T8/2010]`, `[T9/2010]`, `[T10/2010]`, `[T11/2010]` kèm Date Input `DD/MM/YYYY`.
- **Biểu đồ công suất & điện áp tương tác**: Hiển thị đường công suất kết hợp các điểm đánh dấu đỏ/cam cho anomaly, tooltip chi tiết điểm số dị biệt (Unified Hover).
- **Vùng điện áp an toàn**: Dải sáng xanh đánh dấu ngưỡng chuẩn an toàn lưới điện (220V – 250V).
- **Bảng chi tiết Anomaly & Lọc thông minh**: Cho phép lọc theo từng loại bất thường (Đột biến công suất / Sụt áp / Bất thường đêm / Chưa xác định) và hiển thị cột **Nguyên nhân chính (AI Explanation)**.
- **Xuất báo cáo CSV**: Nút download file CSV có hỗ trợ encoding UTF-8 BOM (`utf-8-sig`) tương thích hoàn hảo với Microsoft Excel.
- **Anomaly Timeline Heatmap**: Bản đồ nhiệt 2D (Giờ × Ngày) trực quan hóa mật độ xuất hiện bất thường theo thời gian.
- **Khối tóm tắt hiệu năng mô hình**: Tích hợp trực tiếp các chỉ số Precision, Recall, F1, ROC-AUC và bảng phân rã phát hiện ngay trên giao diện.

### 2. Chế độ Giám sát Thời gian thực (Real-time Stream Mode)
- **Bộ điều khiển Producer tích hợp**: Các nút `Khởi động`, `Dừng`, `Tiếp tục` quản lý tiến trình nền Python độc lập (`subprocess.Popen`), hiển thị trạng thái đèn tín hiệu (Online/Offline) và mã PID.
- **Thanh cảnh báo tức thời (Live Alert Bar)**: Tự động xuất hiện banner cảnh báo đỏ viền đậm nổi bật ngay khi điểm dữ liệu mới nhất được xác định là bất thường.
- **Xử lý luồng tối ưu (File Pointer Tracking)**: Sử dụng kỹ thuật `f.seek()` và `f.tell()` để chỉ đọc các dòng mới ghi vào `stream_buffer.jsonl` mà không cần reload lại file.
- **Thẻ KPI thời gian thực**: Cập nhật tổng số bản tin nhận, số lượng anomaly, tỷ lệ %, delta tăng trưởng và công suất tức thời.
- **Bảng Log Anomaly trực tiếp**: Lưu trữ và hiển thị 30 điểm bất thường gần nhất theo thời gian thực.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
Smart-meter-anomaly/
├── .streamlit/
│   └── config.toml               # Cấu hình theme sáng, headless server & font Streamlit
├── data/
│   ├── raw/                      # Chứa dataset UCI gốc (household_power_consumption.txt)
│   ├── processed/                # Chứa dữ liệu đã xử lý: train.csv, test.csv, contamination.txt
│   └── demo/                     # Chứa demo.csv, demo_with_anomalies.csv, stream_buffer.jsonl
├── models/
│   ├── isolation_forest_model.pkl# Artifact mô hình Isolation Forest đã huấn luyện
│   ├── scaler.pkl                # Artifact bộ chuẩn hóa RobustScaler
│   └── feature_names.pkl         # Danh sách tên các đặc trưng đã trích xuất
├── reports/
│   ├── evaluation_report.html    # Báo cáo đánh giá tương tác trực quan 4-trong-1 (Plotly)
│   ├── metrics_summary.csv       # Bảng tổng hợp các chỉ số đánh giá chính
│   └── metrics_per_type.csv      # Bảng chi tiết tỷ lệ phát hiện theo từng loại bất thường
├── src/
│   ├── config.py                 # Cấu hình tập trung (Đường dẫn, Hyperparameters, Design Tokens)
│   ├── features.py               # Feature Engineering (Single Source of Truth)
│   ├── classify.py               # Phân loại bất thường theo luật & Giải thích đặc trưng (XAI)
│   ├── 01_prepare_data.py        # Tiền xử lý dữ liệu UCI & Phân tích EDA tính Contamination
│   ├── 02_train_model.py         # Huấn luyện mô hình Isolation Forest & RobustScaler
│   ├── 03_inject_anomalies.py    # Giả lập bơm 3 loại lỗi bất thường vào tập Demo
│   ├── 04_producer.py            # Trình phát luồng dữ liệu giả lập JSONL thời gian thực
│   ├── 05_dashboard.py           # Dashboard chính Streamlit giám sát Lịch sử & Realtime
│   ├── 06_evaluate_model.py      # Đánh giá chỉ số mô hình (Precision, Recall, F1, ROC-AUC)
│   └── style.css                 # Design System CSS tùy chỉnh cho giao diện Streamlit
├── requirements.txt              # Danh sách thư viện phụ thuộc của dự án
├── AGENTS.md                     # Tài liệu quy chuẩn kỹ thuật dành cho AI Coding Agents
└── README.md                     # Tài liệu hướng dẫn chi tiết dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Vận Hành

### 1. Yêu cầu môi trường & Cài đặt thư viện

Khuyến nghị sử dụng Python 3.10 trở lên trong một môi trường ảo (`venv` hoặc `conda`):

```bash
# Tạo và kích hoạt môi trường ảo (tùy chọn)
python -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu gốc

1. Tải dataset từ UCI Machine Learning Repository: [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235)
2. Giải nén và đặt file `household_power_consumption.txt` vào thư mục:
   ```
   Smart-meter-anomaly/data/raw/household_power_consumption.txt
   ```

### 3. Thực thi Pipeline xử lý & Huấn luyện mô hình

Chạy tuần tự các bước trong pipeline:

```bash
# Bước 1: Làm sạch dữ liệu, resample 1 giờ, tính Contamination và chia tập Train/Test/Demo
python src/01_prepare_data.py

# Bước 2: Trích xuất 13 đặc trưng và huấn luyện mô hình Isolation Forest
python src/02_train_model.py

# Bước 3: Giả lập bơm 3 loại bất thường (Power Surge, Voltage Drop, Night Spike) vào tập Demo
python src/03_inject_anomalies.py

# Bước 4: Đánh giá mô hình và sinh báo cáo hiệu năng (HTML & CSV)
python src/06_evaluate_model.py
```

### 4. Khởi chạy Dashboard giám sát

Khởi chạy ứng dụng Streamlit Dashboard:

```bash
# Khuyến nghị sử dụng cú pháp python -m trên Windows
python -m streamlit run src/05_dashboard.py
```

Trình duyệt sẽ tự động mở địa chỉ `http://localhost:8501`. Tại đây bạn có thể:
- Xem và lọc dữ liệu lịch sử trên tab **Phân tích lịch sử**.
- Chuyển sang tab **Giám sát thời gian thực** và bấm nút **Khởi động** để theo dõi luồng phát dữ liệu tự động.

*(Tùy chọn) Khởi chạy Producer phát luồng độc lập qua Terminal riêng:*
```bash
# Phát luồng với tốc độ 0.5 giây / 1 bản tin
python src/04_producer.py --speed 0.5

# Các tham số hỗ trợ:
# --speed : Tốc độ phát (giây/bản tin, mặc định: 1.0)
# --limit : Giới hạn số lượng bản tin cần phát
# --skip  : Bỏ qua N bản tin đầu tiên
# --append: Ghi nối tiếp vào file buffer hiện tại thay vì ghi đè
```

---

## 🛡️ Nguyên Tắc Kỹ Thuật Bắt Buộc (Guardrails)

1. **Không Shuffle chuỗi thời gian**: Toàn bộ quá trình phân chia tập dữ liệu Train / Test / Demo tuân thủ phân tách tuyến tính theo dòng thời gian (`iloc[:split]`).
2. **Không phân mảnh Feature Engineering**: Mọi logic biến đổi đặc trưng bắt buộc gọi từ `src/features.py` để tránh hiện tượng sai lệch huấn luyện - suy luận (Train-Serve Skew).
3. **Cố định Random Seed**: Toàn bộ thuật toán ngẫu nhiên (bơm lỗi, Isolation Forest) được ghim cố định `RANDOM_STATE = 42` để đảm bảo tính tái lập kết quả (Reproducibility).
4. **An toàn Encoding trên Windows**: Mọi script CLI đều có cơ chế cấu hình `sys.stdout` UTF-8 chống lỗi font khi in log tiếng Việt trên Command Prompt/PowerShell.

---

## 📄 Bản Quyền & Giấy Phép

Dự án được phát triển nhằm mục đích nghiên cứu, học tập và ứng dụng hệ thống phát hiện bất thường thông minh trong ngành năng lượng điện tử. Dữ liệu gốc thuộc bản quyền của [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235).

© 2026 Smart Meter Anomaly Detection System.