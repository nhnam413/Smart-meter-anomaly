# Báo Cáo Kỹ Thuật Nghiệm Thu: Smart Meter Anomaly Detection

## 1. Lý Thuyết Cơ Bản Về Isolation Forest

Mô hình lõi của hệ thống sử dụng thuật toán học máy không giám sát **Isolation Forest (Rừng Cô Lập)**. Khác với các thuật toán truyền thống cố gắng mô hình hóa vùng dữ liệu "bình thường", Isolation Forest đi thẳng vào việc cô lập các điểm "bất thường" dựa trên hai đặc tính nội tại của chúng:
- **Số lượng ít (Minority):** Dữ liệu bất thường chiếm tỷ lệ rất nhỏ.
- **Khác biệt lớn (Different):** Giá trị của dữ liệu bất thường nằm cách xa cụm dữ liệu bình thường.

**Nguyên lý hoạt động:**
Thuật toán xây dựng nhiều cây nhị phân (Isolation Trees) bằng cách chọn ngẫu nhiên một đặc trưng và một điểm cắt ngẫu nhiên. Vì các điểm bất thường nằm xa trung tâm và rải rác, chúng sẽ bị cô lập tách ra thành một nút lá (leaf node) rất nhanh chỉ sau một vài lần cắt. Ngược lại, để tách các điểm bình thường nằm sâu trong cụm, thuật toán phải chia cắt rất nhiều lần. 
- Chiều dài đường đi từ rễ đến lá càng **ngắn**, điểm đó càng có khả năng cao là **bất thường**.
- Chiều dài đường đi càng **dài**, điểm đó càng có khả năng là **bình thường**.

**Tại sao chọn Isolation Forest?**
- Không yêu cầu dữ liệu phải dán nhãn trước (phù hợp với thực tế dữ liệu điện lưới).
- Tốc độ suy diễn (inference) cực nhanh và tốn ít RAM, dễ dàng triển khai ở thiết bị Edge (IoT).

---

## 2. Kiến Trúc Hệ Thống & Data Pipeline

Hệ thống được thiết kế theo dạng luồng (pipeline) liên tục, cho phép xử lý dữ liệu Streaming thời gian thực:

1. **Sliding Window Buffer:** Hệ thống duy trì một bộ đệm cửa sổ trượt (trạng thái lưu giữ 25-50 bản ghi gần nhất).
2. **Feature Engineering (The Sharp 9):** Chuyển đổi dữ liệu điện thô thành 9 đặc trưng chuỗi thời gian (bao gồm độ lệch giờ, ngày, `is_night`, và Z-Score 6 giờ).
3. **RobustScaler:** Chuẩn hóa dữ liệu bằng Median và IQR (thay vì Mean và Std) để tránh hiện tượng các nhiễu lớn (outliers cực đoan) bóp méo không gian phân phối (Masking effect).
4. **Isolation Forest Model:** Đưa ra `decision_score`, sau đó chuyển thành hàm `% Severity` (độ nghiêm trọng).
5. **XAI (Explainable AI):** Nếu sự cố xảy ra, hệ thống tra ngược lại 9 đặc trưng để giải thích nguyên nhân gây ra cảnh báo.

*Khả năng triển khai:* Toàn bộ pipeline, scaler, và model được đóng gói thành file `model_bundle.pkl` chỉ vỏn vẹn ~4.3MB, đáp ứng tốt cho việc tích hợp vào các hệ thống nhúng hoặc web server nhỏ gọn.

---

## 3. Đánh Giá Baseline Tổng Quan và Explainable AI (XAI)

Dựa trên cấu hình `n_estimators=200`, `contamination=0.08` và `max_samples=512`, mô hình baseline đạt được các chỉ số trên tập kiểm thử (Demo):

| Metric | Giá trị | | Metric | Giá trị |
|--------|---------|---|--------|---------|
| **ROC-AUC** | 0.90 | | **True Positives** | 179 |
| **PR-AUC** | 0.58 | | **False Positives** | 201 |
| **Precision** | 0.47 | | **True Negatives** | 2922 |
| **Recall** | 0.66 | | **False Negatives** | 91 |

**Explainable AI (Khả năng giải thích):**
Thay vì đưa ra phán đoán "Hộp đen", hệ thống sử dụng Median và IQR của tập Train để làm Baseline. Khi có một điểm bất thường, hệ thống tính toán khoảng cách của 9 đặc trưng so với Baseline. Đặc trưng nào có độ lệch chuẩn hóa (Deviation) cao nhất sẽ được báo cáo ra UI (VD: *"Phát hiện bất thường do: Biến động công suất 1 giờ = +4.21, Bất thường công suất 6h = +3.85"*). Điều này giúp kỹ sư vận hành đưa ra quyết định xử lý chính xác hơn.

---

## 4. Root Cause Analysis: Tại Sao Lại Bỏ Lọt Lỗi `night_spike`?

Khảo sát sâu vào các loại lỗi cụ thể, chúng ta thấy sự chênh lệch lớn trong khả năng nhận diện:

| Loại Bất Thường | Số Lượng Lỗi | Phát Hiện | Bỏ Lọt (Missed) | Recall (%) |
|-----------------|--------------|-----------|-----------------|------------|
| Sụt điện áp (`voltage_drop`) | 205 | 193 | 12 | **94.1%** |
| Đột biến công suất (`power_surge`) | 205 | 127 | 78 | **62.0%** |
| Đột biến đêm (`night_spike`) | 135 | 77 | 58 | **57.0%** |

**Nguyên nhân gốc rễ (Root Cause) khiến thuật toán bỏ lọt `night_spike`:**
1. **Bản chất của tải nền ban đêm:** Dữ liệu ban đêm từ 1h-5h có tải nền rất thấp (ví dụ 0.2kW). Khi xảy ra sự cố `night_spike`, công suất có thể tăng gấp 3 lần, lên mức 0.6kW. Mặc dù đây là bất thường cục bộ nghiêm trọng, nhưng xét trên phân phối toàn cục của cả ngày (với những đỉnh tải sinh hoạt 3kW-5kW), con số 0.6kW vẫn chìm sâu vào vùng lõi dữ liệu bình thường toàn cục.
2. **Điểm mù của Isolation Forest:** Do cắt không gian ngẫu nhiên trên toàn bộ tập dữ liệu, mô hình cần rất nhiều nhát cắt mới cô lập được điểm 0.6kW này (chiều dài đường đi dài). Thuật toán đã nhầm tưởng đây là điểm bình thường do giá trị tuyệt đối nhỏ.
3. **Tham số `contamination`:** Với `contamination=0.08`, mô hình buộc phải ưu tiên giữ lại các điểm dễ cô lập nhất (các lỗi bạo lực, biên độ lớn như `voltage_drop`). Các bất thường biên độ nhỏ bị văng ra khỏi danh sách 8% này.

---

## 5. Đề Xuất Chiến Lược Cải Thiện (Next Steps)

Để khắc phục điểm mù của Baseline hiện tại, báo cáo đề xuất các hướng cải tiến tiếp theo cho Data Science Team:
1. **Phân tách Model theo Context (Ensemble):** Tách riêng làm 2 Isolation Forest (một model huấn luyện chuyên biệt cho ban ngày, một model chuyên biệt cho tải nền ban đêm).
2. **Chuyển đổi sang mô hình Local Outlier:** Thử nghiệm thuật toán `Local Outlier Factor (LOF)` để tính mật độ điểm dữ liệu theo cụm lân cận thay vì toàn cục.
3. **Mô hình chuỗi thời gian Deep Learning:** Chuyển sang kiến trúc `LSTM-Autoencoder`. Mô hình này sẽ học quy luật sóng hình sin của ngày/đêm và cảnh báo dựa trên lỗi tái cấu trúc (Reconstruction Error), giải quyết triệt để vấn đề biên độ tuyệt đối nhỏ của `night_spike`.
