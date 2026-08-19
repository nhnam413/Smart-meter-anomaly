# 🔌 Smart Meter Anomaly Detection — Agent Context & Rules

Tài liệu này là **Ground Truth Context & Development Protocol** dành cho AI Coding Assistants và Developers khi làm việc trên repository này. Mọi thay đổi code, kiến trúc hoặc dữ liệu đều phải tuân thủ nghiêm ngặt các quy ước dưới đây.

---

## 🛠 1. Tech Stack & Environment

- **Core**: Python 3.10+ (Yêu cầu xử lý UTF-8 stream trên Windows)
- **Data Engine**: Pandas ≥2.0, NumPy ≥1.24
- **Machine Learning**: Scikit-Learn ≥1.3 (`IsolationForest`, `RobustScaler`), Joblib ≥1.3
- **Visualization & UI**: Streamlit ≥1.30, Plotly ≥5.18, Vanilla CSS (`src/style.css`)
- **Streaming Pipeline**: File-based JSON Lines (`.jsonl`) I/O với file pointer tracking (`f.seek()`, `f.tell()`)

### ⚡ Quick Commands (Khuyến nghị dùng `python -m` trên Windows)
```bash
# 1. Pipeline tuần tự:
python src/01_prepare_data.py
python src/02_train_model.py
python src/03_inject_anomalies.py
python src/06_evaluate_model.py

# 2. Khởi chạy Dashboard:
python -m streamlit run src/05_dashboard.py

# 3. (Tùy chọn) Chạy Producer độc lập:
python src/04_producer.py --speed 0.5
```

---

## 📊 2. Data Flow & Artifact Contract Matrix

Mọi script trong pipeline đều có hợp đồng Input/Output (Data Contract) rõ ràng. **Tuyệt đối không tự ý đổi tên cột hoặc schema dữ liệu:**

| Step / Script | Input Artifacts | Output Artifacts | Schema / Key Columns & Transformations |
|---|---|---|---|
| **01_prepare_data.py** | `data/raw/household_power_consumption.txt` | `data/processed/train.csv`<br>`data/processed/test.csv`<br>`data/demo/demo.csv`<br>`data/processed/contamination.txt` | • Input raw: `;` separator, `?` as NaN, 1-min interval.<br>• Clean: ffill/bfill, deduplicate timestamps.<br>• Resample: 1h mean.<br>• Split: Train (70%), Test (20%), Demo (10%) **Chronological**.<br>• Contamination: Tính bằng IQR ($Q3 + 1.5 \times IQR$), lưu vào file text (giới hạn [0.07, 0.12]). |
| **02_train_model.py** | `data/processed/train.csv`<br>`data/processed/contamination.txt` | `models/isolation_forest_model.pkl`<br>`models/scaler.pkl`<br>`models/feature_names.pkl` | • Features: `create_features(df)` từ `features.py`.<br>• Scaler: `RobustScaler().fit_transform(X)`.<br>• Model: `IsolationForest(n_estimators=200, max_samples=512, max_features=0.5, random_state=42, n_jobs=-1)`. |
| **03_inject_anomalies.py** | `data/demo/demo.csv` | `data/demo/demo_with_anomalies.csv` | • Bơm nhãn giả lập vào tập Demo (Seed=42):<br>  - `power_surge` (3%): Công suất & cường độ $\times [3.0, 5.0]$.<br>  - `voltage_drop` (3%): Điện áp giảm $-[20, 40]\text{V}$.<br>  - `night_spike` (2%): Giờ 1h–5h sáng, công suất $\times [2.0, 3.0]$.<br>• Output columns: sensor cols + `is_anomaly` (0/1), `anomaly_type`. |
| **04_producer.py** | `data/demo/demo_with_anomalies.csv` | `data/demo/stream_buffer.jsonl` | • Stream simulator: Đọc từng dòng CSV, phát JSON line kèm timestamp và `sent_at`.<br>• CLI Args: `--speed` (giây/msg), `--limit`, `--skip`, `--append`. |
| **05_dashboard.py** | `models/*.pkl`<br>`data/demo/demo_with_anomalies.csv`<br>`data/demo/stream_buffer.jsonl` | UI Streamlit Realtime | • Tab 1: Khám phá lịch sử (3,417 mẫu demo, range filter, 1-click presets).<br>• Tab 2: Giám sát luồng Realtime (Buffer 30 dòng, max 100 điểm, lag 24h lookup).<br>• Quản lý background producer qua `subprocess.Popen`. |
| **06_evaluate_model.py** | `models/*.pkl`<br>`data/demo/demo_with_anomalies.csv` | `reports/evaluation_report.html`<br>`reports/evaluation_metrics.json` | • Metrics: Precision, Recall, F1, ROC-AUC, Detection Rate theo từng loại anomaly.<br>• Plots: Confusion Matrix, PR Curve, Anomaly Timeline, Top Anomaly Table. |

---

## 🧬 3. Feature Engineering Specifications (`src/features.py`)

`src/features.py` là **Single Source of Truth** duy nhất cho toàn bộ hệ thống (cả offline batch và online stream).

```
Raw Sensors (7 cột) ──> Drop (Global_intensity, Sub_metering_1..3)
                    ──> Cyclical Time: hour_sin, hour_cos (24h cyclic)
                    ──> Lag & Diff: power_lag_1h, power_diff_1h, power_lag_24h, voltage_lag_1h, voltage_diff_1h
                    ──> Relative Deviation: power_deviation_24h = (P - P_lag24) / (|P_lag24| + eps)
                    ──> Local Z-Score (6h): power_zscore_6h, voltage_zscore_6h = (val - mean_6h) / (std_6h + eps)
                    ──> Ratio: reactive_ratio = clip(Reactive / (|Active| + eps), -10, 10) (rồi drop Global_reactive_power)
                    ──> Night Context: power_hourly_diff (lệch TB giờ), is_night (1 nếu 1h <= hour <= 5h)
```

> [!IMPORTANT]
> **Quy tắc Real-time Buffer:**
> Hàm `create_features_realtime(buffer_df)` yêu cầu tối thiểu **25 mẫu** dữ liệu trong buffer để có thể tính được `power_lag_24h` và rolling 6h. Nếu buffer < 25 dòng, hàm trả về `None`.

---

## 🏷️ 4. Anomaly Classification & Explainable AI (`src/classify.py`)

Sau khi Isolation Forest gán nhãn `predict() == -1` (Bất thường), hệ thống thực hiện phân loại post-hoc theo luật chuyên ngành:

1. **`voltage_drop`**: Khi `voltage_diff_1h < -15.0V`.
2. **`night_spike`**: Khi `is_night == 1` VÀ (`power_zscore_6h > 1.0` HOẶC `power_deviation_24h > 1.5`).
3. **`power_surge`**: Khi `is_night == 0` VÀ (`power_zscore_6h > 1.5` HOẶC `power_deviation_24h > 1.5`).
4. **`unknown`**: Điểm dị biệt được model bắt nhưng không thỏa các ngưỡng vi phân trên.

**Feature Attribution (Explainability):**
- Tính độ lệch của từng feature so với phân phối bình thường: $\text{score} = \frac{|x - \text{median}|}{\text{IQR}}$.
- Lấy Top 3 features có score cao nhất (> 0.5) để hiển thị nguyên nhân trực tiếp lên UI Dashboard.

---

## 🎨 5. UI/UX & Design Guidelines (`src/style.css` & `05_dashboard.py`)

- **Bố cục**: **F-Z Pattern Layout** — Trải rộng toàn màn hình (`layout="wide"`), **KHÔNG sử dụng Sidebar**.
- **Top Control Bar**: Đặt các bộ lọc (Mode, Date Range, Preset Buttons, Action Buttons) trên cùng một hàng ngang.
- **Bảng màu chủ đạo (Design Tokens)**:
  - Primary Purple: `#6B4CE6` (Nền sáng: `#F3F0FF`, Dark: `#5338B5`)
  - Accent Orange: `#EF7D32` (Nền sáng: `#FFF4EB`)
  - Danger/Anomaly: `#EF4444` (Nền sáng: `#FEF2F2`)
  - Background: `#FAFAFA`, Card Surface: `#FFFFFF`, Border: `#E5E7EB`
- **Biểu đồ Plotly**:
  - Luôn sử dụng layout chuẩn `template="plotly_white"`, font family `Inter, -apple-system, sans-serif`.
  - Giữ hover mode `hovermode="x unified"`.

---

## ⚡ 6. Real-time Streaming & Subprocess Management

1. **Cơ chế truyền nhận:** Producer ghi nối đuôi file JSONL (`stream_buffer.jsonl`), Dashboard theo dõi vị trí con trỏ file (`last_position`) bằng `f.seek()` và `f.tell()` để chỉ nạp các dòng mới, không reload toàn bộ file.
2. **Background Subprocess:**
   - Khi kích hoạt từ nút bấm trên Dashboard, producer được spawn qua `subprocess.Popen([sys.executable, "src/04_producer.py", "--speed", ...])`.
   - PID được lưu trong `st.session_state["producer_pid"]`.
   - Phải xử lý tín hiệu terminate sạch (`SIGTERM` hoặc `psutil.terminate()`) khi người dùng bấm Dừng.

---

## 🛡️ 7. Guardrails & Anti-Patterns (BẮT BUỘC TUÂN THỦ)

1. ❌ **KHÔNG BAO GIỜ Shuffle dữ liệu Time-series:** Phân chia tập dữ liệu Train / Test / Demo bắt buộc theo thứ tự thời gian tuyến tính (`iloc[:split]`). Tuyệt đối không dùng `train_test_split(shuffle=True)`.
2. ❌ **KHÔNG Duplicate logic Feature Engineering:** Mọi phép tính feature phải import từ `src/features.py`. Không code chay công thức trong Dashboard hoặc Evaluate script để tránh Train-Serve Skew.
3. ❌ **KHÔNG Dùng `Global_intensity` làm feature:** Cột này có tương quan $r = +0.9992$ với `Global_active_power` và gây nhiễu cây Isolation Forest.
4. ❌ **KHÔNG Thay đổi seed:** Luôn dùng `RANDOM_STATE = 42` trên mọi hàm ngẫu nhiên (`np.random.default_rng(42)`, `IsolationForest(random_state=42)`).
5. ❌ **KHÔNG Commit Artifacts nặng:** `data/raw/`, `data/processed/`, `data/demo/`, `models/*.pkl`, `reports/` phải nằm trong `.gitignore`.
6. ⚠️ **Windows Encoding Safety:** Mọi script CLI có in log tiếng Việt phải bao bọc `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.

---

## 📋 8. Fast AI Modification Playbooks (Checklists)

### A. Thêm một đặc trưng mới (Feature Addition):
- [ ] 1. Thêm công thức tính trong hàm `create_features()` ở `src/features.py`.
- [ ] 2. Thêm logic tương ứng trong hàm `create_features_realtime()` ở `src/features.py`.
- [ ] 3. Thêm nhãn tiếng Việt hiển thị trong `FEATURE_LABELS` ở `src/classify.py`.
- [ ] 4. Chạy lại `python src/02_train_model.py` để cập nhật `feature_names.pkl` và `scaler.pkl`.
- [ ] 5. Chạy `python src/06_evaluate_model.py` kiểm tra độ chính xác trước khi mở Dashboard.

### B. Thêm một loại Anomaly mới:
- [ ] 1. Định nghĩa tỷ lệ & phương thức bơm trong `src/03_inject_anomalies.py`.
- [ ] 2. Thêm rule phân loại & nhãn hiển thị trong `src/classify.py` (`TYPE_LABELS` & `classify_anomaly_type`).
- [ ] 3. Chạy `03_inject_anomalies.py` → `06_evaluate_model.py` → kiểm tra badge hiển thị trên `05_dashboard.py`.
