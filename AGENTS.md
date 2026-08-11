# Smart Meter Anomaly Detection

## Tech Stack
- Python 3.10+
- Xử lý dữ liệu: Pandas ≥2.0, NumPy ≥1.24
- Machine Learning: scikit-learn ≥1.3 (Isolation Forest), Joblib ≥1.3
- Dashboard: Streamlit ≥1.30, Plotly ≥5.18
- Streaming: File-based JSONL (không dùng message broker)

## Commands
- Install deps: `pip install -r requirements.txt`
- Prepare data: `python src/01_prepare_data.py`
- Train model: `python src/02_train_model.py`
- Inject anomalies: `python src/03_inject_anomalies.py`
- Evaluate model: `python src/06_evaluate_model.py`
- Run producer: `python src/04_producer.py [--speed 0.5]`
- Run dashboard: `streamlit run src/05_dashboard.py`

## Pipeline Order
```
01_prepare_data → 02_train_model → 03_inject_anomalies → 06_evaluate_model (đánh giá)
                                              ↓
                              04_producer (terminal 1) + 05_dashboard (terminal 2)
```
- Bước 01–03, 06: Chạy tuần tự 1 lần.
- Bước 04 + 05: Chạy song song (2 terminal riêng biệt).

## Code Conventions
- **File naming**: `NN_snake_case.py` — numbered pipeline steps (01, 02, ..., 06)
- **Language**: Vietnamese docstrings/comments, English variable/function names
- **Feature Engineering**: Phải đồng nhất giữa train (`src/features.py`) và inference (`05_dashboard.py`, `06_evaluate_model.py`). Import từ `src/features.py` — KHÔNG duplicate logic.
- **Random seed**: Mọi random operation dùng seed `42` để reproducible
- **Data splits**: Train 70% / Test 20% / Demo 10% (chronological, KHÔNG shuffle)
- **Config**: Hyperparameters & paths tập trung trong `src/config.py`

## Boundaries
- **KHÔNG BAO GIỜ** shuffle dữ liệu chuỗi thời gian → data leakage
- **KHÔNG BAO GIỜ** sửa feature engineering ở 1 file mà quên file khác → train-serve skew
- **KHÔNG commit**: `data/raw/`, `data/processed/`, `data/demo/`, `models/*.pkl`, `reports/` (đã có trong .gitignore)
- **KHÔNG commit**: `.env`, secrets, hoặc API keys
- Hỏi trước khi thay đổi model hyperparameters (contamination, n_estimators)

## Key Patterns
- Feature Engineering pipeline: Cyclical Encoding (sin/cos) → Lag Features (1h, 2h, 24h) → Rolling Stats (mean/std 6h)
- Anomaly types: `power_surge` (3–5x power), `voltage_drop` (−20–40V), `night_spike` (2–3x power at 1h–5h)
- Model output: `predict()` → 1 (normal) / −1 (anomaly); `decision_function()` → anomaly score

## Project Structure
```
src/
├── config.py               # Centralized configuration
├── features.py             # Shared feature engineering (single source of truth)
├── 01_prepare_data.py      # ETL: UCI CSV → hourly → train/test/demo
├── 02_train_model.py       # Train Isolation Forest → models/*.pkl
├── 03_inject_anomalies.py  # Inject synthetic anomalies into demo set
├── 04_producer.py          # Stream simulator (JSONL, 1 msg/sec)
├── 05_dashboard.py         # Streamlit dashboard (history + real-time)
├── 06_evaluate_model.py    # Model evaluation metrics & Plotly HTML reports
└── style.css               # Custom CSS theme
```
