"""TC01-TC06: feature, classification and severity tests using UCI rows."""

import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ENGINEERED_FEATURE_NAMES, MODEL_BUNDLE_PATH, TARGET_COL
from features import (
    calc_severity,
    classify_type,
    extract_features,
    extract_latest,
    get_severity_level,
)

TRAIN_PATH = PROJECT / "data" / "train_hourly.csv"
DEMO_PATH = PROJECT / "data" / "demo_stream.csv"


# Xác nhận tệp dữ liệu kiểm thử tồn tại.
def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu dữ liệu bắt buộc: {path}")


# Đọc dữ liệu theo giờ dùng trong kiểm thử đặc trưng.
def read_hourly(path: Path, rows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, index_col="datetime", parse_dates=True, nrows=rows)


class UciFeatureTests(unittest.TestCase):
    # Nạp dữ liệu dùng chung một lần cho nhóm kiểm thử.
    @classmethod
    def setUpClass(cls):
        require_file(TRAIN_PATH)
        cls.train = read_hourly(TRAIN_PATH, 64)

    # Kiểm tra cửa sổ warm-up cần đủ 25 mẫu.
    def test_tc01_warmup_requires_25_samples(self):
        self.assertIsNone(extract_latest(self.train.iloc[:24].copy()))

    # Kiểm tra mẫu thứ 25 trả về đủ chín đặc trưng hợp lệ.
    def test_tc02_sample_25_returns_nine_valid_features(self):
        latest = extract_latest(self.train.iloc[:25].copy())
        self.assertIsNotNone(latest)
        self.assertEqual(list(latest.index), ENGINEERED_FEATURE_NAMES)
        self.assertEqual(len(latest), 9)
        self.assertFalse(latest.isna().any())

    # Đối chiếu công thức đặc trưng với dữ liệu UCI.
    def test_tc03_feature_formulas_match_uci_values(self):
        source = self.train.iloc[:25].copy()
        latest = extract_latest(source)
        current, previous, lagged = source.iloc[-1], source.iloc[-2], source.iloc[0]
        hour = source.index[-1].hour

        expected = {
            "power_diff_1h": current[TARGET_COL] - previous[TARGET_COL],
            "voltage_diff_1h": current["Voltage"] - previous["Voltage"],
            "power_dev_24h": (
                (current[TARGET_COL] - lagged[TARGET_COL])
                / (abs(lagged[TARGET_COL]) + 1e-6)
            ),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
        }
        for feature, value in expected.items():
            with self.subTest(feature=feature):
                self.assertAlmostEqual(float(latest[feature]), float(value), places=10)

    # Kiểm tra độ sụt điện áp được ưu tiên khi phân loại.
    def test_tc04_voltage_drop_has_classification_priority(self):
        uci_row = self.train.iloc[0].copy().to_dict()
        uci_row.update(
            {
                "voltage_diff_1h": -20.0,
                "is_night": 1,
                "power_zscore_6h": 2.0,
                "power_dev_24h": 2.0,
            }
        )
        self.assertEqual(classify_type(uci_row), "voltage_drop")

    # Kiểm tra phân loại công suất theo thời điểm ngày và đêm.
    def test_tc05_night_and_day_power_classification(self):
        is_night = (self.train.index.hour >= 1) & (self.train.index.hour <= 5)
        night_source = self.train[is_night].iloc[0].to_dict()
        day_source = self.train[~is_night].iloc[0].to_dict()
        cases = [
            (
                {
                    **night_source,
                    "voltage_diff_1h": 0,
                    "is_night": 1,
                    "power_zscore_6h": 1.0,
                    "power_dev_24h": 0,
                },
                "night_spike",
            ),
            (
                {
                    **day_source,
                    "voltage_diff_1h": 0,
                    "is_night": 0,
                    "power_zscore_6h": 1.0,
                    "power_dev_24h": 1.0,
                },
                "power_surge",
            ),
        ]
        for row, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_type(row), expected)

    # Kiểm tra severity bị chặn biên, đơn điệu và phân cấp đúng.
    def test_tc06_severity_is_bounded_monotonic_and_classified(self):
        require_file(DEMO_PATH)
        require_file(Path(MODEL_BUNDLE_PATH))
        demo = read_hourly(DEMO_PATH, 1000)
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        features = extract_features(demo)
        scaled = bundle["scaler"].transform(features[bundle["features"]].values)
        real_scores = np.sort(bundle["model"].decision_function(scaled))
        real_severity = np.array([calc_severity(float(score)) for score in real_scores])

        self.assertTrue(np.all((real_severity >= 0) & (real_severity <= 1)))
        self.assertTrue(np.all(np.diff(real_severity) <= 0))
        self.assertTrue(np.any(real_scores < 0), "Đoạn UCI không chứa score âm")
        self.assertTrue(np.all(real_severity[real_scores < 0] > 0.5))
        self.assertAlmostEqual(calc_severity(0.0), 0.5, places=12)
        boundaries = [
            (0.4999, "normal"),
            (0.5, "warning"),
            (0.6999, "warning"),
            (0.7, "critical"),
        ]
        for value, expected in boundaries:
            with self.subTest(value=value):
                self.assertEqual(get_severity_level(value), expected)


if __name__ == "__main__":
    unittest.main()
