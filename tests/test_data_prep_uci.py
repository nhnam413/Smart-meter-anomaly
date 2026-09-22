"""TC07: deterministic and constrained anomaly injection on UCI data."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import TARGET_COL

TRAIN_PATH = PROJECT / "data" / "train_hourly.csv"
SPEC = importlib.util.spec_from_file_location("data_prep", SRC / "01_data_prep.py")
DATA_PREP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DATA_PREP)


class UciDataPreparationTests(unittest.TestCase):
    # Kiểm tra quá trình tiêm lỗi có thể tái tạo và đúng ràng buộc.
    def test_tc07_injection_is_reproducible_and_respects_constraints(self):
        self.assertTrue(TRAIN_PATH.is_file(), f"Thiếu dữ liệu bắt buộc: {TRAIN_PATH}")
        source = pd.read_csv(
            TRAIN_PATH, index_col="datetime", parse_dates=True, nrows=1000
        )
        first = DATA_PREP.inject_synthetic_anomalies(source.copy())
        second = DATA_PREP.inject_synthetic_anomalies(source.copy())
        pd.testing.assert_frame_equal(first, second)

        counts = first["anomaly_type"].value_counts().to_dict()
        self.assertEqual(counts.get("power_surge"), 30)
        self.assertEqual(counts.get("voltage_drop"), 30)
        self.assertEqual(counts.get("night_spike"), 20)
        self.assertEqual(int(first["is_anomaly"].sum()), 80)

        surge = first[first["anomaly_type"] == "power_surge"]
        night = first[first["anomaly_type"] == "night_spike"]
        drop = first[first["anomaly_type"] == "voltage_drop"]
        self.assertTrue(((surge.index.hour >= 8) & (surge.index.hour <= 22)).all())
        self.assertTrue(((night.index.hour >= 1) & (night.index.hour <= 5)).all())

        surge_ratio = surge[TARGET_COL] / source.loc[surge.index, TARGET_COL]
        night_ratio = night[TARGET_COL] / source.loc[night.index, TARGET_COL]
        voltage_delta = source.loc[drop.index, "Voltage"] - drop["Voltage"]
        self.assertTrue(np.all((surge_ratio >= 3.0) & (surge_ratio <= 5.0)))
        self.assertTrue(np.all((night_ratio >= 2.0) & (night_ratio <= 3.5)))
        self.assertTrue(np.all((voltage_delta >= 20.0) & (voltage_delta <= 40.0)))
        self.assertTrue(first.loc[first["anomaly_type"] != "normal"].index.is_unique)


if __name__ == "__main__":
    unittest.main()
