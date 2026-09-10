"""TC10: full model-pipeline regression test on the labeled UCI Demo data."""

from pathlib import Path
import sys
import unittest

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ENGINEERED_FEATURE_NAMES
from features import extract_features

DEMO_PATH = PROJECT / "data" / "demo_stream.csv"
BUNDLE_PATH = PROJECT / "models" / "model_bundle.pkl"


class UciModelIntegrationTests(unittest.TestCase):
    def test_tc10_full_demo_pipeline_matches_reported_range(self):
        self.assertTrue(DEMO_PATH.is_file(), f"Thiếu dữ liệu bắt buộc: {DEMO_PATH}")
        self.assertTrue(BUNDLE_PATH.is_file(), f"Thiếu mô hình bắt buộc: {BUNDLE_PATH}")

        demo = pd.read_csv(DEMO_PATH, index_col="datetime", parse_dates=True)
        bundle = joblib.load(BUNDLE_PATH)
        required = {"model", "scaler", "features", "medians", "iqrs"}
        self.assertTrue(required.issubset(bundle), f"Bundle thiếu: {required - set(bundle)}")
        self.assertEqual(list(bundle["features"]), ENGINEERED_FEATURE_NAMES)

        features = extract_features(demo)
        self.assertEqual(len(features), 6810)
        labels = demo.loc[features.index, "is_anomaly"].astype(int).to_numpy()
        self.assertEqual(int(labels.sum()), 545)

        scaled = bundle["scaler"].transform(features[bundle["features"]].values)
        predictions = (bundle["model"].predict(scaled) == -1).astype(int)
        scores = bundle["model"].decision_function(scaled)
        matrix = confusion_matrix(labels, predictions)
        auc = roc_auc_score(labels, -scores)
        recall = recall_score(labels, predictions)

        self.assertEqual(int(matrix.sum()), 6810)
        self.assertGreaterEqual(auc, 0.91)
        self.assertLessEqual(auc, 0.94)
        self.assertGreaterEqual(recall, 0.70)
        self.assertLessEqual(recall, 0.75)


if __name__ == "__main__":
    unittest.main()
