"""TC08: persistent alert workflow using an isolated SQLite database."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEMO_PATH = PROJECT / "data" / "demo_stream.csv"
SPEC = importlib.util.spec_from_file_location("dashboard_alert_test", SRC / "04_dashboard.py")
DASHBOARD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DASHBOARD)


class UciAlertStoreTests(unittest.TestCase):
    def test_tc08_alert_lifecycle_is_idempotent_and_persistent(self):
        self.assertTrue(DEMO_PATH.is_file(), f"Thiếu dữ liệu bắt buộc: {DEMO_PATH}")
        demo = pd.read_csv(DEMO_PATH, index_col="datetime", parse_dates=True)
        source = demo[demo["is_anomaly"] == 1].iloc[0]
        timestamp = source.name

        with tempfile.TemporaryDirectory() as temp_dir:
            DASHBOARD.ALERT_DB_PATH = str(Path(temp_dir) / "alerts.sqlite3")
            DASHBOARD.init_alert_store()
            event = {
                "alert_id": f"TEST-{timestamp:%Y%m%d%H%M%S}",
                "data_time": timestamp.isoformat(),
                "power": float(source["Global_active_power"]),
                "voltage": float(source["Voltage"]),
                "severity": 0.8,
                "severity_level": "critical",
                "anomaly_type": str(source["anomaly_type"]),
                "explanation": "Kiểm thử từ một điểm bất thường UCI Demo",
            }
            DASHBOARD.save_alert(event)
            event["power"] += 0.001
            DASHBOARD.save_alert(event)

            alerts = DASHBOARD.load_alerts()
            self.assertEqual(len(alerts), 1)
            self.assertAlmostEqual(float(alerts.iloc[0]["power"]), event["power"])
            self.assertEqual(alerts.iloc[0]["status"], "new")

            DASHBOARD.update_alert(event["alert_id"], "acknowledged", "Đang kiểm tra")
            acknowledged = DASHBOARD.load_alerts().iloc[0]
            self.assertEqual(acknowledged["status"], "acknowledged")
            self.assertEqual(acknowledged["note"], "Đang kiểm tra")
            self.assertTrue(acknowledged["acknowledged_at"])

            DASHBOARD.update_alert(event["alert_id"], "closed", "Đã xác minh")
            closed = DASHBOARD.load_alerts().iloc[0]
            self.assertEqual(closed["status"], "closed")
            self.assertEqual(closed["note"], "Đã xác minh")
            self.assertTrue(closed["closed_at"])


if __name__ == "__main__":
    unittest.main()
