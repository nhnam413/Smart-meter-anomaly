"""TC09: end-to-end Streamlit rendering against the UCI Demo dataset."""

from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
DASHBOARD = SRC / "04_dashboard.py"
DEMO_PATH = PROJECT / "data" / "demo_stream.csv"


class UciDashboardTests(unittest.TestCase):
    def test_tc09_both_views_render_and_warmup_is_not_normal(self):
        from streamlit.testing.v1 import AppTest

        self.assertTrue(DEMO_PATH.is_file(), f"Thiếu dữ liệu bắt buộc: {DEMO_PATH}")
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db = Path(temp_dir) / "dashboard-alerts.sqlite3"
            script = f"""
import runpy
import sys
sys.path.insert(0, {str(SRC)!r})
ui = runpy.run_path({str(DASHBOARD)!r})
ui['init_alert_store'].__globals__['ALERT_DB_PATH'] = {str(test_db)!r}
ui['main']()
"""
            app = AppTest.from_string(script, default_timeout=60).run()
            self.assertEqual(len(app.exception), 0, "Chế độ lịch sử phát sinh lỗi")
            self.assertGreaterEqual(len(app.date_input), 2)

            app.radio[0].set_value("Thời gian thực (Real-Time Monitoring)").run()
            self.assertEqual(len(app.exception), 0, "Chế độ trực tiếp phát sinh lỗi")
            records = app.session_state["rt_display"]
            self.assertEqual(len(records), 30)
            self.assertEqual(
                sum(row["evaluation_status"] == "warming_up" for row in records), 24
            )
            self.assertTrue(all(
                row["is_anomaly"] is None
                for row in records
                if row["evaluation_status"] == "warming_up"
            ))
            self.assertEqual(len(app.multiselect), 2)
            self.assertGreaterEqual(len(app.selectbox), 2)
            self.assertEqual(len(app.dataframe), 1)
            self.assertEqual(len(app.text_area), 1)
            self.assertEqual(len(app.get("download_button")), 1)
            button_labels = {button.label for button in app.button}
            self.assertIn("Tiếp nhận cảnh báo", button_labels)
            self.assertIn("Đóng cảnh báo", button_labels)
            self.assertTrue(test_db.is_file())


if __name__ == "__main__":
    unittest.main()
