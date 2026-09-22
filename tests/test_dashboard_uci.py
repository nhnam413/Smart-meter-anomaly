"""TC09: end-to-end Streamlit rendering against the UCI Demo dataset."""

import gc
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
DASHBOARD = SRC / "04_dashboard.py"
DEMO_PATH = PROJECT / "data" / "demo_stream.csv"


class UciDashboardTests(unittest.TestCase):
    # Kiểm tra hai chế độ hiển thị và trạng thái warm-up.
    def test_tc09_both_views_render_and_warmup_is_not_normal(self):
        from streamlit.testing.v1 import AppTest

        self.assertTrue(DEMO_PATH.is_file(), f"Thiếu dữ liệu bắt buộc: {DEMO_PATH}")
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db = Path(temp_dir) / "dashboard-alerts.sqlite3"
            dashboard_source = DASHBOARD.read_text(encoding="utf-8")
            dashboard_source = dashboard_source.replace(
                'if __name__ == "__main__":',
                f'ALERT_DB_PATH = {str(test_db)!r}\n\nif __name__ == "__main__":',
                1,
            )
            script = f"import sys\nsys.path.insert(0, {str(SRC)!r})\n{dashboard_source}"
            app = AppTest.from_string(script, default_timeout=60).run()
            self.assertEqual(len(app.exception), 0, "Chế độ lịch sử phát sinh lỗi")
            self.assertGreaterEqual(len(app.date_input), 2)

            app.radio[0].set_value("Thời gian thực (Real-Time Monitoring)").run()
            self.assertEqual(len(app.exception), 0, "Chế độ trực tiếp phát sinh lỗi")
            self.assertEqual(app.session_state["rt_cursor"], 0)
            self.assertEqual(app.session_state["rt_display"], [])
            self.assertEqual(app.session_state["rt_anomaly_events"], [])
            run_id = app.session_state["rt_run_id"]

            with closing(sqlite3.connect(test_db)) as conn:
                conn.execute(
                    """
                    INSERT INTO alerts (
                        alert_id, data_time, power, voltage, severity, severity_level,
                        anomaly_type, explanation, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        "LEGACY-ALERT",
                        "2010-01-01T00:00:00",
                        1.0,
                        230.0,
                        0.8,
                        "critical",
                        "voltage_drop",
                        "Cảnh báo từ phiên cũ",
                        "2026-01-01T00:00:00",
                    ),
                )
                conn.commit()

            app.run()
            self.assertEqual(len(app.dataframe), 0)

            for _ in range(24):
                app.button(key="btn_step_next").click().run()

            records = app.session_state["rt_display"]
            self.assertEqual(len(records), 24)
            self.assertEqual(
                sum(row["evaluation_status"] == "warming_up" for row in records), 24
            )
            self.assertTrue(
                all(
                    row["is_anomaly"] is None
                    for row in records
                    if row["evaluation_status"] == "warming_up"
                )
            )

            for _ in range(6):
                app.button(key="btn_step_next").click().run()

            records = app.session_state["rt_display"]
            self.assertEqual(len(records), 30)
            self.assertEqual(
                sum(row["evaluation_status"] == "evaluated" for row in records), 6
            )
            events = app.session_state["rt_anomaly_events"]
            self.assertGreater(len(events), 0)
            self.assertTrue(
                all(event["alert_id"].startswith(f"ALT-{run_id}-") for event in events)
            )
            self.assertEqual(len(app.multiselect), 2)
            self.assertGreaterEqual(len(app.selectbox), 2)
            self.assertEqual(len(app.dataframe), 1)
            queue = app.dataframe[0].value
            self.assertEqual(len(queue), len(events))
            self.assertNotIn("LEGACY-ALERT", queue["Mã cảnh báo"].tolist())
            self.assertEqual(
                set(app.multiselect(key="alert_type_filter").value),
                {"power_surge", "voltage_drop", "night_spike"},
            )
            self.assertEqual(len(app.text_area), 1)
            self.assertEqual(len(app.get("download_button")), 1)
            button_labels = {button.label for button in app.button}
            self.assertIn("Tiếp nhận cảnh báo", button_labels)
            self.assertIn("Đóng cảnh báo", button_labels)
            self.assertTrue(test_db.is_file())

            with closing(sqlite3.connect(test_db)) as conn:
                stored_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            app.button(key="btn_reset_stream").click().run()
            self.assertNotEqual(app.session_state["rt_run_id"], run_id)
            self.assertEqual(app.session_state["rt_cursor"], 0)
            self.assertEqual(app.session_state["rt_display"], [])
            self.assertEqual(app.session_state["rt_anomaly_events"], [])
            self.assertEqual(len(app.dataframe), 0)
            with closing(sqlite3.connect(test_db)) as conn:
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0],
                    stored_count,
                )
            del app
            gc.collect()


if __name__ == "__main__":
    unittest.main()
