"""Regression checks for theme resolution and the controls in the alert queue."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT / "src" / "04_dashboard.py"


class LightThemeTests(unittest.TestCase):
    def test_native_theme_is_light_from_all_launch_directories(self):
        # Use the same script-level config path as Streamlit's bootstrap.
        # Checking CSS alone would miss the canvas dataframe's native palette.
        probe = f"""
import json
from streamlit import config
config._main_script_path = {str(DASHBOARD)!r}
config.get_config_options(force_reparse=True)
keys = ['theme.base', 'theme.backgroundColor', 'theme.textColor',
        'theme.dataframeHeaderBackgroundColor', 'theme.primaryColor']
print(json.dumps({{k: config.get_option(k) for k in keys}}))
"""
        expected = {
            "theme.base": "light",
            "theme.backgroundColor": "#F8FAFC",
            "theme.textColor": "#0F172A",
            "theme.dataframeHeaderBackgroundColor": "#F1F5F9",
            "theme.primaryColor": "#0066FF",
        }
        for cwd in (PROJECT.parent, PROJECT, PROJECT / "src"):
            with self.subTest(cwd=cwd):
                result = subprocess.run(
                    [sys.executable, "-c", probe], cwd=cwd,
                    capture_output=True, text=True, check=True,
                )
                self.assertEqual(json.loads(result.stdout), expected)

    def test_alert_controls_render_without_touching_live_history(self):
        from streamlit.testing.v1 import AppTest

        # Exercise the actual CSS and queue with synthetic data and no database IO.
        script = f"""
import runpy
import sys
sys.path.insert(0, {str(PROJECT / 'src')!r})
import pandas as pd
import streamlit as st
ui = runpy.run_path({str(DASHBOARD)!r})
st.set_page_config(layout='wide')
ui['load_custom_css']()
queue = ui['render_alert_queue']
queue.__globals__['load_alerts'] = lambda alert_ids: pd.DataFrame([{{
    'alert_id': 'THEME-OLD', 'data_time': '2010-02-08T23:00:00',
    'power': 1.514, 'voltage': 216.8, 'severity': 0.95,
    'severity_level': 'warning', 'anomaly_type': 'voltage_drop',
    'explanation': 'Voltage change', 'status': 'new', 'note': '',
}}, {{
    'alert_id': 'THEME-NEW', 'data_time': '2010-02-10T23:00:00',
    'power': 1.614, 'voltage': 226.8, 'severity': 0.55,
    'severity_level': 'warning', 'anomaly_type': 'power_surge',
    'explanation': 'Power change', 'status': 'new', 'note': '',
}}, {{
    'alert_id': 'THEME-MIDDLE', 'data_time': '2010-02-09T23:00:00',
    'power': 1.414, 'voltage': 236.8, 'severity': 0.75,
    'severity_level': 'critical', 'anomaly_type': 'night_spike',
    'explanation': 'Night change', 'status': 'new', 'note': '',
}}])
queue(['THEME-OLD', 'THEME-NEW', 'THEME-MIDDLE'])
"""
        app = AppTest.from_string(script, default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.multiselect), 2)
        self.assertEqual(len(app.selectbox), 1)
        self.assertEqual(len(app.text_area), 1)
        self.assertEqual(len(app.get("download_button")), 1)
        self.assertEqual(
            app.dataframe[0].value["Mã cảnh báo"].tolist(),
            ["THEME-NEW", "THEME-MIDDLE", "THEME-OLD"],
        )
        self.assertIn("color-scheme: only light", app.markdown[0].value)


if __name__ == "__main__":
    unittest.main()
