"""Chụp giao diện báo cáo bằng Edge và cơ sở dữ liệu tạm."""

import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from websockets.sync.client import connect

PROJECT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT / "reports/figures"
EDGE = (
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    / "Microsoft/Edge/Application/msedge.exe"
)


# Chọn một cổng cục bộ đang trống.
def port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# Chờ dịch vụ HTTP sẵn sàng.
def wait_http(url: str, timeout: int = 40) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                return response.read()
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(url)


# Gửi lệnh Chrome DevTools Protocol tới trình duyệt.
class CDP:
    # Mở kết nối WebSocket tới trình duyệt.
    def __init__(self, url):
        self.socket = connect(url, max_size=40 * 1024 * 1024, proxy=None)
        self.counter = 0

    # Gửi một lệnh CDP và chờ phản hồi tương ứng.
    def call(self, method, **params):
        self.counter += 1
        self.socket.send(
            json.dumps({"id": self.counter, "method": method, "params": params})
        )
        while True:
            msg = json.loads(self.socket.recv(timeout=40))
            if msg.get("id") == self.counter:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    # Thực thi JavaScript và trả về giá trị.
    def js(self, expression):
        result = self.call(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=True,
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    # Chờ giao diện và biểu đồ của chế độ hoàn tất.
    def ready(self, mode):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            error = self.js(
                "document.querySelector('[data-testid=stException]')?.innerText"
            )
            if error:
                raise RuntimeError(error)
            text = self.js("document.body.innerText") or ""
            marker = "Chi tiết cảnh báo" if mode == "queue" else "Danh sách"
            charts = self.js("document.querySelectorAll('.js-plotly-plot').length")
            if marker.lower() in text.lower() and charts >= 2:
                time.sleep(1)
                return
            time.sleep(0.25)
        raise TimeoutError(f"UI not ready ({mode}): {text[-1200:]}")

    # Chụp một vùng trang và lưu thành PNG.
    def screenshot(self, path, clip):
        result = self.call(
            "Page.captureScreenshot",
            format="png",
            captureBeyondViewport=True,
            fromSurface=True,
            clip={**clip, "scale": 1},
        )
        path.write_bytes(base64.b64decode(result["data"]))


# Khởi chạy dashboard tạm và chụp các vùng báo cáo.
def capture_dashboard_figures() -> None:
    if not EDGE.is_file():
        raise FileNotFoundError(f"Microsoft Edge not found: {EDGE}")
    from collect_report_evidence import fingerprints

    before = fingerprints()
    FIGURES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="meter-report-") as td:
        temp = Path(td)
        app_port, debug_port = port(), port()
        wrapper = temp / "report_app.py"
        wrapper.write_text(
            f"""
import runpy, sys
from datetime import date
import streamlit as st
sys.path.insert(0, {str(PROJECT / "src")!r})
ui = runpy.run_path({str(PROJECT / "src/04_dashboard.py")!r})
ui['init_alert_store'].__globals__['ALERT_DB_PATH'] = {str(temp / "alerts.sqlite3")!r}
st.set_page_config(layout='wide', initial_sidebar_state='collapsed')
if not st.session_state.get('report_seeded'):
    if st.query_params.get('report_view') == 'queue':
        st.session_state['main_mode_select'] = 'Thời gian thực (Real-Time Monitoring)'
        defaults = dict(rt_is_playing=False, rt_cursor=0, rt_buffer=[], rt_display=[], rt_anomaly_events=[], rt_event_keys=set())
        for key, value in defaults.items(): st.session_state[key] = value
        ui['init_alert_store']()
        ui['_step_stream_engine'](ui['load_historical_data'](), ui['load_bundle'](), n_steps=96)
        ui['update_alert']('ALT-20100206090000', 'acknowledged', 'Đã tiếp nhận để kiểm tra số đo; chưa xác nhận sự cố thiết bị.')
        st.session_state['selected_alert_id'] = 'ALT-20100206090000'
    else:
        st.session_state['hist_start'] = date(2010, 2, 5)
        st.session_state['hist_end'] = date(2010, 2, 9)
    st.session_state['report_seeded'] = True
ui['render_application']()
""",
            encoding="utf-8",
        )
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
        processes = []
        cdp = None
        log_path = PROJECT / "reports/report_capture_log.txt"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            "-m",
                            "streamlit",
                            "run",
                            str(wrapper),
                            "--server.address=127.0.0.1",
                            f"--server.port={app_port}",
                            "--server.headless=true",
                            "--browser.gatherUsageStats=false",
                            "--theme.base=light",
                        ],
                        cwd=PROJECT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                )
                wait_http(f"http://127.0.0.1:{app_port}/_stcore/health")
                processes.append(
                    subprocess.Popen(
                        [
                            str(EDGE),
                            "--headless=new",
                            "--disable-gpu",
                            "--no-first-run",
                            "--no-default-browser-check",
                            "--hide-scrollbars",
                            f"--remote-debugging-port={debug_port}",
                            "--remote-debugging-address=127.0.0.1",
                            f"--user-data-dir={temp / 'edge-profile'}",
                            "--window-size=1600,1100",
                            "about:blank",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                )
                pages = json.loads(
                    wait_http(f"http://127.0.0.1:{debug_port}/json/list")
                )
                cdp = CDP(
                    next(
                        p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page"
                    )
                )
                cdp.call("Page.enable")
                cdp.call(
                    "Emulation.setDeviceMetricsOverride",
                    width=1600,
                    height=1100,
                    deviceScaleFactor=2,
                    mobile=False,
                )
                cdp.call(
                    "Emulation.setEmulatedMedia",
                    features=[{"name": "prefers-color-scheme", "value": "light"}],
                )
                for mode, filename in [
                    ("history", "Hinh_4.2_GiaoDien_LichSu.png"),
                    ("queue", "Hinh_4.3_XuLy_CanhBao.png"),
                ]:
                    cdp.call(
                        "Page.navigate",
                        url=f"http://127.0.0.1:{app_port}/?report_view={mode}",
                    )
                    cdp.ready(mode)
                    if mode == "history":
                        clip = {"x": 0, "y": 0, "width": 1600, "height": 1100}
                    else:
                        clip = cdp.js("""(() => {
                            const heading=[...document.querySelectorAll('h4')].find(x=>x.innerText.includes('Cảnh báo cần xử lý'));
                            heading.scrollIntoView({block:'start'});
                            const close=[...document.querySelectorAll('button')].find(x=>x.innerText.includes('Đóng cảnh báo'));
                            const top=heading.getBoundingClientRect().top;
                            const bottom=close.getBoundingClientRect().bottom;
                            return {x:0,y:Math.max(0,top-12),width:1600,height:bottom-top+36};
                        })()""")
                        time.sleep(0.5)
                    cdp.screenshot(FIGURES / filename, clip)
                    print(f"Captured {filename}", flush=True)
                for svg in sorted(FIGURES.glob("*.svg")):
                    cdp.call("Page.navigate", url=svg.as_uri())
                    time.sleep(0.4)
                    size = cdp.js(
                        "({width:document.querySelector('svg').width.baseVal.value,height:document.querySelector('svg').height.baseVal.value})"
                    )
                    preview = PROJECT / "reports" / (svg.stem + "_preview.png")
                    cdp.screenshot(preview, {"x": 0, "y": 0, **size})
            finally:
                if cdp:
                    try:
                        cdp.call("Browser.close")
                    except Exception:
                        pass
                    cdp.socket.close()
                for process in reversed(processes):
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
        assert before == fingerprints(), "Live artifacts changed during capture"
    print("Temporary database/profile removed; live data and source hashes unchanged.")


if __name__ == "__main__":
    capture_dashboard_figures()
