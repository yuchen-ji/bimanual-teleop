"""Read-only local browser preview, separate from operator control."""

from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bimanual_teleop.sim._dependencies import Image


class Preview:
    def __init__(self, bind, port):
        self.images = [None]*3
        self.state = {"state": "starting", "boundary": None}
        self.pose = None
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path == "/":
                    body = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>双臂导轨仿真</title>
<style>body{background:#111827;color:#e5e7eb;font:16px system-ui;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{width:100%;border-radius:8px}pre{white-space:pre-wrap}</style>
<h2>双臂导轨任务仿真</h2><p>请在本地客户端终端按键；此页面显示三路画面与采集状态。</p>
<p>Enter 接管／暂停 | B 打断点 | S 保存 | X 丢弃 | H 待机 | R 重置 | Q 退出</p>
<div class="grid"><div>头部<img src="/camera/0"></div><div>左腕<img src="/camera/1"></div><div>右腕<img src="/camera/2"></div></div>
<pre id="state"></pre><script>setInterval(async()=>{try{let s=await(await fetch('/status')).json();document.getElementById('state').textContent=JSON.stringify(s,null,2)}catch(e){}},500)</script></html>'''.encode("utf-8")
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)
                elif self.path in ("/status", "/pose"):
                    with outer.lock:
                        body = json.dumps(outer.state if self.path == "/status" else outer.pose, ensure_ascii=False).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers(); self.wfile.write(body)
                elif self.path in ("/camera/0", "/camera/1", "/camera/2"):
                    index = int(self.path[-1])
                    self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
                    try:
                        while True:
                            with outer.lock:
                                payload = outer.images[index]
                            if payload:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+str(len(payload)).encode()+b"\r\n\r\n"+payload+b"\r\n")
                            time.sleep(1/15)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer((bind, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, name="sim-preview", daemon=True)
        self.thread.start()

    def update(self, images=None, **status):
        encoded = None
        if images is not None:
            encoded = []
            for image in images:
                buffer = io.BytesIO(); Image.fromarray(image).save(buffer, "JPEG", quality=75)
                encoded.append(buffer.getvalue())
        with self.lock:
            if encoded is not None:
                self.images = encoded
            self.state.update(status)

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def update_pose(self, state):
        with self.lock:
            self.pose = {"q": state["q"], "rail": state["rail"], "tick": state["tick"],
                         "received_ns": time.monotonic_ns()}
