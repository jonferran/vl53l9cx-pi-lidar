#!/usr/bin/env python3
"""
LiDAR Web Studio -- a networked dashboard for the VL53L9CX ToF LiDAR.

Standalone companion to the particle viewer: boots the sensor via its own
FIFO, reads the 100 Hz depth stream in a background thread, and serves a rich
real-time web UI (open http://<pi-ip>:8080 from any device on the LAN):

  * a live, orbitable 3D point cloud rendered in raw WebGL in the browser
  * a live depth heatmap (MJPEG stream)
  * real-time gesture recognition (hand tracking, wave / push-pull / swipe)
  * recording to disk + instant replay
  * one-click export of the current cloud as a standard .PLY point cloud
    (openable in MeshLab / CloudCompare / Blender / online viewers)

Pure standard library + numpy + cv2 -- no web framework, nothing to pip
install. Uses the Pi's spare cores (reader thread + threaded HTTP) and RAM
(frame buffers, in-memory replay) that the GPU viewer leaves idle.

Sensor bring-up (v4l2 arming + vl53l9_bringup) is done by run_lidar_web.sh,
exactly like the particle viewer's launcher; this program only reads the FIFO.
"""
import os
import io
import sys
import time
import json
import array
import fcntl
import termios
import threading
import collections
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2

from lidar_gestures import GestureEngine

# ---- sensor stream geometry (mirrors visualize_gpu.py) --------------------
BINNING = int(os.environ.get("VL_BINNING", "2"))
GEOM = {2: (54, 42, 148), 4: (24, 24, 38)}
RAW_W, RAW_H, CSI_H = GEOM[BINNING]
N_ZONES = RAW_W * RAW_H
CH_LEN = N_ZONES * 2
CSI_W = 100
STRIDE = 128
FRAME_BYTES = STRIDE * CSI_H
DIST_MASK = 0x7FFF
PIPE = os.environ.get("VL_PIPE", "/tmp/tof_pipe_web")
MAX_MM = 4000
PORT = int(os.environ.get("VL_PORT", "8080"))
REC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
ORIENT = 3


def orient(a, m):
    if m == 1:
        return a[::-1, :]
    if m == 2:
        return a[:, ::-1]
    if m == 3:
        return a[::-1, ::-1]
    return a


def read_exact(f, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        m = f.readinto(view[got:])
        if not m:
            return None
        got += m
    return buf


# ---- shared state (one producer thread, many HTTP reader threads) ---------
class Studio:
    def __init__(self):
        self.lock = threading.Lock()
        self.depth = np.zeros((RAW_H, RAW_W), dtype=np.uint16)   # latest oriented frame
        self.gesture = {}
        self.stats = {"fps": 0.0, "min_mm": 0, "max_mm": 0, "mean_mm": 0,
                      "active": 0, "clients": 0, "mode": "live"}
        self.gest = GestureEngine(RAW_W, RAW_H, MAX_MM)
        self.running = True
        self.clients = 0

        # recording / playback
        self.rec_file = None
        self.rec_name = ""
        self.rec_count = 0
        self.play_frames = None     # np.ndarray (n, H, W) uint16 when replaying
        self.play_idx = 0
        self.playing = False

        # HD snapshot: a short ring of recent frames to temporally average.
        self.recent = collections.deque(maxlen=32)
        # Air-drawing: accumulate fingertip positions into a 3D sketch.
        self.paint_on = False
        self.paint = []             # list of (x,y,z, r,g,b) floats
        self._paint_last = None

    # -- called by the sensor thread each frame --
    def push_live(self, depth):
        g = self.gest.update(depth)
        with self.lock:
            if not self.playing:
                self.depth = depth
                self.gesture = g
                self.recent.append(depth)
            if self.rec_file is not None:
                self.rec_file.write(depth.tobytes())
                self.rec_count += 1
        if self.paint_on and g.get("presence"):
            self._add_paint(g)

    def _add_paint(self, g):
        """Drop a colored point at the current fingertip, spaced out."""
        # Fingertip world position, matching the browser cloud's mapping.
        x = g["hand_x"] * (RAW_W / 2.0)
        y = g["hand_y"] * (RAW_H / 2.0)
        z = -(g["hand_z_mm"] / 22.0 - 40.0)
        p = (x, y, z)
        if self._paint_last is not None:
            d = sum((a - b) ** 2 for a, b in zip(p, self._paint_last))
            if d < 0.6:          # min spacing so it doesn't clump when still
                return
        self._paint_last = p
        # Color by depth (TURBO-ish), same feel as the cloud.
        t = 1.0 - min(g["hand_z_mm"], MAX_MM) / MAX_MM
        r = max(0.0, 1.0 - 1.8 * abs(t - 0.75))
        gg = max(0.0, 1.0 - 2.2 * abs(t - 0.5))
        b = max(0.0, 1.0 - 2.0 * abs(t - 0.25))
        with self.lock:
            self.paint.append((x, y, z, r, gg, b))
            if len(self.paint) > 20000:
                self.paint = self.paint[-20000:]

    def push_playback(self):
        with self.lock:
            if self.play_frames is None:
                return
            self.depth = self.play_frames[self.play_idx]
            self.play_idx = (self.play_idx + 1) % len(self.play_frames)
            # gestures still run on replayed frames (fun to see them fire again)
        self.gesture = self.gest.update(self.depth)

    def snapshot(self):
        with self.lock:
            return self.depth.copy(), dict(self.gesture), dict(self.stats)

    def hd_depth(self):
        """Temporally-averaged depth over the recent ring -> denoised frame."""
        with self.lock:
            frames = list(self.recent)
        if not frames:
            return self.depth.copy()
        stack = np.stack(frames).astype(np.float32)   # (n,H,W)
        valid = (stack > 0) & (stack < MAX_MM)
        cnt = valid.sum(axis=0)
        summ = np.where(valid, stack, 0.0).sum(axis=0)
        avg = np.zeros_like(summ)
        nz = cnt > 0
        avg[nz] = summ[nz] / cnt[nz]
        return avg.astype(np.uint16)

    def paint_bytes(self):
        with self.lock:
            if not self.paint:
                return b""
            return np.asarray(self.paint, dtype="<f4").tobytes()

    def paint_toggle(self, on=None):
        with self.lock:
            self.paint_on = (not self.paint_on) if on is None else bool(on)
            if self.paint_on:
                self._paint_last = None
            return self.paint_on

    def paint_clear(self):
        with self.lock:
            self.paint = []
            self._paint_last = None

    # -- recording control --
    def rec_start(self):
        os.makedirs(REC_DIR, exist_ok=True)
        name = time.strftime("rec_%Y%m%d_%H%M%S.ldr")
        path = os.path.join(REC_DIR, name)
        f = open(path, "wb")
        # header: magic, W, H (little-endian uint16 x3) so it's self-describing
        f.write(np.array([0x4C44, RAW_W, RAW_H], dtype="<u2").tobytes())
        with self.lock:
            self.rec_file = f
            self.rec_name = name
            self.rec_count = 0
        return name

    def rec_stop(self):
        with self.lock:
            if self.rec_file:
                self.rec_file.close()
            name, n = self.rec_name, self.rec_count
            self.rec_file = None
            self.rec_name = ""
        return name, n

    def list_recordings(self):
        if not os.path.isdir(REC_DIR):
            return []
        out = []
        for fn in sorted(os.listdir(REC_DIR)):
            if fn.endswith(".ldr"):
                p = os.path.join(REC_DIR, fn)
                out.append({"name": fn, "kb": os.path.getsize(p) // 1024})
        return out

    def play_load(self, name):
        p = os.path.join(REC_DIR, os.path.basename(name))
        if not os.path.exists(p):
            return False
        raw = np.fromfile(p, dtype="<u2")
        if raw.size < 3:
            return False
        w, h = int(raw[1]), int(raw[2])
        body = raw[3:]
        nfr = body.size // (w * h)
        if nfr <= 0:
            return False
        frames = body[:nfr * w * h].reshape(nfr, h, w).astype(np.uint16)
        with self.lock:
            self.play_frames = frames
            self.play_idx = 0
            self.playing = True
            self.stats["mode"] = "replay:" + name
        return True

    def play_stop(self):
        with self.lock:
            self.playing = False
            self.play_frames = None
            self.stats["mode"] = "live"


STUDIO = Studio()


def _synth_depth(t):
    """A moving-hand depth scene for VL_SYNTH mode (no sensor needed)."""
    bg = 1800.0
    frame = np.full((RAW_H, RAW_W), bg, dtype=np.float32)
    cx = RAW_W / 2 + 14 * np.sin(t * 1.6)
    cy = RAW_H / 2 + 5 * np.cos(t * 0.9)
    cz = 900 + 250 * np.sin(t * 0.7)
    yy, xx = np.mgrid[0:RAW_H, 0:RAW_W]
    mask = (xx - cx) ** 2 + ((yy - cy) * 1.3) ** 2 <= 7.0 ** 2
    frame[mask] = cz
    frame += np.random.uniform(-8, 8, frame.shape)   # sensor-like noise
    return np.clip(frame, 0, MAX_MM).astype(np.uint16)


# ---- sensor reader thread -------------------------------------------------
def sensor_thread():
    """Read the FIFO at full rate, decode, and update shared state."""
    fps_n, fps_t0 = 0, time.time()

    if os.environ.get("VL_SYNTH"):
        print("[web] SYNTH mode -- generating a fake moving hand (no sensor)", flush=True)
        t0 = time.time()
        while STUDIO.running:
            if STUDIO.playing:
                STUDIO.push_playback(); time.sleep(0.03); continue
            depth = _synth_depth(time.time() - t0)
            STUDIO.push_live(depth)
            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                valid = (depth > 0) & (depth < MAX_MM)
                with STUDIO.lock:
                    STUDIO.stats.update(fps=round(fps_n / (now - fps_t0), 1),
                        min_mm=int(depth[valid].min()) if valid.any() else 0,
                        max_mm=int(depth[valid].max()) if valid.any() else 0,
                        mean_mm=int(depth[valid].mean()) if valid.any() else 0,
                        active=int(valid.sum()), clients=STUDIO.clients,
                        rec=STUDIO.rec_name, rec_n=STUDIO.rec_count)
                fps_n, fps_t0 = 0, now
            time.sleep(0.01)
        return

    # Wait for the FIFO to exist (launcher creates it).
    for _ in range(200):
        if os.path.exists(PIPE):
            break
        time.sleep(0.05)

    try:
        fifo = open(PIPE, "rb", buffering=0)
    except Exception as e:
        print(f"[web] cannot open FIFO {PIPE}: {e}", flush=True)
        return
    nready = array.array("i", [0])
    fd = fifo.fileno()
    print(f"[web] reading {PIPE}  {RAW_W}x{RAW_H} zones", flush=True)

    while STUDIO.running:
        if STUDIO.playing:
            STUDIO.push_playback()
            time.sleep(0.03)          # ~30 fps replay
            continue

        raw = read_exact(fifo, FRAME_BYTES)
        if raw is None:
            time.sleep(0.01)
            continue
        # Drain to the newest frame so we never lag behind the 100 Hz stream.
        fcntl.ioctl(fd, termios.FIONREAD, nready, True)
        while nready[0] >= FRAME_BYTES:
            nxt = read_exact(fifo, FRAME_BYTES)
            if nxt is None:
                break
            raw = nxt
            nready[0] -= FRAME_BYTES

        full = np.frombuffer(raw, np.uint8).reshape((CSI_H, STRIDE))
        linear = full[:, :CSI_W].reshape(-1)
        if linear.size < CH_LEN:
            continue
        depth = orient((linear[0:CH_LEN].view("<u2") & DIST_MASK)
                       .reshape(RAW_H, RAW_W).astype(np.uint16), ORIENT)
        STUDIO.push_live(depth)

        # Stats once a second.
        fps_n += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            valid = (depth > 0) & (depth < MAX_MM)
            with STUDIO.lock:
                STUDIO.stats.update(
                    fps=round(fps_n / (now - fps_t0), 1),
                    min_mm=int(depth[valid].min()) if valid.any() else 0,
                    max_mm=int(depth[valid].max()) if valid.any() else 0,
                    mean_mm=int(depth[valid].mean()) if valid.any() else 0,
                    active=int(valid.sum()),
                    clients=STUDIO.clients,
                    rec=STUDIO.rec_name, rec_n=STUDIO.rec_count)
            fps_n, fps_t0 = 0, now


# ---- rendering helpers ----------------------------------------------------
def heatmap_jpeg(depth, gesture, scale=9):
    """Colormapped, upscaled depth heatmap as JPEG bytes, with a gesture HUD."""
    norm = np.clip(255.0 * (1.0 - np.clip(depth, 0, MAX_MM) / MAX_MM), 0, 255).astype(np.uint8)
    norm[depth == 0] = 0
    bgr = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    bgr[depth == 0] = (16, 16, 16)
    big = cv2.resize(bgr, (RAW_W * scale, RAW_H * scale), interpolation=cv2.INTER_NEAREST)
    # Draw the tracked hand centroid.
    if gesture.get("presence"):
        hx = int((gesture["hand_x"] * 0.5 + 0.5) * (RAW_W * scale))
        hy = int((-gesture["hand_y"] * 0.5 + 0.5) * (RAW_H * scale))
        cv2.circle(big, (hx, hy), 14, (255, 255, 255), 2)
        cv2.circle(big, (hx, hy), 3, (255, 255, 255), -1)
    ok, buf = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def make_ply(depth):
    """ASCII PLY point cloud of the current frame, colored by depth (TURBO)."""
    ys, xs = np.nonzero((depth > 0) & (depth < MAX_MM))
    if xs.size == 0:
        ys, xs = np.array([0]), np.array([0])
    z = depth[ys, xs].astype(np.float32)
    # World coords: x/y in zone units centred, z in cm toward the viewer.
    X = (xs - RAW_W / 2.0)
    Y = -(ys - RAW_H / 2.0)
    Z = -(z / 20.0)
    norm = np.clip(255.0 * (1.0 - z / MAX_MM), 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]
    n = xs.size
    hdr = ("ply\nformat ascii 1.0\n"
           f"comment VL53L9CX LiDAR snapshot {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    lines = [hdr]
    for i in range(n):
        lines.append(f"{X[i]:.2f} {Y[i]:.2f} {Z[i]:.2f} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")
    return "".join(lines).encode()


def paint_ply(points):
    """ASCII PLY of the air-drawn sketch (list of x,y,z,r,g,b floats 0..1)."""
    n = len(points) if points else 1
    hdr = ("ply\nformat ascii 1.0\n"
           f"comment VL53L9CX air-drawing {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    lines = [hdr]
    if not points:
        lines.append("0 0 0 0 0 0\n")
    for (x, y, z, r, g, b) in points:
        lines.append(f"{x:.2f} {y:.2f} {z:.2f} {int(r*255)} {int(g*255)} {int(b*255)}\n")
    return "".join(lines).encode()


# ---- the dashboard page ---------------------------------------------------
def page_html():
    html = _PAGE
    html = html.replace("__RAW_W__", str(RAW_W)).replace("__RAW_H__", str(RAW_H))
    html = html.replace("__MAX_MM__", str(MAX_MM)).replace("__PORT__", str(PORT))
    return html.encode()


# ---- HTTP handler ---------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass   # quiet

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        q = parse_qs(u.query)
        try:
            if path == "/" or path == "/index.html":
                self._send(200, "text/html; charset=utf-8", page_html())
            elif path == "/stats.json":
                depth, gesture, stats = STUDIO.snapshot()
                with STUDIO.lock:
                    stats["paint_on"] = STUDIO.paint_on
                    stats["paint_n"] = len(STUDIO.paint)
                self._send(200, "application/json",
                           json.dumps({"stats": stats, "gesture": gesture,
                                       "recordings": STUDIO.list_recordings()}).encode())
            elif path == "/depth.bin":
                depth, _, _ = STUDIO.snapshot()
                self._send(200, "application/octet-stream", depth.astype("<u2").tobytes())
            elif path == "/snapshot.ply":
                depth, _, _ = STUDIO.snapshot()
                fn = time.strftime("lidar_%Y%m%d_%H%M%S.ply")
                self._send(200, "application/octet-stream", make_ply(depth),
                           {"Content-Disposition": f'attachment; filename="{fn}"'})
            elif path == "/snapshot_hd.ply":
                fn = time.strftime("lidar_hd_%Y%m%d_%H%M%S.ply")
                self._send(200, "application/octet-stream", make_ply(STUDIO.hd_depth()),
                           {"Content-Disposition": f'attachment; filename="{fn}"'})
            elif path == "/paint.bin":
                self._send(200, "application/octet-stream", STUDIO.paint_bytes())
            elif path == "/paint.ply":
                fn = time.strftime("airdraw_%Y%m%d_%H%M%S.ply")
                with STUDIO.lock:
                    pts = list(STUDIO.paint)
                self._send(200, "application/octet-stream", paint_ply(pts),
                           {"Content-Disposition": f'attachment; filename="{fn}"'})
            elif path == "/stream.mjpg":
                self._mjpeg()
            elif path == "/action":
                self._action(q)
            else:
                self._send(404, "text/plain", b"not found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _action(self, q):
        cmd = q.get("cmd", [""])[0]
        msg = {"ok": True}
        if cmd == "rec_start":
            msg["name"] = STUDIO.rec_start()
        elif cmd == "rec_stop":
            name, n = STUDIO.rec_stop()
            msg["name"], msg["frames"] = name, n
        elif cmd == "play":
            msg["ok"] = STUDIO.play_load(q.get("name", [""])[0])
        elif cmd == "play_stop":
            STUDIO.play_stop()
        elif cmd == "bg_reset":
            depth, _, _ = STUDIO.snapshot()
            STUDIO.gest.reset_background(depth)
        elif cmd == "paint_toggle":
            msg["on"] = STUDIO.paint_toggle()
        elif cmd == "paint_clear":
            STUDIO.paint_clear()
        else:
            msg = {"ok": False, "error": "unknown cmd"}
        self._send(200, "application/json", json.dumps(msg).encode())

    def _mjpeg(self):
        STUDIO.clients += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while STUDIO.running:
                depth, gesture, _ = STUDIO.snapshot()
                jpg = heatmap_jpeg(depth, gesture)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.04)      # ~25 fps
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            STUDIO.clients = max(0, STUDIO.clients - 1)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---- the front-end (HTML + raw WebGL, no external deps) --------------------
_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LiDAR Web Studio</title>
<style>
  :root{--bg:#0a0e14;--pan:#121822;--ln:#1e2836;--tx:#c8d4e0;--dim:#6b7a8d;--ac:#39d0ff;--hot:#ff6b3d;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.4 system-ui,Segoe UI,Roboto,sans-serif}
  header{display:flex;align-items:center;gap:12px;padding:12px 18px;border-bottom:1px solid var(--ln);background:var(--pan)}
  header h1{font-size:16px;margin:0;letter-spacing:.5px}
  header .dot{width:9px;height:9px;border-radius:50%;background:var(--hot);box-shadow:0 0 8px var(--hot)}
  header .dot.live{background:#42e07a;box-shadow:0 0 8px #42e07a}
  .wrap{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px;max-width:1400px;margin:0 auto}
  @media(max-width:900px){.wrap{grid-template-columns:1fr}}
  .card{background:var(--pan);border:1px solid var(--ln);border-radius:10px;overflow:hidden}
  .card h2{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin:0;padding:10px 14px;border-bottom:1px solid var(--ln)}
  .card .body{padding:12px 14px}
  #cloud{width:100%;height:420px;display:block;background:#05080d;cursor:grab;touch-action:none}
  #cloud:active{cursor:grabbing}
  #heat{width:100%;display:block;background:#05080d;border-radius:6px}
  .stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
  .stat{background:#0d131c;border:1px solid var(--ln);border-radius:7px;padding:9px 11px}
  .stat .k{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:var(--dim)}
  .stat .v{font-size:19px;font-weight:600;margin-top:2px}
  .stat .v small{font-size:11px;color:var(--dim);font-weight:400}
  .ges{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
  .chip{padding:6px 12px;border-radius:20px;border:1px solid var(--ln);background:#0d131c;color:var(--dim);font-size:12px;transition:.15s}
  .chip.on{background:var(--ac);color:#04121a;border-color:var(--ac);box-shadow:0 0 12px rgba(57,208,255,.5);font-weight:600}
  .chip.hot.on{background:var(--hot);border-color:var(--hot);box-shadow:0 0 12px rgba(255,107,61,.5);color:#1a0a04}
  .row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}
  button{background:#182231;color:var(--tx);border:1px solid var(--ln);border-radius:7px;padding:8px 13px;font-size:13px;cursor:pointer;transition:.12s}
  button:hover{border-color:var(--ac);color:#fff}
  button.rec{border-color:var(--hot);color:var(--hot)}
  button.rec.active{background:var(--hot);color:#1a0a04}
  a.btn{text-decoration:none;display:inline-block}
  select{background:#182231;color:var(--tx);border:1px solid var(--ln);border-radius:7px;padding:8px}
  .hint{color:var(--dim);font-size:11px;margin-top:8px}
  .bar{height:6px;border-radius:3px;background:#0d131c;overflow:hidden;margin-top:6px}
  .bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--ac),var(--hot));width:0%}
</style></head>
<body>
<header>
  <span class="dot live" id="dot"></span>
  <h1>LiDAR Web Studio</h1>
  <span style="color:var(--dim);font-size:12px" id="mode">live</span>
  <span style="flex:1"></span>
  <span style="color:var(--dim);font-size:12px" id="fps">-- fps</span>
</header>
<div class="wrap">
  <div class="card">
    <h2>3D point cloud &mdash; drag to orbit, scroll to zoom</h2>
    <canvas id="cloud"></canvas>
    <div class="body">
      <div class="row">
        <button id="viewbtn" onclick="toggleView()">◎ View: Points</button>
        <button id="spin">◔ Auto-spin: on</button>
        <button onclick="act('bg_reset')">⟳ Reset background</button>
      </div>
      <div class="row">
        <a class="btn" href="/snapshot.ply" download><button>⬇ Snapshot .PLY</button></a>
        <a class="btn" href="/snapshot_hd.ply" download><button>⬇ HD Snapshot .PLY</button></a>
      </div>
      <div class="hint">Drag to orbit · scroll to zoom · Mesh view triangulates the depth grid into a lit surface. HD snapshot averages recent frames to denoise. .PLY opens in MeshLab / CloudCompare / Blender.</div>
    </div>
  </div>

  <div class="card">
    <h2>Depth heatmap &amp; live stats</h2>
    <div class="body">
      <img id="heat" src="/stream.mjpg" alt="depth heatmap">
      <div class="stats" style="margin-top:12px">
        <div class="stat"><div class="k">Closest</div><div class="v" id="s_min">--<small> mm</small></div></div>
        <div class="stat"><div class="k">Mean depth</div><div class="v" id="s_mean">--<small> mm</small></div></div>
        <div class="stat"><div class="k">Active zones</div><div class="v" id="s_act">--</div></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Gesture recognition</h2>
    <div class="body">
      <div class="ges">
        <span class="chip" id="g_pres">no hand</span>
        <span class="chip" id="g_wave">wave</span>
        <span class="chip hot" id="g_push">push</span>
        <span class="chip" id="g_pull">pull</span>
        <span class="chip hot" id="g_swl">◀ swipe L</span>
        <span class="chip hot" id="g_swr">swipe R ▶</span>
      </div>
      <div style="margin-top:14px">
        <div class="k" style="font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.8px">Hand position</div>
        <canvas id="pos" width="300" height="150" style="width:100%;max-width:300px;background:#0d131c;border:1px solid var(--ln);border-radius:7px;margin-top:6px"></canvas>
      </div>
      <div style="margin-top:12px">
        <div class="k" style="font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.8px">Wave energy</div>
        <div class="bar"><i id="wavebar"></i></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Record &amp; replay</h2>
    <div class="body">
      <div class="row">
        <button id="recbtn" class="rec" onclick="toggleRec()">● Record</button>
        <span id="recinfo" style="color:var(--dim)"></span>
      </div>
      <div class="row">
        <select id="reclist"></select>
        <button onclick="playSel()">▶ Replay</button>
        <button onclick="act('play_stop')">■ Back to live</button>
      </div>
      <div class="hint">Recordings are stored on the Pi as .ldr files and replay right here in the browser.</div>
    </div>
  </div>

  <div class="card">
    <h2>Play &amp; create</h2>
    <div class="body">
      <div class="row">
        <button id="thbtn" onclick="toggleTheremin()">🔊 Theremin: off</button>
        <span class="hint" style="margin:0">Wave your hand: height = pitch, distance = volume, sideways = tone. Plays in <em>your</em> browser.</span>
      </div>
      <div class="row" style="margin-top:14px">
        <button id="paintbtn" class="rec" onclick="togglePaint()">✏ Air-draw: off</button>
        <button onclick="act('paint_clear')">🗑 Clear</button>
        <a class="btn" href="/paint.ply" download><button>⬇ Export drawing .PLY</button></a>
        <span id="paintinfo" class="hint" style="margin:0"></span>
      </div>
      <div class="hint">Air-draw traces your fingertip through 3D space into the point cloud — paint a sculpture in the air, then export it.</div>
    </div>
  </div>
</div>

<script>
const RAW_W=__RAW_W__, RAW_H=__RAW_H__, MAX_MM=__MAX_MM__;
let recording=false, autospin=true;

// ---------- gesture / stats polling ----------
const posCtx=document.getElementById('pos').getContext('2d');
let posTrail=[];
async function poll(){
  try{
    const r=await fetch('/stats.json'); const d=await r.json();
    const st=d.stats, g=d.gesture||{};
    document.getElementById('fps').textContent=(st.fps||0)+' fps';
    document.getElementById('mode').textContent=st.mode||'live';
    document.getElementById('dot').className='dot'+((st.mode||'').startsWith('replay')?'':' live');
    document.getElementById('s_min').innerHTML=(st.min_mm||0)+'<small> mm</small>';
    document.getElementById('s_mean').innerHTML=(st.mean_mm||0)+'<small> mm</small>';
    document.getElementById('s_act').textContent=(st.active||0);
    setChip('g_pres', g.presence, g.presence?'hand tracked':'no hand');
    setChip('g_wave', g.wave); setChip('g_push', g.pushpull==='push');
    setChip('g_pull', g.pushpull==='pull');
    setChip('g_swl', g.swipe==='left'); setChip('g_swr', g.swipe==='right');
    document.getElementById('wavebar').style.width=Math.min(100,(g.waving_amp||0)*100)+'%';
    // hand-position mini map + trail
    if(g.presence){ posTrail.push([g.hand_x,g.hand_y]); if(posTrail.length>40)posTrail.shift(); }
    drawPos(g);
    // recordings list
    const sel=document.getElementById('reclist'); const cur=sel.value;
    sel.innerHTML=(d.recordings||[]).map(x=>`<option value="${x.name}">${x.name} (${x.kb} KB)</option>`).join('');
    if(cur) sel.value=cur;
    if(st.rec) document.getElementById('recinfo').textContent='● '+st.rec+'  '+(st.rec_n||0)+' frames';
    else if(!recording) document.getElementById('recinfo').textContent='';
    // theremin + air-draw
    updateTheremin(g);
    if(st.paint_n!==undefined) document.getElementById('paintinfo').textContent=(st.paint_n>1?st.paint_n+' points':'');
  }catch(e){}
  setTimeout(poll, 100);
}
function setChip(id,on,txt){const e=document.getElementById(id);e.classList.toggle('on',!!on);if(txt)e.textContent=txt;}
function drawPos(g){
  const c=posCtx, W=300,H=150; c.clearRect(0,0,W,H);
  c.strokeStyle='#1e2836'; c.beginPath(); c.moveTo(W/2,0);c.lineTo(W/2,H);c.moveTo(0,H/2);c.lineTo(W,H/2);c.stroke();
  c.strokeStyle='rgba(57,208,255,.4)'; c.beginPath();
  posTrail.forEach((p,i)=>{const x=(p[0]*0.5+0.5)*W,y=(-p[1]*0.5+0.5)*H; i?c.lineTo(x,y):c.moveTo(x,y);}); c.stroke();
  if(g.presence){const x=(g.hand_x*0.5+0.5)*W,y=(-g.hand_y*0.5+0.5)*H;
    c.fillStyle='#ff6b3d'; c.beginPath(); c.arc(x,y,6,0,7); c.fill();}
}

// ---------- record / replay ----------
async function act(cmd,extra){const r=await fetch('/action?cmd='+cmd+(extra||''));return r.json();}
async function toggleRec(){
  const b=document.getElementById('recbtn');
  if(!recording){await act('rec_start');recording=true;b.classList.add('active');b.textContent='■ Stop';}
  else{const j=await act('rec_stop');recording=false;b.classList.remove('active');b.textContent='● Record';
       document.getElementById('recinfo').textContent='saved '+j.name+' ('+j.frames+' frames)';}
}
function playSel(){const s=document.getElementById('reclist').value; if(s) act('play','&name='+encodeURIComponent(s));}
document.getElementById('spin').onclick=function(){autospin=!autospin;this.textContent='◔ Auto-spin: '+(autospin?'on':'off');};

// ---------- WebGL 3D view (points / mesh) + air-draw overlay ----------
const canvas=document.getElementById('cloud');
const gl=canvas.getContext('webgl',{antialias:true,alpha:false});
gl.enable(gl.DEPTH_TEST);
let az=0.6, el=-0.35, zoom=1.0, dragging=false, lx=0, ly=0, viewMode=0;
function resize(){const r=canvas.getBoundingClientRect();canvas.width=r.width*devicePixelRatio;canvas.height=420*devicePixelRatio;}
resize(); addEventListener('resize',resize);
canvas.addEventListener('pointerdown',e=>{dragging=true;lx=e.clientX;ly=e.clientY;canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointerup',e=>{dragging=false;});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;az+=(e.clientX-lx)*0.008;el+=(e.clientY-ly)*0.008;el=Math.max(-1.5,Math.min(1.5,el));lx=e.clientX;ly=e.clientY;});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom*=e.deltaY>0?0.92:1.08;zoom=Math.max(0.3,Math.min(4,zoom));},{passive:false});

const ROT=`vec3 rot(vec3 p){float ca=cos(u_az),sa=sin(u_az),ce=cos(u_el),se=sin(u_el);
  vec3 r=vec3(p.x*ca+p.z*sa,p.y,-p.x*sa+p.z*ca);
  return vec3(r.x, r.y*ce-r.z*se, r.y*se+r.z*ce);}`;
const TURBO=`vec3 turbo(float t){return clamp(vec3(1.0-1.8*abs(t-0.75),1.0-2.2*abs(t-0.5),1.0-2.0*abs(t-0.25)),0.0,1.0);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
  if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))console.log(gl.getShaderInfoLog(o));return o;}
function mkProg(vs,fs){const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,vs));gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
  p._u={az:gl.getUniformLocation(p,'u_az'),el:gl.getUniformLocation(p,'u_el'),zoom:gl.getUniformLocation(p,'u_zoom'),aspect:gl.getUniformLocation(p,'u_aspect')};return p;}
function setCam(p){gl.useProgram(p);gl.uniform1f(p._u.az,az);gl.uniform1f(p._u.el,el);gl.uniform1f(p._u.zoom,zoom);gl.uniform1f(p._u.aspect,canvas.width/canvas.height);}

// -- points program --
const pProg=mkProg(
 `attribute vec3 a_pos; attribute float a_d; uniform float u_az,u_el,u_zoom,u_aspect; varying float v_d; ${ROT}
  void main(){vec3 r=rot(a_pos);float s=0.026*u_zoom;gl_Position=vec4(r.x*s/u_aspect,r.y*s,r.z*0.004,1.0);gl_PointSize=max(1.5,3.5*u_zoom);v_d=a_d;}`,
 `precision mediump float; varying float v_d; ${TURBO}
  void main(){vec2 c=gl_PointCoord-vec2(0.5);if(dot(c,c)>0.25)discard;gl_FragColor=vec4(turbo(v_d),1.0);}`);
const pPos=gl.getAttribLocation(pProg,'a_pos'), pD=gl.getAttribLocation(pProg,'a_d');
// -- mesh program (lit surface) --
const mProg=mkProg(
 `attribute vec3 a_pos; attribute vec3 a_nrm; attribute float a_d; uniform float u_az,u_el,u_zoom,u_aspect; varying float v_d; varying vec3 v_n; ${ROT}
  void main(){vec3 r=rot(a_pos);float s=0.026*u_zoom;gl_Position=vec4(r.x*s/u_aspect,r.y*s,r.z*0.004,1.0);v_n=rot(a_nrm);v_d=a_d;}`,
 `precision mediump float; varying float v_d; varying vec3 v_n; ${TURBO}
  void main(){vec3 N=normalize(v_n);float df=abs(dot(N,normalize(vec3(0.4,0.7,0.6))));gl_FragColor=vec4(turbo(v_d)*(0.32+0.68*df),1.0);}`);
const mPos=gl.getAttribLocation(mProg,'a_pos'), mNrm=gl.getAttribLocation(mProg,'a_nrm'), mD=gl.getAttribLocation(mProg,'a_d');
// -- paint program (explicit color, glowing dots) --
const paintProg=mkProg(
 `attribute vec3 a_pos; attribute vec3 a_col; uniform float u_az,u_el,u_zoom,u_aspect; varying vec3 v_c; ${ROT}
  void main(){vec3 r=rot(a_pos);float s=0.026*u_zoom;gl_Position=vec4(r.x*s/u_aspect,r.y*s,r.z*0.004-0.001,1.0);gl_PointSize=max(3.0,7.0*u_zoom);v_c=a_col;}`,
 `precision mediump float; varying vec3 v_c; void main(){vec2 c=gl_PointCoord-vec2(0.5);float d=dot(c,c);if(d>0.25)discard;gl_FragColor=vec4(v_c*(1.3-d*2.0),1.0);}`);
const qPos=gl.getAttribLocation(paintProg,'a_pos'), qCol=gl.getAttribLocation(paintProg,'a_col');

const pBuf=gl.createBuffer(), mBuf=gl.createBuffer(), qBuf=gl.createBuffer();
let nPoints=0, nMeshV=0, nPaint=0;
const ok=z=>z>0&&z<MAX_MM;
const wp=(c,r,z)=>[c-RAW_W/2, -(r-RAW_H/2), -(z/22.0-40.0)];

function buildPoints(d){
  const arr=new Float32Array(RAW_W*RAW_H*4);let n=0;
  for(let r=0;r<RAW_H;r++)for(let c=0;c<RAW_W;c++){const z=d[r*RAW_W+c];if(!ok(z))continue;
    const p=wp(c,r,z);arr[n*4]=p[0];arr[n*4+1]=p[1];arr[n*4+2]=p[2];arr[n*4+3]=1-z/MAX_MM;n++;}
  nPoints=n;gl.bindBuffer(gl.ARRAY_BUFFER,pBuf);gl.bufferData(gl.ARRAY_BUFFER,arr.subarray(0,n*4),gl.DYNAMIC_DRAW);
}
function nrm(A,B,C){const u=[B[0]-A[0],B[1]-A[1],B[2]-A[2]],w=[C[0]-A[0],C[1]-A[1],C[2]-A[2]];
  let x=u[1]*w[2]-u[2]*w[1],y=u[2]*w[0]-u[0]*w[2],z=u[0]*w[1]-u[1]*w[0];const L=Math.hypot(x,y,z)||1;return [x/L,y/L,z/L];}
function buildMesh(d){
  const v=[];
  const pushT=(A,B,C,za,zb,zc)=>{const n=nrm(A,B,C);
    v.push(A[0],A[1],A[2],n[0],n[1],n[2],1-za/MAX_MM, B[0],B[1],B[2],n[0],n[1],n[2],1-zb/MAX_MM, C[0],C[1],C[2],n[0],n[1],n[2],1-zc/MAX_MM);};
  for(let r=0;r<RAW_H-1;r++)for(let c=0;c<RAW_W-1;c++){
    const z00=d[r*RAW_W+c],z10=d[r*RAW_W+c+1],z01=d[(r+1)*RAW_W+c],z11=d[(r+1)*RAW_W+c+1];
    if(!ok(z00)||!ok(z10)||!ok(z01)||!ok(z11))continue;
    if(Math.max(z00,z10,z01,z11)-Math.min(z00,z10,z01,z11)>400)continue;  // no rubber sheets over edges
    const p00=wp(c,r,z00),p10=wp(c+1,r,z10),p01=wp(c,r+1,z01),p11=wp(c+1,r+1,z11);
    pushT(p00,p10,p11,z00,z10,z11);pushT(p00,p11,p01,z00,z11,z01);
  }
  nMeshV=v.length/7;gl.bindBuffer(gl.ARRAY_BUFFER,mBuf);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(v),gl.DYNAMIC_DRAW);
}
async function pullCloud(){
  try{const r=await fetch('/depth.bin');const d=new Uint16Array(await r.arrayBuffer());
    if(viewMode===0)buildPoints(d);else buildMesh(d);}catch(e){}
  setTimeout(pullCloud,55);
}
async function pullPaint(){
  try{const r=await fetch('/paint.bin');const ab=await r.arrayBuffer();
    if(ab.byteLength>0){gl.bindBuffer(gl.ARRAY_BUFFER,qBuf);gl.bufferData(gl.ARRAY_BUFFER,ab,gl.DYNAMIC_DRAW);nPaint=ab.byteLength/24;}
    else nPaint=0;}catch(e){}
  setTimeout(pullPaint,150);
}
function render(){
  if(autospin&&!dragging)az+=0.004;
  gl.viewport(0,0,canvas.width,canvas.height);
  gl.clearColor(0.02,0.03,0.05,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if(viewMode===0&&nPoints>0){setCam(pProg);gl.bindBuffer(gl.ARRAY_BUFFER,pBuf);
    gl.enableVertexAttribArray(pPos);gl.vertexAttribPointer(pPos,3,gl.FLOAT,false,16,0);
    gl.enableVertexAttribArray(pD);gl.vertexAttribPointer(pD,1,gl.FLOAT,false,16,12);
    gl.drawArrays(gl.POINTS,0,nPoints);}
  else if(viewMode===1&&nMeshV>0){setCam(mProg);gl.bindBuffer(gl.ARRAY_BUFFER,mBuf);
    gl.enableVertexAttribArray(mPos);gl.vertexAttribPointer(mPos,3,gl.FLOAT,false,28,0);
    gl.enableVertexAttribArray(mNrm);gl.vertexAttribPointer(mNrm,3,gl.FLOAT,false,28,12);
    gl.enableVertexAttribArray(mD);gl.vertexAttribPointer(mD,1,gl.FLOAT,false,28,24);
    gl.drawArrays(gl.TRIANGLES,0,nMeshV);}
  if(nPaint>0){setCam(paintProg);gl.bindBuffer(gl.ARRAY_BUFFER,qBuf);
    gl.enableVertexAttribArray(qPos);gl.vertexAttribPointer(qPos,3,gl.FLOAT,false,24,0);
    gl.enableVertexAttribArray(qCol);gl.vertexAttribPointer(qCol,3,gl.FLOAT,false,24,12);
    gl.drawArrays(gl.POINTS,0,nPaint);}
  requestAnimationFrame(render);
}
function toggleView(){viewMode=1-viewMode;document.getElementById('viewbtn').textContent=viewMode?'▦ View: Mesh':'◎ View: Points';}

// ---------- Theremin (Web Audio, runs in YOUR browser) ----------
let actx=null,osc=null,osc2=null,gain=null,filt=null,thOn=false;
const SCALE=[0,2,4,7,9,12,14,16,19,21,24];   // 2-octave pentatonic -> always musical
function toggleTheremin(){
  thOn=!thOn;document.getElementById('thbtn').textContent='🔊 Theremin: '+(thOn?'on':'off');
  if(thOn){
    if(!actx){actx=new (window.AudioContext||window.webkitAudioContext)();
      osc=actx.createOscillator();osc.type='sawtooth';
      osc2=actx.createOscillator();osc2.type='triangle';osc2.detune.value=6;
      filt=actx.createBiquadFilter();filt.type='lowpass';filt.frequency.value=900;
      gain=actx.createGain();gain.gain.value=0;
      osc.connect(filt);osc2.connect(filt);filt.connect(gain);gain.connect(actx.destination);
      osc.start();osc2.start();}
    actx.resume();
  }else if(gain){gain.gain.setTargetAtTime(0,actx.currentTime,0.05);}
}
function updateTheremin(g){
  if(!thOn||!actx)return;const t=actx.currentTime;
  if(g&&g.presence){
    const idx=Math.max(0,Math.min(SCALE.length-1,Math.round((g.hand_y*0.5+0.5)*(SCALE.length-1))));
    const midi=45+SCALE[idx];const f=440*Math.pow(2,(midi-69)/12);
    osc.frequency.setTargetAtTime(f,t,0.05);osc2.frequency.setTargetAtTime(f,t,0.05);
    filt.frequency.setTargetAtTime(350+(g.hand_x*0.5+0.5)*3200,t,0.05);
    const vol=Math.max(0,Math.min(0.22,(1-Math.min(g.hand_z_mm,3000)/3000)*0.28));
    gain.gain.setTargetAtTime(vol,t,0.04);
  }else{gain.gain.setTargetAtTime(0,t,0.1);}
}

// ---------- Air-draw ----------
async function togglePaint(){const j=await act('paint_toggle');
  const b=document.getElementById('paintbtn');const on=j.on;
  b.classList.toggle('active',on);b.textContent='✏ Air-draw: '+(on?'on':'off');}

poll(); pullCloud(); pullPaint(); render();
</script>
</body></html>
"""


def main():
    print(f"[web] LiDAR Web Studio -> http://0.0.0.0:{PORT}  ({RAW_W}x{RAW_H} @ {PIPE})", flush=True)
    t = threading.Thread(target=sensor_thread, daemon=True)
    t.start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STUDIO.running = False


if __name__ == "__main__":
    main()
