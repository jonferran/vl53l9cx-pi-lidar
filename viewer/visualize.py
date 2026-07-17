#!/usr/bin/env python3
"""
Live VL53L9CX depth viewer.  Binning selectable via env VL_BINNING (2 or 4).

Decodes the ST result buffer streamed over CSI-2 as RAW8:
    [ distance(u16*N) | amplitude(u16*N) | ambient(u16*N) | dss | status-line ]
We render the distance array (raw 15-bit radial values -- uncalibrated, so
absolute mm are approximate, but spatial structure is real).
"""
import cv2
import numpy as np
import time
import os

BINNING = int(os.environ.get("VL_BINNING", "4"))

# (raw_w, raw_h, captured_csi_lines) per binning
GEOM = {
    2: (54, 42, 148),   # CSI 100x149, capture 148
    4: (24, 24, 38),    # CSI 100x39,  capture 38
}
RAW_W, RAW_H, CSI_H = GEOM[BINNING]
N_ZONES = RAW_W * RAW_H
DIST_LEN = N_ZONES * 2
CSI_W = 100
STRIDE = 128
FRAME_BYTES = STRIDE * CSI_H
DIST_MASK = 0x7FFF

PIPE = os.environ.get("VL_PIPE", "tof_pipe")
WIN = f"VL53L9CX Depth {RAW_W}x{RAW_H} (binning {BINNING})"
MIN_MM, MAX_MM = 0, 4000
UPSCALE = 12 if BINNING == 2 else 24

# Orientation: 0=raw, 1=flipV, 2=flipH, 3=rot180.  Press 'r' to cycle live.
ORIENT = int(os.environ.get("VL_ORIENT", "2"))
def orient(a, m):
    if m == 1: return a[::-1, :]
    if m == 2: return a[:, ::-1]
    if m == 3: return a[::-1, ::-1]
    return a

print(f"[viz] {PIPE}  binning={BINNING}  {RAW_W}x{RAW_H} zones  frame={FRAME_BYTES}B", flush=True)
fps_n, t0, fps = 0, time.time(), 0.0

def read_exact(f, n):
    buf = b""
    while len(buf) < n:
        c = f.read(n - len(buf))
        if not c:
            return None
        buf += c
    return buf

try:
    with open(PIPE, "rb") as fifo:
        while True:
            raw = read_exact(fifo, FRAME_BYTES)
            if raw is None:
                time.sleep(0.05); continue
            full = np.frombuffer(raw, np.uint8).reshape((CSI_H, STRIDE))
            linear = full[:, :CSI_W].reshape(-1)
            if linear.size < DIST_LEN:
                continue
            dist = (linear[:DIST_LEN].view("<u2") & DIST_MASK).reshape(RAW_H, RAW_W)
            dist = orient(dist, ORIENT)

            valid = dist[(dist > 0) & (dist < 0x7FFF)]
            med = int(np.median(valid)) if valid.size else 0

            clipped = np.clip(dist, MIN_MM, MAX_MM).astype(np.float32)
            norm = (255.0 * (1.0 - clipped / MAX_MM)).astype(np.uint8)
            color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            color[dist == 0] = (0, 0, 0)
            disp = cv2.resize(color, (RAW_W * UPSCALE, RAW_H * UPSCALE),
                              interpolation=cv2.INTER_NEAREST)

            fps_n += 1
            now = time.time()
            if now - t0 >= 0.5:
                fps = fps_n / (now - t0); fps_n, t0 = 0, now
            cv2.putText(disp, f"{fps:4.0f}FPS med:{med}mm valid:{valid.size}/{N_ZONES} orient:{ORIENT}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                ORIENT = (ORIENT + 1) % 4
                print(f"[viz] orient -> {ORIENT}", flush=True)
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    print("[viz] stopped.")
