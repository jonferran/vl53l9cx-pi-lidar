#!/usr/bin/env python3
"""
Live VL53L9CX viewer.  Binning selectable via env VL_BINNING (2 or 4).

Decodes the ST result buffer streamed over CSI-2 as RAW8:
    [ distance(u16*N) | amplitude(u16*N) | ambient(u16*N) | dss(4bit*N) | status-line ]
Distance is raw 15-bit radial (uncalibrated). Amplitude/ambient are raw counts.
(DSS is skipped -- disabled on the sensor for 100fps, so it's always empty.)

Keys: r = rotate/flip, c = cycle channel, m = cycle colormap, v = toggle 3D view,
      f = front view, 3 = 3/4 view, z = fullscreen, b = blur (2D), l = log color,
      q = quit.
3D mode: left-drag orbits the camera, right-drag zooms at the cursor. Scale%
(cloud size) and ZDepth% (log-scaled depth exaggeration) are the panel sliders.
"""
import cv2
import numpy as np
import time
import os
import array
import fcntl
import termios
import math

def _get_screen_size():
    """Real desktop resolution via Tk (fast, one-shot, no visible window) --
    far more reliable than querying our own OpenCV/Qt window's geometry right
    after a fullscreen toggle, which races the window manager's transition."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 0, 0

SCREEN_W, SCREEN_H = _get_screen_size()

BINNING = int(os.environ.get("VL_BINNING", "4"))

GEOM = {
    2: (54, 42, 148),   # CSI 100x149, capture 148
    4: (24, 24, 38),    # CSI 100x39,  capture 38
}
RAW_W, RAW_H, CSI_H = GEOM[BINNING]
N_ZONES = RAW_W * RAW_H
CH_LEN = N_ZONES * 2          # bytes per u16 channel (distance/amplitude/ambient)
CSI_W = 100
STRIDE = 128
FRAME_BYTES = STRIDE * CSI_H
DIST_MASK = 0x7FFF

PIPE = os.environ.get("VL_PIPE", "tof_pipe")
WIN = f"VL53L9CX {RAW_W}x{RAW_H} (binning {BINNING})"
MIN_MM, MAX_MM = 0, 4000
UPSCALE = 18 if BINNING == 2 else 32
DEFAULT_IMG_W, DEFAULT_IMG_H = RAW_W * UPSCALE, RAW_H * UPSCALE
IMG_W, IMG_H = DEFAULT_IMG_W, DEFAULT_IMG_H   # updated on fullscreen toggle

ORIENT = int(os.environ.get("VL_ORIENT", "0"))       # 0=raw 1=flipV 2=flipH 3=rot180
def orient(a, m):
    if m == 1: return a[::-1, :]
    if m == 2: return a[:, ::-1]
    if m == 3: return a[::-1, ::-1]
    return a

CHANNELS = ["distance", "amplitude", "ambient"]   # 'c' cycles live
chan_idx = 0

COLORMAPS = [
    ("jet", cv2.COLORMAP_JET),
    ("turbo", cv2.COLORMAP_TURBO),
    ("viridis", cv2.COLORMAP_VIRIDIS),
    ("hot", cv2.COLORMAP_HOT),
    ("inferno", cv2.COLORMAP_INFERNO),
]                                                          # 'm' cycles live
map_idx = 0

view3d = False                                             # 'v' toggles
fullscreen = False                                          # 'z' toggles
blur = False                                                # 'b' toggles (2D only)
log_color = False                                           # 'l' toggles -- log-scaled colour, more near-range detail
fs_full_w, fs_full_h = 0, 0                                 # raw fullscreen size, set once on 'z'

def color_norm(vals, vmax):
    """Map a value array to a 0-255 'closeness' scale (255=near/0mm, 0=vmax).
    Log-scaled when log_color is on: compresses far-range color steps so more
    of the 0-255 range is spent distinguishing near-range detail."""
    clipped = np.clip(vals, 0, vmax).astype(np.float32)
    if log_color:
        frac = np.log1p(clipped) / np.log1p(vmax)
    else:
        frac = clipped / vmax
    return np.clip(255.0 * (1.0 - frac), 0, 255).astype(np.uint8)

# --- Compact right-hand slider panel (3D mode only) -------------------------
PANEL_W = 130
SLIDERS = [  # name, min, max, default -- Scale = cloud geometry size, ZDepth = log-depth
             # exaggeration, Curve = log knee in mm (lower = sharper near/far compression)
    ("Scale%", 10, 300, 100),
    ("ZDepth%", 0, 200, 100),
    ("Curve(mm)", 10, 2000, 300),
]
slider_val = [d for (_, _, _, d) in SLIDERS]
SLIDER_Y0, SLIDER_ROW_H, SLIDER_BAR_H, SLIDER_BAR_W = 50, 46, 10, 100
SLIDER_PAD_X = 15
dragging = -1

# Mouse-driven camera state (3D mode). Orbit via drag, zoom+pan via scroll.
cam_azimuth = 45.0     # degrees
cam_tilt = 25.0        # degrees, clamped to +/-89
cam_zoom = 1.0         # scroll-driven camera zoom (screen-space)
cam_pan_x, cam_pan_y = 0.0, 0.0
_drag_last = None

def slider_rect(i):
    x0 = IMG_W + SLIDER_PAD_X
    y0 = SLIDER_Y0 + i * SLIDER_ROW_H
    return x0, y0, x0 + SLIDER_BAR_W, y0 + SLIDER_BAR_H

def draw_panel(canvas):
    x0 = IMG_W
    cv2.rectangle(canvas, (x0, 0), (x0 + PANEL_W, IMG_H), (40, 40, 40), -1)
    cv2.putText(canvas, "Cloud", (x0 + SLIDER_PAD_X, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    for i, (name, lo, hi, _) in enumerate(SLIDERS):
        bx0, by0, bx1, by1 = slider_rect(i)
        cv2.putText(canvas, f"{name} {slider_val[i]}", (bx0, by0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (90, 90, 90), -1)
        frac = (slider_val[i] - lo) / (hi - lo)
        fill_x = bx0 + int(frac * (bx1 - bx0))
        cv2.rectangle(canvas, (bx0, by0), (fill_x, by1), (0, 200, 255), -1)
    hint_y = SLIDER_Y0 + len(SLIDERS) * SLIDER_ROW_H + 10
    for line in ("L-drag: orbit", "R-drag: zoom"):
        cv2.putText(canvas, line, (x0 + SLIDER_PAD_X, hint_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (140, 140, 140), 1, cv2.LINE_AA)
        hint_y += 18

# Right-drag zoom instead of the scroll wheel: OpenCV's Qt backend has its own
# built-in wheel-zoom baked into the native viewport widget (confirmed via
# cv2.getBuildInformation() -- GUI: QT5) that intercepts wheel events before
# our callback and can't be disabled via a public flag while keeping the
# window resizable. Right-drag isn't claimed by that widget (same reason
# left-drag orbit already works cleanly), so it's used for zoom instead.
_zoom_anchor = None   # (start_y, start_zoom, start_pan_x, start_pan_y, cursor_x, cursor_y)

def on_mouse(event, x, y, flags, _param):
    global dragging, _drag_last, cam_azimuth, cam_tilt, cam_zoom, cam_pan_x, cam_pan_y, _zoom_anchor
    if not view3d:
        return
    over_panel = x >= IMG_W

    if event == cv2.EVENT_LBUTTONDOWN:
        if over_panel:
            for i in range(len(SLIDERS)):
                bx0, by0, bx1, by1 = slider_rect(i)
                if bx0 - 4 <= x <= bx1 + 4 and by0 - 4 <= y <= by1 + 4:
                    dragging = i
                    break
        else:
            _drag_last = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = -1
        _drag_last = None
    elif event == cv2.EVENT_RBUTTONDOWN and not over_panel:
        _zoom_anchor = (y, cam_zoom, cam_pan_x, cam_pan_y, x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        _zoom_anchor = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging >= 0 and (flags & cv2.EVENT_FLAG_LBUTTON):
            bx0, _, bx1, _ = slider_rect(dragging)
            lo, hi = SLIDERS[dragging][1], SLIDERS[dragging][2]
            frac = max(0.0, min(1.0, (x - bx0) / (bx1 - bx0)))
            slider_val[dragging] = int(round(lo + frac * (hi - lo)))
        elif _drag_last is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            dx, dy = x - _drag_last[0], y - _drag_last[1]
            cam_azimuth = (cam_azimuth + dx * 0.3) % 360.0
            cam_tilt = max(-89.0, min(89.0, cam_tilt - dy * 0.3))
            _drag_last = (x, y)
        elif _zoom_anchor is not None and (flags & cv2.EVENT_FLAG_RBUTTON):
            start_y, start_zoom, start_pan_x, start_pan_y, ax, ay = _zoom_anchor
            new_zoom = max(0.1, min(20.0, start_zoom * (1.01 ** (start_y - y))))
            # Keep the point under the drag's start position fixed on screen.
            cx, cy = IMG_W / 2.0, IMG_H / 2.0
            wx = (ax - cx - start_pan_x) / start_zoom
            wy = (ay - cy - start_pan_y) / start_zoom
            cam_zoom = new_zoom
            cam_pan_x = ax - cx - wx * new_zoom
            cam_pan_y = ay - cy - wy * new_zoom

print(f"[viz] {PIPE}  binning={BINNING}  {RAW_W}x{RAW_H} zones  frame={FRAME_BYTES}B", flush=True)

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

# WINDOW_GUI_NORMAL drops the GTK toolbar/status-bar and its built-in
# interactive zoom/pan -- that was fighting our own scroll-to-zoom handler.
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(WIN, IMG_W, IMG_H)
cv2.setMouseCallback(WIN, on_mouse)

fps_n, t0, fps = 0, time.time(), 0.0
stat_med, stat_valid = 0, 0

def render_3d(dist_mm, color_vals, color_vmax, cmap, canvas_w, canvas_h):
    """Point cloud: XY from the (optionally upsampled) zone grid, Z from a
    log-scaled distance so nearby geometry gets more visual separation than
    far geometry. Colour from whichever channel is currently selected.
    Camera: mouse-drag orbits, scroll zooms at the cursor; Scale%/ZDepth%/Curve
    are the side-panel sliders. Grid density scales up with zoom -- more points
    only where they're actually visible, cheap when zoomed out."""
    cloud_scale = max(slider_val[0], 1) / 100.0
    zdepth = slider_val[1] / 100.0
    curve_k = max(slider_val[2], 1)
    upsample = max(1, min(4, int(round(cam_zoom))))

    h, w = dist_mm.shape
    if upsample > 1:
        dist_up = cv2.resize(dist_mm, (w * upsample, h * upsample), interpolation=cv2.INTER_LINEAR)
        color_up = cv2.resize(color_vals.astype(np.float32), (w * upsample, h * upsample),
                              interpolation=cv2.INTER_LINEAR)
        h, w = dist_up.shape
    else:
        dist_up, color_up = dist_mm, color_vals.astype(np.float32)

    valid = dist_up > 0
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((canvas_h, canvas_w, 3), np.uint8)

    z = dist_up[ys, xs].astype(np.float32)
    x = (xs - w / 2.0).astype(np.float32) * cloud_scale / upsample
    y = (ys - h / 2.0).astype(np.float32) * -1.0 * cloud_scale / upsample
    # Log-scale depth: near geometry spreads out, far geometry compresses forward.
    # curve_k is the knee (mm) -- smaller = sharper near/far compression, larger = flatter/linear-like.
    log_z = np.log1p(z / curve_k)
    zc = (log_z - log_z.mean()) * (32.0 * zdepth * cloud_scale)

    azimuth, tilt = math.radians(cam_azimuth), math.radians(cam_tilt)
    ct, st = math.cos(azimuth), math.sin(azimuth)
    xr = x * ct + zc * st
    zr = -x * st + zc * ct
    yr = y * math.cos(tilt) - zr * math.sin(tilt)
    depth_order = np.argsort(zr)  # painter's algorithm: far first

    scale = (canvas_w / (w / upsample * 1.8)) * cam_zoom
    sx = (canvas_w / 2 + cam_pan_x + xr * scale).astype(np.int32)
    sy = (canvas_h / 2 + cam_pan_y + yr * scale).astype(np.int32)

    cvals = color_up[ys, xs]
    norm = color_norm(cvals, color_vmax)
    colors = cv2.applyColorMap(norm.reshape(-1, 1), cmap).reshape(-1, 3)

    # Vectorized "voxel" scatter: one fancy-index assignment (sorted far-to-near
    # so nearer points win on overlap) instead of a per-point cv2.circle loop --
    # that loop was the actual bottleneck (~18fps with thousands of points).
    # A single cv2.dilate then thickens each 1px dot into a flat colour block.
    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
    in_bounds = (sx >= 0) & (sx < canvas_w) & (sy >= 0) & (sy < canvas_h)
    order = depth_order[in_bounds[depth_order]]
    canvas[sy[order], sx[order]] = colors[order]

    r = min(max(1, int(scale * 0.55 / upsample)), 15)
    if r > 1:
        kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        canvas = cv2.dilate(canvas, kernel)
    return canvas

try:
    with open(PIPE, "rb", buffering=0) as fifo:
        nready = array.array("i", [0])
        fd = fifo.fileno()
        while True:
            # Block for one frame, then drain any already-buffered whole frames so
            # v4l2 never stalls on a full FIFO -- keeps capture at a true 100 fps
            # while we render only the newest frame.
            raw = read_exact(fifo, FRAME_BYTES)
            if raw is None:
                time.sleep(0.05); continue
            fps_n += 1
            fcntl.ioctl(fd, termios.FIONREAD, nready, True)
            while nready[0] >= FRAME_BYTES:
                nxt = read_exact(fifo, FRAME_BYTES)
                if nxt is None:
                    break
                raw = nxt
                fps_n += 1
                nready[0] -= FRAME_BYTES

            full = np.frombuffer(raw, np.uint8).reshape((CSI_H, STRIDE))
            linear = full[:, :CSI_W].reshape(-1)
            if linear.size < 3 * CH_LEN:
                continue

            # 3D's point-cloud axes need a different flip than the flat 2D image to
            # look "upright" -- orient 3 in the old scheme == orient 0 in 3D mode,
            # so apply a fixed +3 offset in 3D and keep 2D as-is. User-facing
            # orient 0 is now correct by default in both.
            eff_orient = (ORIENT + 3) % 4 if view3d else ORIENT
            dist_mm = orient((linear[0:CH_LEN].view("<u2") & DIST_MASK).reshape(RAW_H, RAW_W), eff_orient)

            chan = CHANNELS[chan_idx]
            if chan == "distance":
                vals, vmax = dist_mm, MAX_MM
            else:
                off = CH_LEN if chan == "amplitude" else 2 * CH_LEN
                vals = orient(linear[off:off + CH_LEN].view("<u2").reshape(RAW_H, RAW_W), eff_orient)
                vmax = 2000  # rough display ceiling for amplitude/ambient counts

            cmap_name, cmap = COLORMAPS[map_idx]

            # Recompute the render size each frame from the stored raw fullscreen
            # size (cheap) rather than the window (no per-frame querying -- that
            # caused a shrink feedback loop before).
            if fullscreen and fs_full_w > 0:
                IMG_W = fs_full_w - (PANEL_W if view3d else 0)
                IMG_H = fs_full_h
            else:
                IMG_W, IMG_H = DEFAULT_IMG_W, DEFAULT_IMG_H

            if view3d:
                img = render_3d(dist_mm, vals, vmax, cmap, IMG_W, IMG_H)
            else:
                norm = color_norm(vals, vmax)
                color = cv2.applyColorMap(norm, cmap)
                color[vals == 0] = (0, 0, 0)
                interp = cv2.INTER_LINEAR if blur else cv2.INTER_NEAREST
                img = cv2.resize(color, (IMG_W, IMG_H), interpolation=interp)
                if blur:
                    img = cv2.GaussianBlur(img, (0, 0), sigmaX=UPSCALE / 3.0)

            if view3d:
                disp = np.zeros((IMG_H, IMG_W + PANEL_W, 3), np.uint8)
                disp[:, :IMG_W] = img
                draw_panel(disp)
            else:
                disp = img

            now = time.time()
            if now - t0 >= 0.5:
                fps = fps_n / (now - t0); fps_n, t0 = 0, now
                nz = vals[vals > 0]
                stat_med = int(np.median(nz)) if nz.size else 0
                stat_valid = int(nz.size)

            mode = ("3D" if view3d else "2D" + (" blur" if blur else "")) + (" log" if log_color else "")
            cv2.putText(disp, f"{fps:4.0f}FPS {mode} {chan}/{cmap_name} med:{stat_med} valid:{stat_valid}/{N_ZONES} orient:{ORIENT}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                ORIENT = (ORIENT + 1) % 4
                print(f"[viz] orient -> {ORIENT}", flush=True)
            elif k == ord('c'):
                chan_idx = (chan_idx + 1) % len(CHANNELS)
                print(f"[viz] channel -> {CHANNELS[chan_idx]}", flush=True)
            elif k == ord('m'):
                map_idx = (map_idx + 1) % len(COLORMAPS)
                print(f"[viz] colormap -> {COLORMAPS[map_idx][0]}", flush=True)
            elif k == ord('v'):
                view3d = not view3d
                print(f"[viz] 3D view -> {view3d}", flush=True)
            elif k == ord('f'):
                cam_azimuth, cam_tilt = 0.0, 0.0
                cam_zoom, cam_pan_x, cam_pan_y = 1.0, 0.0, 0.0
                print("[viz] camera -> front", flush=True)
            elif k == ord('3'):
                cam_azimuth, cam_tilt = 45.0, 25.0
                cam_zoom, cam_pan_x, cam_pan_y = 1.0, 0.0, 0.0
                print("[viz] camera -> 3/4 view", flush=True)
            elif k == ord('z'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                       cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if fullscreen:
                    fs_full_w, fs_full_h = SCREEN_W, SCREEN_H
                else:
                    cv2.resizeWindow(WIN, DEFAULT_IMG_W + (PANEL_W if view3d else 0), DEFAULT_IMG_H)
                print(f"[viz] fullscreen -> {fullscreen}", flush=True)
            elif k == ord('b'):
                blur = not blur
                print(f"[viz] blur -> {blur}", flush=True)
            elif k == ord('l'):
                log_color = not log_color
                print(f"[viz] log color -> {log_color}", flush=True)
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    print("[viz] stopped.")
