#!/usr/bin/env python3
"""
Gesture recognition engine for the VL53L9CX ToF LiDAR web studio.

Pure NumPy, no sensor / GL / network dependencies, so it can be unit-tested
with synthetic depth frames. Given a stream of (RAW_H x RAW_W) depth-in-mm
grids it maintains a slow background model, segments the nearest foreground
"hand" blob, tracks its centroid over time, and classifies coarse gestures:

    presence   -- is a hand in front of the sensor at all
    hand_xyz   -- normalized hand centroid (x,y in [-1,1], z in metres)
    openness   -- foreground area as a rough open/closed proxy (0..1)
    wave       -- lateral oscillation (hand waving side to side)
    push/pull  -- sustained motion toward / away from the sensor
    swipe      -- a quick left / right lateral flick (edge-triggered event)

The classifiers work off a short deque of recent centroids + timestamps, so
they're framerate-independent (they use real elapsed time, not frame counts).
"""
import time
import collections
import numpy as np
try:
    import cv2                       # for largest-connected-component segmentation
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


class GestureEngine:
    def __init__(self, raw_w, raw_h, max_mm=4000):
        self.w = raw_w
        self.h = raw_h
        self.max_mm = max_mm
        self.bg = None                 # slow background depth estimate (mm)
        self.smooth = None             # fast EMA of depth (mm), denoises segmentation
        self.hist = collections.deque(maxlen=90)   # (t, x, y, z_mm, area) samples
        self.swipe_cooldown = 0.0      # timestamp until which new swipes are ignored
        self.last_swipe = ""           # "left" / "right" / "" -- last fired swipe
        self.last_swipe_t = 0.0

        # Tunables
        self.BG_ALPHA = 0.02           # background adaptation rate (~slow)
        self.SMOOTH_ALPHA = 0.4        # fast denoise EMA
        self.FG_THRESH_MM = 120.0      # min diff from background to be "foreground"
        self.MIN_AREA = 6              # min foreground zones (largest blob) to be a hand
        self.SWIPE_MIN_DX = 0.9        # normalized-x travel over the window to fire
        self.SWIPE_WINDOW = 0.4        # seconds to accumulate a swipe over
        self.SWIPE_COOLDOWN = 0.6      # seconds to suppress repeat swipes
        self.WAVE_MIN_CROSSINGS = 4    # direction reversals within the wave window
        self.WAVE_WINDOW = 1.2         # seconds
        self.WAVE_MIN_AMP = 0.25       # min normalized-x amplitude to count as waving
        self.PUSHPULL_WINDOW = 0.5     # seconds
        self.PUSHPULL_MIN_MM = 120.0   # z travel over the window to fire push/pull

    def reset_background(self, depth_mm=None):
        """Snap the background to the current frame (clears stale reference)."""
        self.bg = None if depth_mm is None else depth_mm.astype(np.float32).copy()
        self.smooth = None

    def update(self, depth_mm, now=None):
        """Ingest one depth frame (mm, shape (h,w)); return a gesture dict."""
        if now is None:
            now = time.time()
        depth = depth_mm.astype(np.float32)
        valid = (depth > 0) & (depth < self.max_mm)

        # Fast denoise EMA (segmentation stability).
        if self.smooth is None or self.smooth.shape != depth.shape:
            self.smooth = depth.copy()
        else:
            m = valid
            self.smooth[m] += self.SMOOTH_ALPHA * (depth[m] - self.smooth[m])

        # Slow background model (what the empty scene looks like).
        if self.bg is None or self.bg.shape != depth.shape:
            self.bg = self.smooth.copy()
        else:
            self.bg += self.BG_ALPHA * (self.smooth - self.bg)

        # Foreground = closer than background by a margin (a hand approaching).
        diff = self.bg - self.smooth              # positive where something is nearer
        fg = valid & (diff > self.FG_THRESH_MM)
        # Connected-component analysis: the LARGEST blob is the tracked "hand"
        # (rejecting scattered noise), while ALL blobs above MIN_AREA become
        # the object-detection list for the vision panel.
        objects = []
        if _HAVE_CV2 and fg.any():
            nlab, labels, st, cent = cv2.connectedComponentsWithStats(
                fg.astype(np.uint8), connectivity=8)
            if nlab > 1:
                order = np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1
                for lab in order:
                    a = int(st[lab, cv2.CC_STAT_AREA])
                    if a < self.MIN_AREA:
                        continue
                    x0 = int(st[lab, cv2.CC_STAT_LEFT]); y0 = int(st[lab, cv2.CC_STAT_TOP])
                    x1 = x0 + int(st[lab, cv2.CC_STAT_WIDTH]); y1 = y0 + int(st[lab, cv2.CC_STAT_HEIGHT])
                    m = labels == lab
                    dist = int(self.smooth[m].mean())
                    objects.append({"area": a, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                    "cx": float(cent[lab][0]), "cy": float(cent[lab][1]),
                                    "dist_mm": dist})
                biggest = int(order[0])
                fg = labels == biggest
        area = int(fg.sum())

        out = {
            "presence": False, "hand_x": 0.0, "hand_y": 0.0, "hand_z_mm": 0.0,
            "openness": 0.0, "area": area, "wave": False, "pushpull": "",
            "swipe": "", "waving_amp": 0.0, "objects": objects,
        }

        if area >= self.MIN_AREA:
            ys, xs = np.nonzero(fg)
            # Weight the centroid by "how foreground" each zone is, so the
            # closest part of the hand dominates -> steadier tracking.
            wsum = diff[ys, xs]
            wtot = float(wsum.sum()) + 1e-6
            cx = float((xs * wsum).sum() / wtot)
            cy = float((ys * wsum).sum() / wtot)
            cz = float(self.smooth[ys, xs].mean())
            nx = (cx / (self.w - 1)) * 2.0 - 1.0          # -1 (left) .. +1 (right)
            ny = -((cy / (self.h - 1)) * 2.0 - 1.0)       # +1 (up) .. -1 (down)
            openness = min(area / (0.35 * self.w * self.h), 1.0)

            out.update(presence=True, hand_x=nx, hand_y=ny, hand_z_mm=cz,
                       openness=openness)
            self.hist.append((now, nx, ny, cz, area))
            self._classify(now, out)
        else:
            # Let stale history age out so gestures don't linger after the
            # hand leaves.
            while self.hist and now - self.hist[0][0] > self.WAVE_WINDOW:
                self.hist.popleft()

        return out

    def _classify(self, now, out):
        # Trim history to the longest window we care about.
        while self.hist and now - self.hist[0][0] > self.WAVE_WINDOW:
            self.hist.popleft()
        if len(self.hist) < 4:
            return
        ts = np.array([s[0] for s in self.hist])
        xs = np.array([s[1] for s in self.hist])
        zs = np.array([s[3] for s in self.hist])

        # --- WAVE: many left/right reversals with meaningful amplitude ---
        wmask = (now - ts) <= self.WAVE_WINDOW
        wx = xs[wmask]
        if wx.size >= 5:
            amp = float(wx.max() - wx.min())
            dx = np.diff(wx)
            sign = np.sign(dx)
            sign = sign[sign != 0]
            crossings = int((np.diff(sign) != 0).sum()) if sign.size >= 2 else 0
            out["waving_amp"] = amp
            if crossings >= self.WAVE_MIN_CROSSINGS and amp >= self.WAVE_MIN_AMP:
                out["wave"] = True

        # --- PUSH / PULL: sustained z travel over the short window ---
        pmask = (now - ts) <= self.PUSHPULL_WINDOW
        pz = zs[pmask]
        if pz.size >= 3:
            dz = pz[-1] - pz[0]        # negative = got nearer (push toward sensor)
            if dz <= -self.PUSHPULL_MIN_MM:
                out["pushpull"] = "push"
            elif dz >= self.PUSHPULL_MIN_MM:
                out["pushpull"] = "pull"

        # --- SWIPE: quick net lateral travel, edge-triggered w/ cooldown ---
        smask = (now - ts) <= self.SWIPE_WINDOW
        sx = xs[smask]
        if sx.size >= 3 and now >= self.swipe_cooldown:
            travel = sx[-1] - sx[0]
            if abs(travel) >= self.SWIPE_MIN_DX:
                out["swipe"] = "right" if travel > 0 else "left"
                self.last_swipe = out["swipe"]
                self.last_swipe_t = now
                self.swipe_cooldown = now + self.SWIPE_COOLDOWN


# --------------------------------------------------------------------------
# Self-test: synthesize depth frames with a moving "hand" blob and confirm the
# engine reports the right gestures. Run:  python3 lidar_gestures.py
# --------------------------------------------------------------------------
def _synth_frame(w, h, bg_mm, hand_center=None, hand_z=None, hand_r=6.0):
    frame = np.full((h, w), bg_mm, dtype=np.float32)
    if hand_center is not None:
        cx, cy = hand_center
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= hand_r ** 2
        frame[mask] = hand_z
    return frame


def _run_selftest():
    w, h = 54, 42
    bg = 1800.0
    eng = GestureEngine(w, h)
    t = 0.0
    dt = 0.02   # 50 fps synthetic

    # Warm the background with empty frames.
    for _ in range(120):
        eng.update(_synth_frame(w, h, bg), now=t); t += dt

    print("== SWIPE RIGHT ==")
    got = ""
    for k in range(20):
        cx = 6 + k * 2.0           # sweep left->right fast
        g = eng.update(_synth_frame(w, h, bg, (min(cx, w - 6), h // 2), 900), now=t)
        t += dt
        if g["swipe"]:
            got = g["swipe"]
    print("  presence during swipe:", g["presence"], "| swipe fired:", got)
    assert got == "right", f"expected right swipe, got {got!r}"

    # settle / clear
    for _ in range(120):
        eng.update(_synth_frame(w, h, bg), now=t); t += dt

    print("== WAVE ==")
    waved = False
    for k in range(80):
        cx = w / 2 + 12 * np.sin(k * 0.6)   # oscillate side to side
        g = eng.update(_synth_frame(w, h, bg, (cx, h // 2), 900), now=t)
        t += dt
        waved = waved or g["wave"]
    print("  wave detected:", waved, "| amp:", round(g["waving_amp"], 2))
    assert waved, "expected wave detection"

    for _ in range(120):
        eng.update(_synth_frame(w, h, bg), now=t); t += dt

    print("== PUSH ==")
    pushed = ""
    for k in range(20):
        z = 1500 - k * 40          # hand comes toward sensor
        g = eng.update(_synth_frame(w, h, bg, (w // 2, h // 2), z), now=t)
        t += dt
        if g["pushpull"]:
            pushed = g["pushpull"]
    print("  pushpull:", pushed, "| hand_z_mm:", round(g["hand_z_mm"]))
    assert pushed == "push", f"expected push, got {pushed!r}"

    print("\nALL GESTURE SELF-TESTS PASSED")


if __name__ == "__main__":
    _run_selftest()
