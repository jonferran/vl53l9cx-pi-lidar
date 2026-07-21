# LiDAR Web Studio

A networked, browser-based dashboard for the VL53L9CX ToF LiDAR — open it from
your phone, laptop, or any device on the same network and watch the sensor live.
It runs as a standalone companion to the GPU particle viewer, putting the Pi's
otherwise-idle CPU cores, RAM, and disk to work.

```
./run_lidar_web.sh            # boots the sensor, serves on :8080
# then open  http://<pi-ip>:8080  in any browser on the LAN
```

<p align="center"><em>Live 3D point cloud · depth heatmap · gesture recognition · record &amp; replay · PLY export</em></p>

---

## What it does

| Panel | What you get |
|---|---|
| **3D view** | The depth field rendered in raw **WebGL** — drag to orbit, scroll to zoom, auto-spin. Toggle between a **point cloud** and a **lit triangulated surface mesh** (the depth grid stitched into a shaded 3D surface, with silhouette edges left open so it doesn't rubber-sheet). Runs entirely in the browser; the Pi just streams raw depth. |
| **Depth heatmap** | A live MJPEG stream of the colormapped depth grid with the tracked hand overlaid, plus closest-object / mean-depth / active-zone stats. |
| **Gesture recognition** | Real-time hand tracking with **wave / push / pull / swipe-left / swipe-right** detection and a hand-position trail map. |
| **Object detection** | A computer-vision **Vision mode** on the heatmap that segments distinct foreground objects (connected-component analysis), draws **bounding boxes + distance labels**, and counts them live. |
| **LiDAR Theremin** | Play the sensor like an instrument — hand **height sets pitch** (quantized to a pentatonic scale so it's always musical), **distance sets volume**, **sideways sets tone/filter**, **swipe fires a drum hit**, **push plucks a bass note**. The Pi tracks your hand; the audio is synthesized in **your browser** via Web Audio, so no speakers on the Pi are needed. |
| **Ambient soundscape** | A generative drone-pad that follows the *whole room* — depth brightness modulates the tone, with slow LFOs and a feedback-delay space. **Listen to the room.** Also browser-side Web Audio. |
| **Air-drawing** | Trace your fingertip through 3D space to paint a persistent point trail into the cloud — sculpt in the air, then export the drawing as its own `.ply`. |
| **Long-exposure 3D scan** | Fuse many seconds of frames into one **dense, denoised, gap-filled** cloud (a zone that's valid in *any* frame gets filled) — turns the live sensor into a capture tool. View the frozen result and export it. |
| **Hands-free control** | Control the studio by gesture — **swipe** switches the 3D view, **wave** resets the background — no keyboard or mouse. |
| **Record clip** | Capture the spinning 3D view to a downloadable **WebM video** (via the browser's MediaRecorder) — a shareable clip of your cloud. |
| **Sentry mode** | Arm it and the studio becomes a **privacy-friendly smart camera**: it auto-records a depth clip whenever motion is detected and stops when the scene clears, with a timestamped **event log**. Captures *depth*, not video. |
| **Record & replay** | One-click recording of the depth stream to disk (`.ldr` files), replayed right in the same UI. |
| **PLY export** | Export the current frame as a standard **`.ply` point cloud** — plus an **HD snapshot** that temporally averages recent frames to denoise it. Opens in MeshLab, CloudCompare, Blender, or any online point-cloud viewer, and is easy to share. |

Everything updates in real time and multiple devices can watch at once.

---

## How it's built

- **Pure standard library** for the server (`http.server` + `socketserver`
  threading mix-in) — no Flask, nothing to `pip install`. Just `numpy` + `cv2`,
  which the sensor pipeline already needs.
- A **background reader thread** drains the sensor FIFO at the full 100 Hz,
  decodes each RAW8 frame into a depth grid, runs gesture recognition, and
  publishes the latest frame + stats into a lock-guarded shared state.
- **Threaded HTTP** serves many clients at once. Endpoints:
  - `GET /` — the single-page dashboard (HTML + inline WebGL + Web Audio, no external assets)
  - `GET /stream.mjpg` — `multipart/x-mixed-replace` depth heatmap (~25 fps)
  - `GET /depth.bin` — the latest frame as raw `uint16` (feeds the WebGL cloud/mesh, ~18 Hz)
  - `GET /paint.bin` — the accumulated air-drawing points (`float32` x,y,z,r,g,b)
  - `GET /stats.json` — live stats, gesture state, recording list, paint state
  - `GET /snapshot.ply`, `GET /snapshot_hd.ply`, `GET /scan.ply` — current / averaged / fused cloud as PLY
  - `GET /scan.bin` — the fused long-exposure scan as raw `uint16`
  - `GET /paint.ply` — the air-drawing as a PLY download
  - `GET /action?cmd=…` — record, replay, background reset, air-draw, scan start/stop
- **Gesture engine** (`web/lidar_gestures.py`) is dependency-light and
  **unit-tested with synthetic frames** (`python3 lidar_gestures.py`). It keeps a
  slow background model, segments the **largest connected foreground blob** (so
  scattered sensor noise isn't tracked as a phantom hand), follows the centroid
  over a short time-windowed history, and classifies gestures from real elapsed
  time (framerate-independent).

## Files

| Path | Role |
|---|---|
| `web/lidar_web.py` | The server: sensor reader thread + HTTP endpoints + the inline dashboard/WebGL front-end. |
| `web/lidar_gestures.py` | Standalone, testable gesture-recognition engine. |
| `run_lidar_web.sh` | Launcher — boots the sensor (same bring-up as the particle viewer) and starts the server. Pass a port as `$1` to override 8080. |
| `scripts/lidar_web.desktop` | Desktop launcher icon. |

## Demo mode (no sensor)

`VL_SYNTH=1 VL_PORT=8081 python3 web/lidar_web.py` serves the full dashboard
driven by a synthetic moving-hand scene — handy for developing the UI or showing
it off without the sensor attached.

## Notes

- The web studio and the particle viewer both consume the sensor, so run one or
  the other (the launcher clears prior sensor processes on start).
- Recordings live in `web/recordings/` on the Pi.
- Gesture thresholds are all tunables at the top of `GestureEngine.__init__`.
