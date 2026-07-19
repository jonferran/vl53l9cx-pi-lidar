# GPU Particle Viewer — Controls Reference

This documents every slider, button, and keyboard shortcut in `viewer/visualize_gpu.py`,
the GPU-shader particle visualizer for the VL53L9CX ToF LiDAR. It assumes some
familiarity with the pipeline (sensor → per-zone depth grid → particle spawn → GPU
simulation), but explains what each control actually changes under the hood so you can
reason about interactions between them, not just "turn it up/down."

## How the pipeline works, briefly

Each frame, the CPU decodes a small depth grid from the sensor (54×42 zones at binning=2),
computes per-zone **motion** (how different this zone is from a slowly-adapting background
estimate) and **optical flow** (per-zone apparent x/y velocity between consecutive frames,
via Farneback), then decides which zones spawn a new particle this frame and with what
initial color/size/velocity. Once a particle is born, its entire lifetime — position, size,
color, fade — is recomputed **every frame on the GPU** as a pure function of its birth
attributes and the current time. The CPU never touches an existing particle again; this is
what lets it handle hundreds of thousands of particles at high framerates.

This split matters for understanding the sliders: some control what happens **at spawn time**
(density, initial velocity, spawn color) and some control the **ongoing GPU simulation**
(speed, noise, size decay, rendering). A few (MotionRef, MoveFX) sit right at the boundary
and shape how CPU-side motion sensing turns into spawn parameters.

---

## Sliders

### Density%  *(2–100, default 100)*
Multiplies the per-zone spawn probability uniformly across the whole scene. Doesn't change
*how* a zone's motion is interpreted (that's MotionRef/MoveFX) — just scales the odds that a
qualifying zone actually spawns a particle this frame. Lower it to thin out the cloud without
changing how "energetic" individual particles look.

### Size  *(1–20, default 5)*
Base point-sprite size in pixels, before per-particle scaling. Two things scale on top of this
automatically and can't be disabled: the particle's own motion (faster-moving particles render
larger) and its lifetime fraction (every particle now shrinks from full size at spawn to exactly
0 at end of life — see "Rendering details" below). Size does **not** affect edge blurriness —
the fragment shader keeps a constant-width antialiased edge at any size, so bigger particles
stay crisp rather than turning into soft blobs that melt together under additive blending.

### Speed  *(0.1–20, default 2)*
Scales two things at once: (1) the particle's inherited ballistic velocity — the real,
optical-flow-measured motion it was born with, which decays exponentially over its life
(`v(age) = v0 · e^(−3·age)`, so it's fastest the instant it spawns and monotonically slows down);
and (2) partially scales the ambient flow-field wobble's amplitude too (see NoiseAmp), so
turning Speed up makes the whole field feel bigger and more turbulent, not just individual
particles darting around faster in a straight line.

### Freq  *(0.01–0.155, default 0.06)*
Spatial frequency of the noise field that drives ambient particle wobble (and the debug flow
field's visualization). Lower = larger, slower-varying swirls; higher = tighter, busier curls.
(Historical note: this slider's range used to go to 0.30 — the max was halved, so what used to
sit at 50% is now 100%, per user preference for the lower half of the old range.)

### Life(s)  *(0.3–3.0, default 1.2)*
Base particle lifetime in seconds before fully fading out. The *effective* lifetime a given
particle actually gets is this value scaled by its own motion at spawn
(`eff_life = Life × clamp(0.35 + motion×0.7, 0.35, 2.5)`) — calmer particles live proportionally
shorter, energetic ones live longer, up to 2.5× this base value.

### MotionRef(mm)  *(5–200, default 110)*
The frame-to-frame millimeter change in a zone that counts as "full" motion (i.e., maps to a
motion ratio of 1.0). This is a **divisor**: a zone's raw motion ratio is
`(zone's diff from background) / MotionRef × MoveFX%`. Set it low and even small changes read
as intensely energetic; set it high and only large, obvious motion registers strongly. At the
extreme high end, nearly every zone's ratio gets crushed toward zero — spawn probability has a
small floor (tied to how confidently foreground each zone currently is) specifically so the
scene doesn't collapse to a sparse, flickery handful of particles when MotionRef is maxed out.

### MoveFX (drama)%  *(10–300, default 100)*
A gain multiplier applied on top of MotionRef's ratio — the "how dramatic does motion feel"
dial. Affects spawn probability and each particle's initial size/life/wobble scale uniformly.
Turning it down makes everything calmer across the board; turning it up exaggerates motion
response. The motion signal that drives this has a short temporal smoothing pass (~0.25 EMA)
applied before MoveFX sees it, specifically so that when a moving object comes to rest, newly
spawned particles ease down toward calm instead of every particle's size/wobble snapping
instantly to the floor the moment frame-to-frame diff drops — which is what used to look like a
jarring "Z normalizing" glitch, especially at low MoveFX values where the whole range was
already compressed near that floor.

### Max Z(mm)  *(300–8800, default 8800)*
Hard distance cutoff — any zone farther than this from the sensor is ignored entirely (no
spawn, no render). Use it to crop out a background wall/room bounds you don't want
contributing particles at all. (Named "Max Z" — it's a straight-line sensor distance cutoff,
not a background-subtraction control; see BgThresh for that.)

### BgThresh(mm)  *(10–500, default 60)*
Minimum difference from the background reference (see below) for a zone to count as
foreground. Below this threshold, a zone is treated as part of the static scene and
contributes essentially nothing. The background reference itself is always the slow,
always-on adaptive estimate (~0.3s time constant) — there used to be a manual "Capture BG"
button for a one-shot snapshot reference, but it was removed because a stale snapshot
consistently made results *worse* than just trusting the adaptive estimate (a fixed snapshot
drifts out of date and starts misreading real background as foreground noise; the adaptive
version never goes stale). The foreground/background comparison near this threshold uses a
smooth quadratic ramp rather than a hard on/off gate, specifically so ordinary sensor noise
hovering right at the threshold doesn't cause scattered spurious particles flickering in and
out across an otherwise-static background.

### JumpThresh(mm)  *(20–400, default 400)*
Part of the flying-pixel/mixed-pixel edge-flicker suppression system. A zone sitting on a
sharp depth discontinuity (an object's silhouette edge) can flicker frame-to-frame between
reading the near surface and the far surface, purely from sensor noise — not real motion. Any
frame-to-frame jump bigger than this threshold gets flagged and clamped (see MinVel), and the
particle's position snaps to the *nearer* of the two readings (real edges are almost always the
close surface bleeding through, not the far one). This is now supplemented by a **spatial**
check — a zone whose delta disagrees sharply with its own 3×3-neighborhood median is flagged
as an outlier too, even if its raw magnitude alone wouldn't cross this threshold — so lowering
this slider is no longer the only way to catch flicker; EdgeSmooth (below) handles the spatial
side.

### MinVel(mm)  *(0–50, default 5)*
Does two things: (1) it's the velocity a jump-flagged zone (see JumpThresh) gets clamped down
to, instead of its raw (spurious) delta; (2) it's *also* an always-on deadband — any
frame-to-frame delta smaller than this is treated as pure sensor noise and zeroed outright,
regardless of whether JumpThresh ever fires. That second role is what makes this slider have a
consistently visible effect even when JumpThresh is set high enough that its clamp branch
rarely triggers.

### EdgeSmooth%  *(0–100, default 50)*
Spatial denoising strength. This does two separate things, both aimed at the same root cause
(isolated single-zone flicker at sharp depth edges):
1. **Feeds the background/foreground classification.** A flickering edge zone's smoothed depth
   value never fully settles, so without this it can read as permanent "foreground" forever and
   spawn particles continuously. EdgeSmooth blends the signal feeding the background-adaptation
   pipeline toward a 3×3 neighbor-median, letting a flickering-but-actually-static edge zone
   settle into the background like a real static zone should.
2. **Widens jump detection** (see JumpThresh) with the spatial-outlier check described above.

Both effects only ever *add* suppression on top of the raw per-pixel magnitude check — they
can't let a genuinely large jump slip through unclamped. (An earlier version blended the
*position* signal itself before the magnitude check, which could actually pull a real edge's
delta *below* JumpThresh and let it through unclamped — backwards from the intent. That's been
fixed; EdgeSmooth is now purely additive suppression.) The actual rendered particle position
always uses the raw, un-blurred depth — only the *decision* pipeline sees the smoothed version,
so the point cloud's shape itself never gets visually blurred.

### NoiseAmp  *(0–3, default 0.8)*
Independent strength control for the ambient flow-field wobble (separate from Speed, which
mostly drives the ballistic/inherited-velocity term). The flow field itself is **true curl
noise** — the curl of a 3-component Perlin vector potential, computed via finite differences —
not just three independent noise samples. Curl of any potential field is mathematically
guaranteed divergence-free, which is the actual mechanism that makes real fluids swirl and roll
coherently instead of each point jittering independently ("hula dancing"); it's the standard
real-time stand-in for actually solving Navier-Stokes, used throughout VFX for smoke/fluid-like
particle motion. The field is sampled at each particle's *actual, flow-carried* position (not
its fixed birth position) using the shared global clock, so nearby particles moving together
sample nearly the same curl vector and read as one coherent stream, not independent noise.
Wobble amplitude scales purely with each particle's own motion (no artificial floor), so
static/near-static particles stay calm — only genuinely moving particles get pushed around by
the field.

### DepthGlow%  *(0–200, default 80)*
Depth-cueing strength: particles nearer the camera side of the current view get brighter and
more saturated (in HSV space); farther ones get dimmer and more muted. This is a cheap
per-vertex effect, not true ambient occlusion (a real AO pass would need a second render pass
with depth-buffer sampling — out of scope for a single-pass vertex-shader pipeline on this
hardware) but gives a similar sense-of-depth payoff for near-zero cost.

### SpawnHue / SpawnSat / SpawnVal  *(0–360° / 0–1 / 0–1)*
### DeathHue / DeathSat / DeathVal  *(0–360° / 0–1 / 0–1)*
The two color endpoints used by the Velocity and Lifetime color modes (see the **Color**
button below) — "spawn" and "death" name the two ends of whatever gradient is being
interpolated (age, in Lifetime mode; speed, in Velocity mode), not literally "when the
particle is born" vs "when it dies" in every mode. The **Palette** button (below) sets these
six sliders to a preset in one click; they stay freely adjustable afterward, same as any other
slider — palettes are just a shortcut, not a separate locked-in mode.

Independent of color mode, saturation always fades to zero over each particle's life (full
color at spawn → fully desaturated by end of life), layered on top of whatever the color mode
computes.

### Twinkle  *(0–1, default 0)*
A per-particle random-phase brightness flicker/sparkle. Each particle gets its own randomly
hashed flicker frequency and phase (derived from its birth position and birth time), so
particles twinkle independently — no shared clock, no synchronized pulsing. **At exactly 0 this
is a literal no-op**: the shader does `mix(1.0, twinkle_wave, u_twinkle)`, which evaluates to
precisely `1.0` (no brightness change at all) when `u_twinkle == 0.0`, not just "a small effect."

---

## Buttons

### Color: Depth / Velocity / Life
Cycles through three ways a particle's base color is computed:
- **Depth** — the original behavior: a JET colormap based on the zone's distance from the
  sensor at spawn time, baked in on the CPU.
- **Velocity** — hue is swept between SpawnHue and DeathHue based on the particle's speed
  magnitude at spawn (capped at a reference speed), so fast and slow particles read as visibly
  different colors.
- **Life** — hue/saturation/value are interpolated between the Spawn* and Death* sliders over
  the particle's own age, so every particle visibly shifts color as it ages, independent of
  motion or depth.

All three modes still get the DepthGlow brightness/saturation boost and the twinkle effect
layered on top afterward.

### Palette
Cycles through five preset (SpawnHSV, DeathHSV) pairs and applies them to the six color
sliders in one click: **Ocean** (cyan → deep blue), **Fire** (hot yellow-orange → dark red),
**Rainbow** (red → purple), **Ice** (pale blue-white → muted blue), **Mono** (white → dark
gray). This only matters visually in Velocity or Life color mode — Depth mode ignores these
sliders entirely.

### Field: OFF / ON
Toggles a debug overlay: a coarse 6×5×10 grid of short lines (300 sample points) showing the
*actual* flow field steering the particles — both the ambient curl-noise layer (using the
live Freq slider) and the real, current optical-flow velocity at each grid point (temporally
smoothed with a short EMA so it doesn't jitter frame-to-frame from raw Farneback noise the way
individual particles never do — particles only ever sample flow once at spawn and freeze it,
so they're naturally immune to this; the debug view recomputes every frame, so it needed
explicit smoothing to match). Blue→yellow gradient marks each line's base→tip.

**First use in a session takes noticeably longer to appear** (the shader/attribute
combination needs a one-time GPU pipeline warm-up on this hardware — up to a minute or so,
though this is now pre-warmed automatically at app startup so the *live* toggle itself should
be instant). If you ever see the whole app appear to freeze right after toggling something new
for the first time, it's very likely this same phenomenon, not a crash — worth waiting a couple
of minutes before assuming otherwise.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `R` | Cycle sensor image orientation (flip/rotate) |
| `F` | Reset camera (azimuth/tilt/zoom/pan) to defaults |

Mouse: left-drag orbits the camera, scroll wheel zooms. Sliders and buttons live in the
top-left panel — click-drag on a slider's track to set its value.

---

## Rendering details (not sliders, but worth knowing)

- **Additive blending** is used for particles (`ONE, ONE`), which is why overlapping particles
  glow brighter rather than obscuring each other — this is also why the sharp-edge fragment
  shader change mattered: with additive blending, soft/blurry edges compound into a haze very
  quickly as particle count grows, whereas crisp edges stay readable.
- **Size always reaches exactly 0 at end of life** (`gl_PointSize` is multiplied by the
  particle's life fraction, which is 1.0 at spawn and 0.0 at death), so particles fade out by
  shrinking to nothing, not by an alpha cut alone.
- **Per-pixel radial shading** is layered independently of the edge-sharpness fix: each
  particle's fragment gets a bright-center-to-dim-rim brightness falloff in HSV space, so it
  reads as a shaded sphere/glossy dot rather than a flat cutout, without reintroducing the soft
  edges that caused particles to blur together.
- All Z-position math uses a **fixed absolute depth reference** (not the current frame's mean
  depth) — this was a deliberate fix for an early bug where the whole background would visually
  shift in Z whenever something moved in the foreground, because Z was being computed relative
  to a value that changed with scene content. It never does that now, by design.
