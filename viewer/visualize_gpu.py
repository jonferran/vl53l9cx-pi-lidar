#!/usr/bin/env python3
"""
GPU shader particle viewer for the VL53L9CX ToF LiDAR -- a separate, standalone
file from viewer/visualize.py so the CPU version stays untouched and this can
be dropped/reverted independently.

The whole particle simulation (Perlin-noise flow drift, per-particle fade,
motion-scaled size/speed, camera projection) runs in a GLSL vertex shader on
the Pi4's GPU (Broadcom V3D). The CPU side only does two things per frame:
decode the sensor's RAW8 CSI stream into (position, color, motion) records for
newly-visible zones, and upload that small "new particles" batch into a
ring-buffer VBO -- it never touches existing particles again. Their entire
trajectory (drift, fade, culling once ~1s old) is a pure function of
(birth_pos, birth_time, motion) evaluated fresh every frame on the GPU, which
is why this scales to far more particles than the CPU version could.

Classic Perlin noise GLSL from Ashima Arts / Stefan Gustavson's public-domain
"webgl-noise" (https://github.com/ashima/webgl-noise), the same reference
family as typical ShaderToy flow-field demos.

Keys: r = rotate/flip, q = quit.  Mouse: left-drag orbits, scroll zooms.
"""
import numpy as np
import cv2
import glfw
import moderngl
import time
import os
import array
import fcntl
import termios
import struct

BINNING = int(os.environ.get("VL_BINNING", "2"))
GEOM = {2: (54, 42, 148), 4: (24, 24, 38)}
RAW_W, RAW_H, CSI_H = GEOM[BINNING]
N_ZONES = RAW_W * RAW_H
CH_LEN = N_ZONES * 2
CSI_W = 100
STRIDE = 128
FRAME_BYTES = STRIDE * CSI_H
DIST_MASK = 0x7FFF
PIPE = os.environ.get("VL_PIPE", "tof_pipe")

MAX_MM = 4000
CLOUD_SCALE = 1.0
CURVE_K = 300.0
ZDEPTH = 1.0
Z_REF_MM = 1500.0   # fixed reference distance for the log-depth Z coordinate --
                    # a CONSTANT, not derived from the current frame, so a zone's
                    # Z position depends only on its own absolute distance and
                    # never shifts because something else in the scene moved.
Z_REF_LOG = np.log1p(Z_REF_MM / CURVE_K)
MOTION_REF_MM = 40.0
MOTION_MIN_SCALE = 0.08
MOTION_MAX_SCALE = 3.0
PARTICLE_LIFE = 1.2
PARTICLE_MAX = 200_000     # GPU can hold vastly more live particles than the CPU version
BASE_SIZE = 5.0
DRIFT_SPEED = 14.0
NOISE_FREQ = 0.06
FLOW_GAIN = 4.0     # extra multiplier on optical-flow x/y so real motion reads clearly

ORIENT = 3   # matches the CPU 3D-mode default (see visualize.py's eff_orient note)
def orient(a, m):
    if m == 1: return a[::-1, :]
    if m == 2: return a[:, ::-1]
    if m == 3: return a[::-1, ::-1]
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

VERTEX_SHADER = """
#version 140
in vec3 a_birth_pos;
in float a_born_time;
in float a_motion;
in vec3 a_color;
in vec3 a_velocity;   // inherited from optical flow (+ signed depth delta) at spawn

uniform float u_time;
uniform float u_life;
uniform float u_azimuth;
uniform float u_tilt;
uniform float u_zoom;
uniform vec2 u_pan;
uniform vec2 u_canvas;
uniform float u_base_size;
uniform float u_speed;
uniform float u_freq;
uniform float u_noise_amp;
uniform float u_depth_glow;
uniform float u_energy;
uniform float u_color_mode;   // 0=Depth (a_color from CPU), 1=Velocity, 2=Lifetime
uniform vec3 u_spawn_hsv;
uniform vec3 u_death_hsv;
uniform float u_twinkle;

out vec3 v_color;
out float v_life_frac;
out float v_motion_alpha;

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// --- classic 3D Perlin noise (Ashima Arts / Stefan Gustavson, public domain) --
vec3 mod289v3(vec3 x) { return x - floor(x / 289.0) * 289.0; }
vec4 mod289v4(vec4 x) { return x - floor(x / 289.0) * 289.0; }
vec4 permute(vec4 x) { return mod289v4(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec3 fade3(vec3 t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float pnoise3(vec3 P) {
    vec3 Pi0 = mod289v3(floor(P));
    vec3 Pi1 = mod289v3(Pi0 + vec3(1.0));
    vec3 Pf0 = fract(P);
    vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;

    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);

    vec4 gx0 = ixy0 / 7.0;
    vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);

    vec4 gx1 = ixy1 / 7.0;
    vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);

    vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
    vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
    vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
    vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
    vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
    vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
    vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
    vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);

    vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x; g010 *= norm0.y; g100 *= norm0.z; g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x; g011 *= norm1.y; g101 *= norm1.z; g111 *= norm1.w;

    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);

    vec3 fxyz = fade3(Pf0);
    vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fxyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fxyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fxyz.x);
    return 2.2 * n_xyz;
}

// Curl of a 3-component vector potential (each component its own offset
// Perlin field), via forward-difference partials. Curl of ANY potential
// field is guaranteed divergence-free, which is exactly what makes real
// fluids (Navier-Stokes) swirl and roll instead of just jittering
// randomly -- this is the standard cheap real-time stand-in for actually
// solving NS (no grid/pressure-projection pass needed, just derivatives of
// noise), used throughout VFX for smoke/fluid-like particle motion.
vec3 curlNoise(vec3 p) {
    const float e = 0.6;
    vec3 dx = vec3(e, 0.0, 0.0), dy = vec3(0.0, e, 0.0), dz = vec3(0.0, 0.0, e);
    vec3 offX = vec3(37.2, 17.1, 0.0);
    vec3 offY = vec3(91.7, 63.3, 0.0);

    float x0 = pnoise3(p + offX), x_dy = pnoise3(p + dy + offX), x_dz = pnoise3(p + dz + offX);
    float y0 = pnoise3(p + offY), y_dx = pnoise3(p + dx + offY), y_dz = pnoise3(p + dz + offY);
    float z0 = pnoise3(p),        z_dx = pnoise3(p + dx),        z_dy = pnoise3(p + dy);

    float dzdy = (z_dy - z0) / e, dydz = (y_dz - y0) / e;
    float dxdz = (x_dz - x0) / e, dzdx = (z_dx - z0) / e;
    float dydx = (y_dx - y0) / e, dxdy = (x_dy - x0) / e;

    return vec3(dzdy - dydz, dxdz - dzdx, dydx - dxdy);
}

void main() {
    // Everything below is a pure function of THIS particle's own a_motion --
    // no shared/global term of any kind, so motion anywhere in the scene can
    // never move, resize, brighten or extend the life of a particle that
    // belongs to an unrelated (static) zone. Speed, life and size all scale
    // up together with a_motion; the noise field's own displacement amplitude
    // is also directly proportional to it (dir * a_motion below), so it's
    // near-still for low motion and energetic for high motion.
    float eff_life = u_life * clamp(0.35 + a_motion * 0.7, 0.35, 2.5);
    float age = u_time - a_born_time;
    v_life_frac = 1.0 - clamp(age / eff_life, 0.0, 1.0);

    if (age < 0.0 || age > eff_life) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);  // clip -- never rasterized
        gl_PointSize = 0.0;
        v_color = a_color;
        v_motion_alpha = 0.0;
        return;
    }

    // Real ballistic motion: the particle inherits a_velocity (measured via
    // optical flow + signed depth delta at the moment it spawned) and that
    // velocity decays exponentially -- v(age) = v0*exp(-k*age), so speed is
    // unambiguously highest the instant it's born and drops monotonically
    // from there (unlike the previous displacement-envelope approach, whose
    // motion still visually read as accelerating). Integrating that decaying
    // velocity gives the position term below.
    const float DAMPING = 3.0;   // 1/seconds; higher = faster decel to a stop
    float damp_env = (1.0 - exp(-age * DAMPING)) / DAMPING;
    vec3 ballistic = a_velocity * u_speed * damp_env;

    // Curl-noise-style shared flow field, not independent per-particle
    // jitter: sampled at the position the particle's REAL (optical-flow)
    // velocity has already carried it to, using the GLOBAL clock (u_time)
    // rather than each particle's own age. That means nearby particles --
    // especially ones moving together as part of the same fast-moving
    // object -- sample nearly the same noise vector at any given instant and
    // get deflected together, reading as one coherent stream instead of each
    // one independently wiggling in place ("hula dancing"). Scaling purely
    // by a_motion (no floor) keeps static/near-static particles calm.
    vec3 flow_pos = a_birth_pos + ballistic;
    vec3 seed = flow_pos * u_freq + vec3(0.0, 0.0, u_time * 0.15);
    vec3 dir = curlNoise(seed);
    // Speed and the system's overall "energy" both pump the FIELD's
    // amplitude now, not just each particle's own straight-line ballistic
    // term -- so turning Speed up makes the whole flow field feel bigger and
    // more turbulent, not just individual particles darting in straight
    // lines faster. u_energy is a smoothed (spawn count * avg speed)
    // estimate computed CPU-side each frame -- a cheap proxy for how much is
    // happening in the whole scene right now, without any GPU readback.
    float energy_boost = 1.0 + clamp(u_energy, 0.0, 5.0) * 0.25;
    vec3 wobble = dir * u_noise_amp * (0.4 + u_speed * 0.08) * energy_boost * a_motion * damp_env;

    vec3 pos = a_birth_pos + ballistic + wobble;

    float ca = cos(u_azimuth), sa = sin(u_azimuth);
    float ct = cos(u_tilt), st = sin(u_tilt);
    float xr = pos.x * ca + pos.z * sa;
    float zr = -pos.x * sa + pos.z * ca;
    float yr = pos.y * ct - zr * st;

    float scale = (u_canvas.y / 40.0) * u_zoom;
    vec2 screen = vec2(u_canvas.x / 2.0 + u_pan.x + xr * scale,
                        u_canvas.y / 2.0 + u_pan.y + yr * scale);
    vec2 clip = (screen / u_canvas) * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
    // Size always largest right at spawn (v_life_frac==1) and shrinks to
    // exactly 0 by end of life (v_life_frac==0), on top of the existing
    // motion-based scale.
    gl_PointSize = clamp(u_base_size * (0.3 + a_motion * 1.2) * v_life_frac, 0.0, 40.0);

    // Cheap depth cueing (not true AO): particles nearer the camera side of
    // the rotated view get brighter and more saturated, farther ones dimmer
    // and more muted -- gives a sense of depth/occlusion without a second
    // rendering pass. zr's sign convention: at azimuth=0 it's ~= our log-depth
    // z, where smaller/negative = nearer the sensor.
    float near_amt = 1.0 - clamp((zr + 40.0) / 80.0, 0.0, 1.0);

    vec3 base_hsv;
    if (u_color_mode < 0.5) {
        // Depth: JET colormap baked in CPU-side at spawn (a_color).
        base_hsv = rgb2hsv(a_color);
        base_hsv.y *= v_life_frac;   // fade to gray over life, as before
    } else if (u_color_mode < 1.5) {
        // Velocity: hue swept from spawn to death hue by speed magnitude,
        // so fast particles read as one color, slow as another.
        float speed_t = clamp(length(a_velocity) / 3.0, 0.0, 1.0);
        base_hsv = vec3(mix(u_spawn_hsv.x, u_death_hsv.x, speed_t),
                         mix(u_spawn_hsv.y, u_death_hsv.y, speed_t),
                         mix(u_spawn_hsv.z, u_death_hsv.z, speed_t));
    } else {
        // Lifetime: interpolate spawn color -> death color over the
        // particle's own age (v_life_frac 1 at birth -> 0 at death).
        float age_t = 1.0 - v_life_frac;
        base_hsv = vec3(mix(u_spawn_hsv.x, u_death_hsv.x, age_t),
                         mix(u_spawn_hsv.y, u_death_hsv.y, age_t),
                         mix(u_spawn_hsv.z, u_death_hsv.z, age_t));
    }
    base_hsv.y = clamp(base_hsv.y * (1.0 + u_depth_glow * near_amt), 0.0, 1.0);
    base_hsv.z = clamp(base_hsv.z * (1.0 + u_depth_glow * near_amt * 0.8), 0.0, 3.0);

    // Twinkle: each particle gets its own random flicker rate and phase
    // (hashed from its birth attributes, so it's stable for that particle's
    // whole life and different from every other particle -- no shared clock,
    // no synchronized pulsing). At u_twinkle == 0 this is an exact multiply
    // by 1.0, i.e. provably zero effect, not just "small".
    float tw_seed = dot(a_birth_pos, vec3(12.9898, 78.233, 37.719)) + a_born_time * 91.7;
    float tw_freq = 6.0 + fract(sin(tw_seed) * 43758.5453) * 14.0;
    float tw_phase = fract(sin(tw_seed * 1.37) * 24634.6345) * 6.2831853;
    float tw_wave = 0.5 + 0.5 * sin(u_time * tw_freq + tw_phase);
    base_hsv.z *= mix(1.0, tw_wave * 1.6, u_twinkle);

    v_color = hsv2rgb(base_hsv);

    // Static particles fade toward invisible, not just small; moving ones
    // ramp up to full brightness quickly.
    v_motion_alpha = clamp(smoothstep(0.0, 1.0, a_motion), 0.03, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 140
in vec3 v_color;
in float v_life_frac;
in float v_motion_alpha;
out vec4 f_color;

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // Two SEPARATE concerns, kept separate on purpose: edge sharpness (alpha
    // coverage) vs. shading (per-pixel brightness). A thin, ~constant-width
    // antialiased edge (via fwidth) keeps the disk's OUTLINE crisp at any
    // size -- that's what stops bigger points from turning into fuzzy blobs
    // that melt together under additive blending. Shading is layered on top
    // independently: a radial brightness falloff in HSV space (bright core,
    // dimmer rim) so each particle still reads as a shaded sphere/glossy
    // dot instead of a flat cutout, without reintroducing the soft edge.
    float d = length(gl_PointCoord - vec2(0.5));
    float edge = fwidth(d) * 1.5;
    float alpha = 1.0 - smoothstep(0.5 - edge, 0.5, d);
    if (alpha <= 0.0) discard;

    vec3 hsv = rgb2hsv(v_color);
    float shade = 1.0 - clamp(d / 0.5, 0.0, 1.0);   // 1 at center, 0 at rim
    hsv.z *= mix(0.35, 1.0, shade);
    vec3 shaded = hsv2rgb(hsv);

    float a = alpha * v_life_frac * v_motion_alpha;
    f_color = vec4(shaded * a, a);
}
"""

# --- Debug: draw the actual curl-noise flow field as a coarse grid of little
# lines, one per sample point, direction = the SAME curlNoise() function the
# particles themselves use (duplicated here since GLSL has no #include) so
# what you see is exactly what's steering the particles, not an approximation.
FIELD_VERTEX_SHADER = """
#version 140
in vec3 a_pos;
in float a_t;         // 0.0 = line base, 1.0 = line tip
in vec3 a_flow_vel;    // this grid point's live optical-flow velocity, written
                       // fresh into a plain VBO every frame from the CPU --
                       // NOT a texture. A float-texture sampling test on this
                       // GPU/driver silently returned zero instead of the
                       // real data (and the full app hung using one), so this
                       // sticks to the VBO-write path already proven safe by
                       // the particle system's own per-frame updates.

uniform float u_time;
uniform float u_freq;
uniform float u_arrow_len;
uniform float u_azimuth;
uniform float u_tilt;
uniform float u_zoom;
uniform vec2 u_pan;
uniform vec2 u_canvas;
uniform float u_flow_vis_scale;

out float v_t;

vec3 mod289v3(vec3 x) { return x - floor(x / 289.0) * 289.0; }
vec4 mod289v4(vec4 x) { return x - floor(x / 289.0) * 289.0; }
vec4 permute(vec4 x) { return mod289v4(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec3 fade3(vec3 t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float pnoise3(vec3 P) {
    vec3 Pi0 = mod289v3(floor(P));
    vec3 Pi1 = mod289v3(Pi0 + vec3(1.0));
    vec3 Pf0 = fract(P);
    vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;

    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);

    vec4 gx0 = ixy0 / 7.0;
    vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);

    vec4 gx1 = ixy1 / 7.0;
    vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);

    vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
    vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
    vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
    vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
    vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
    vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
    vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
    vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);

    vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x; g010 *= norm0.y; g100 *= norm0.z; g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x; g011 *= norm1.y; g101 *= norm1.z; g111 *= norm1.w;

    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);

    vec3 fxyz = fade3(Pf0);
    vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fxyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fxyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fxyz.x);
    return 2.2 * n_xyz;
}

vec3 curlNoise(vec3 p) {
    const float e = 0.6;
    vec3 dx = vec3(e, 0.0, 0.0), dy = vec3(0.0, e, 0.0), dz = vec3(0.0, 0.0, e);
    vec3 offX = vec3(37.2, 17.1, 0.0);
    vec3 offY = vec3(91.7, 63.3, 0.0);

    float x0 = pnoise3(p + offX), x_dy = pnoise3(p + dy + offX), x_dz = pnoise3(p + dz + offX);
    float y0 = pnoise3(p + offY), y_dx = pnoise3(p + dx + offY), y_dz = pnoise3(p + dz + offY);
    float z0 = pnoise3(p),        z_dx = pnoise3(p + dx),        z_dy = pnoise3(p + dy);

    float dzdy = (z_dy - z0) / e, dydz = (y_dz - y0) / e;
    float dxdz = (x_dz - x0) / e, dzdx = (z_dx - z0) / e;
    float dydx = (y_dx - y0) / e, dxdy = (x_dy - x0) / e;

    return vec3(dzdy - dydz, dxdz - dzdx, dydx - dxdy);
}

void main() {
    vec3 seed = a_pos * u_freq + vec3(0.0, 0.0, u_time * 0.15);
    vec3 dir = curlNoise(seed) + a_flow_vel * u_flow_vis_scale;
    float len = length(dir);
    if (len > 0.0001) dir /= len;   // arrow direction only; length is fixed so
                                     // it reads as "which way", not magnitude
    vec3 pos = a_pos + dir * u_arrow_len * a_t;

    float ca = cos(u_azimuth), sa = sin(u_azimuth);
    float ct = cos(u_tilt), st = sin(u_tilt);
    float xr = pos.x * ca + pos.z * sa;
    float zr = -pos.x * sa + pos.z * ca;
    float yr = pos.y * ct - zr * st;

    float scale = (u_canvas.y / 40.0) * u_zoom;
    vec2 screen = vec2(u_canvas.x / 2.0 + u_pan.x + xr * scale,
                        u_canvas.y / 2.0 + u_pan.y + yr * scale);
    vec2 clip = (screen / u_canvas) * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
    v_t = a_t;
}
"""
FIELD_FRAGMENT_SHADER = """
#version 140
in float v_t;
out vec4 f_color;
void main() {
    vec3 col = mix(vec3(0.1, 0.55, 1.0), vec3(1.0, 0.95, 0.2), v_t);
    f_color = vec4(col, 1.0);
}
"""

# --- Minimal flat-colour quad shader for the on-screen particle-count slider -
UI_VERTEX_SHADER = """
#version 140
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""
UI_FRAGMENT_SHADER = """
#version 140
uniform vec3 u_color;
out vec4 f_color;
void main() { f_color = vec4(u_color, 1.0); }
"""

# Slider labels: cv2.putText bakes each name into a small alpha texture once
# at startup (no font-rendering pipeline otherwise exists in raw GL here).
TEXT_VERTEX_SHADER = """
#version 140
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_uv; }
"""
TEXT_FRAGMENT_SHADER = """
#version 140
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() { f_color = texture(u_tex, v_uv); }
"""

# --- particle ring buffer (CPU-side staging; GPU owns the live simulation) --
_cursor = 0
STRUCT_FMT = "<3f f f 3f 3f"   # birth_pos(3f) born_time(f) motion(f) color(3f) velocity(3f)
RECORD_FLOATS = 11

def main():
    if not glfw.init():
        raise SystemExit("glfw.init() failed")
    win = glfw.create_window(1024, 768, "VL53L9CX GPU Particles", None, None)
    if not win:
        glfw.terminate()
        raise SystemExit("Failed to create GL window (no display / GL context?)")
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    ctx = moderngl.create_context(require=310)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.ONE, moderngl.ONE   # additive glow

    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    vbo = ctx.buffer(reserve=PARTICLE_MAX * RECORD_FLOATS * 4, dynamic=True)
    vao = ctx.vertex_array(prog, [(vbo, "3f 1f 1f 3f 3f",
                                   "a_birth_pos", "a_born_time", "a_motion", "a_color", "a_velocity")])

    # Pre-zero the buffer so unwritten slots have born_time far in the past (dead).
    zero_born = np.zeros(PARTICLE_MAX * RECORD_FLOATS, dtype="f4")
    zero_born[3::RECORD_FLOATS] = -1e6
    vbo.write(zero_born.tobytes())

    # Debug flow-field grid: static, coarse (6x5x10=300 points), downscaled
    # well below actual particle density -- built once, never changes (only
    # u_time/u_freq animate the arrows via the shader, same as particles).
    field_prog = ctx.program(vertex_shader=FIELD_VERTEX_SHADER, fragment_shader=FIELD_FRAGMENT_SHADER)
    gx = np.linspace(-24.0, 24.0, 6, dtype="f4")
    gy = np.linspace(-18.0, 18.0, 5, dtype="f4")
    gz = np.linspace(-40.0, 40.0, 10, dtype="f4")
    gxx, gyy, gzz = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_pts = np.stack([gxx.ravel(), gyy.ravel(), gzz.ravel()], axis=1).astype("f4")
    n_field = grid_pts.shape[0]
    # Single INTERLEAVED buffer (pos, t, flow_vel all in one VBO), not two
    # separate buffer objects on one VAO -- a two-buffer VertexArray hung
    # render() outright on this GPU/driver even with trivial content
    # (isolated test confirmed it), while a single interleaved buffer with
    # the identical attributes renders fine. This mirrors the exact pattern
    # the particle system's own VBO already uses successfully every frame.
    field_base = np.empty((n_field * 2, 4), dtype="f4")
    field_base[0::2, 0:3] = grid_pts
    field_base[1::2, 0:3] = grid_pts
    field_base[0::2, 3] = 0.0
    field_base[1::2, 3] = 1.0
    field_data = np.zeros((n_field * 2, 7), dtype="f4")
    field_data[:, 0:4] = field_base
    field_vbo = ctx.buffer(field_data.tobytes(), dynamic=True)
    field_vao = ctx.vertex_array(field_prog, [(field_vbo, "3f 1f 3f", "a_pos", "a_t", "a_flow_vel")])

    # grid_row_idx/grid_col_idx map each grid point to its nearest sensor
    # zone, used every frame to look up that point's live optical-flow
    # velocity before rewriting the whole interleaved buffer.
    grid_col_idx = np.clip(np.round(grid_pts[:, 0] + RAW_W / 2.0).astype(np.intp), 0, RAW_W - 1)
    grid_row_idx = np.clip(np.round(RAW_H / 2.0 - grid_pts[:, 1]).astype(np.intp), 0, RAW_H - 1)

    # Pre-warm this shader's GPU pipeline right now, during the startup delay
    # a user already expects, instead of the first time they click "Field:
    # ON" mid-session. Confirmed via isolated testing: the FIRST render() of
    # a brand-new shader/attribute combination on this GPU/driver can stall
    # for 60-90+ seconds before ever completing (later renders are instant)
    # -- indistinguishable from a genuine freeze if it happens on a live
    # click instead of here.
    print("[gpu] warming up debug field shader (one-time)...", flush=True)
    ctx.viewport = (0, 0, 1024, 768)
    field_prog["u_time"].value = 0.0
    field_prog["u_freq"].value = 0.06
    field_prog["u_arrow_len"].value = 3.0
    field_prog["u_azimuth"].value = 0.0
    field_prog["u_tilt"].value = 0.0
    field_prog["u_zoom"].value = 1.0
    field_prog["u_pan"].value = (0.0, 0.0)
    field_prog["u_canvas"].value = (1024.0, 768.0)
    field_prog["u_flow_vis_scale"].value = 1.0
    field_vao.render(moderngl.LINES, vertices=n_field * 2)
    print("[gpu] field shader warm.", flush=True)

    ui_prog = ctx.program(vertex_shader=UI_VERTEX_SHADER, fragment_shader=UI_FRAGMENT_SHADER)
    ui_vbo = ctx.buffer(reserve=4 * 2 * 4, dynamic=True)
    ui_vao = ctx.vertex_array(ui_prog, [(ui_vbo, "2f", "in_pos")])

    def draw_quad(x0, y0, x1, y1, fbw, fbh, color):
        # pixel rect -> NDC (Y flipped: pixel origin top-left, NDC origin center/up)
        def ndc(px, py):
            return (px / fbw * 2.0 - 1.0, 1.0 - py / fbh * 2.0)
        p0, p1 = ndc(x0, y0), ndc(x1, y1)
        verts = np.array([p0[0], p0[1], p1[0], p0[1], p0[0], p1[1], p1[0], p1[1]], dtype="f4")
        ui_vbo.write(verts.tobytes())
        ui_prog["u_color"].value = color
        ui_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    text_prog = ctx.program(vertex_shader=TEXT_VERTEX_SHADER, fragment_shader=TEXT_FRAGMENT_SHADER)
    text_vbo = ctx.buffer(reserve=4 * 4 * 4, dynamic=True)
    text_vao = ctx.vertex_array(text_prog, [(text_vbo, "2f 2f", "in_pos", "in_uv")])

    def make_text_texture(text, color=(255, 255, 255)):
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        pad = 2
        w, h = tw + pad * 2, th + baseline + pad * 2
        alpha = np.zeros((h, w), dtype=np.uint8)
        cv2.putText(alpha, text, (pad, h - pad - baseline), font, scale, 255, thick, cv2.LINE_AA)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
        rgba[..., 3] = alpha
        tex = ctx.texture((w, h), 4, rgba[::-1].tobytes())  # flip: GL texture V=0 is the bottom row
        tex.filter = moderngl.LINEAR, moderngl.LINEAR
        return tex, w, h

    def draw_text(tex, tw, th, x0, y0, fbw, fbh):
        def ndc(px, py):
            return (px / fbw * 2.0 - 1.0, 1.0 - py / fbh * 2.0)
        p0, p1 = ndc(x0, y0), ndc(x0 + tw, y0 + th)
        verts = np.array([
            p0[0], p0[1], 0.0, 1.0,
            p1[0], p0[1], 1.0, 1.0,
            p0[0], p1[1], 0.0, 0.0,
            p1[0], p1[1], 1.0, 0.0,
        ], dtype="f4")
        text_vbo.write(verts.tobytes())
        tex.use(location=0)
        text_prog["u_tex"].value = 0
        text_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    # --- Slider panel (top-left): name, min, max, default -----------------
    SLIDER_DEFS = [
        ("Density%", 2.0, 100.0, 100.0),   # overall spawn probability multiplier
        ("Size", 1.0, 20.0, 5.0),          # base point size
        ("Speed", 0.1, 20.0, 2.0),         # total displacement scale over a particle's life
        ("Freq", 0.01, 0.155, 0.06),        # spatial scale of the noise field (max halved: old 100% == new 50%)
        ("Life(s)", 0.3, 3.0, 1.2),        # particle lifetime / fade duration
        ("MotionRef(mm)", 5.0, 200.0, 110.0),   # frame-to-frame mm change treated as "full" motion
        ("MoveFX (drama)%", 10.0, 300.0, 100.0),   # gain on how much that motion affects everything downstream
        ("Max Z(mm)", 300.0, 8800.0, 8800.0),   # crop anything farther (background cutoff)
        ("BgThresh(mm)", 10.0, 500.0, 60.0),   # min diff from captured background to count as foreground
        ("JumpThresh(mm)", 20.0, 400.0, 400.0),  # frame-to-frame jump above this = edge-flicker noise
        ("MinVel(mm)", 0.0, 50.0, 5.0),         # residual velocity clamp applied at flicker zones
        ("EdgeSmooth%", 0.0, 100.0, 50.0),      # spatial (neighbor) denoise applied only to the
                                                 # jump/delta decision -- catches isolated flying-pixel
                                                 # zones a per-pixel time threshold alone can't
        ("NoiseAmp", 0.0, 3.0, 0.8),            # ambient Perlin wobble strength, independent of Speed
        ("DepthGlow%", 0.0, 200.0, 80.0),       # brighter+more saturated the closer to the sensor
        # Spawn/death color pickers (HSV) -- used directly in Velocity/Lifetime
        # color modes, and as the tint target for Depth mode's saturation/
        # brightness falloff. The "Palette" button below sets these six to a
        # preset; they stay freely adjustable afterward like any slider.
        ("SpawnHue", 0.0, 360.0, 200.0),
        ("SpawnSat", 0.0, 1.0, 0.9),
        ("SpawnVal", 0.0, 1.0, 1.0),
        ("DeathHue", 0.0, 360.0, 0.0),
        ("DeathSat", 0.0, 1.0, 0.0),
        ("DeathVal", 0.0, 1.0, 0.4),
        ("Twinkle", 0.0, 1.0, 0.0),   # per-particle random-phase brightness flicker;
                                       # 0 == exactly no effect (see vertex shader)
    ]
    slider_val = [d for (_, _, _, d) in SLIDER_DEFS]
    SLIDER_X, SLIDER_Y0, SLIDER_W, SLIDER_H, SLIDER_ROW = 20, 20, 200, 16, 26
    slider_labels = [make_text_texture(name) for (name, _, _, _) in SLIDER_DEFS]

    def slider_rect_px(i):
        y0 = SLIDER_Y0 + i * SLIDER_ROW
        return SLIDER_X, y0, SLIDER_X + SLIDER_W, y0 + SLIDER_H

    # Buttons row, below the sliders. name -> action key in `state`.
    BTN_Y = SLIDER_Y0 + len(SLIDER_DEFS) * SLIDER_ROW + 6
    BTN_W, BTN_H = 96, 22
    BUTTONS = [("Color: Depth", "cycle_mode"), ("Palette", "cycle_palette"), ("Field: OFF", "toggle_field")]
    COLOR_MODES = ["Depth", "Velocity", "Life"]
    # (name, spawn_hsv, death_hsv) -- H in 0..360, S/V in 0..1, matching the
    # SpawnHue/SpawnSat/SpawnVal/DeathHue/DeathSat/DeathVal slider ranges.
    PALETTES = [
        ("Ocean",   (190.0, 0.9, 1.0), (230.0, 0.6, 0.3)),
        ("Fire",    (45.0, 1.0, 1.0),  (0.0, 0.85, 0.3)),
        ("Rainbow", (0.0, 0.9, 1.0),   (270.0, 0.9, 1.0)),
        ("Ice",     (200.0, 0.3, 1.0), (220.0, 0.7, 0.5)),
        ("Mono",    (0.0, 0.0, 1.0),   (0.0, 0.0, 0.2)),
    ]
    button_labels = [make_text_texture(name) for (name, _) in BUTTONS]
    # Every label variant for the toggling buttons is baked ONCE here, up
    # front -- clicking Color/Field used to call make_text_texture() again on
    # every click, creating a brand-new GPU texture each time without
    # releasing the old one (moderngl doesn't reliably auto-release textures
    # via Python GC on this driver). A few clicks in a row leaked enough GPU
    # memory to make the driver start thrashing and hang the whole app. Now
    # a click just switches which already-baked texture gets drawn -- zero
    # runtime texture creation.
    color_mode_labels = [make_text_texture(f"Color: {m}") for m in COLOR_MODES]
    field_labels = [make_text_texture("Field: OFF"), make_text_texture("Field: ON")]

    def button_rect_px(i):
        x0 = SLIDER_X + i * (BTN_W + 8)
        return x0, BTN_Y, x0 + BTN_W, BTN_Y + BTN_H

    SPAWN_HSV_IDX, DEATH_HSV_IDX = 14, 17   # index of Hue within SLIDER_DEFS for each triplet

    def apply_palette(i):
        name, spawn, death = PALETTES[i % len(PALETTES)]
        slider_val[SPAWN_HSV_IDX:SPAWN_HSV_IDX + 3] = list(spawn)
        slider_val[DEATH_HSV_IDX:DEATH_HSV_IDX + 3] = list(death)
        print(f"[gpu] palette -> {name}", flush=True)

    def set_value_from_x(window, i, x):
        fbw, fbh = glfw.get_framebuffer_size(window)
        wx, _ = glfw.get_window_size(window)
        scale = fbw / max(wx, 1)  # framebuffer may be HiDPI-scaled vs window coords
        sx0, _, sx1, _ = slider_rect_px(i)
        frac = max(0.0, min(1.0, (x * scale - sx0) / (sx1 - sx0)))
        lo, hi = SLIDER_DEFS[i][1], SLIDER_DEFS[i][2]
        slider_val[i] = lo + frac * (hi - lo)

    state = {"az": 45.0, "tilt": 25.0, "zoom": 1.0, "pan": [0.0, 0.0],
             "dragging": False, "last": (0, 0), "orient": ORIENT, "slider_drag": -1,
             "energy": 0.0, "color_mode": 0, "palette": 0, "show_field": False}

    def on_mouse_button(window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.PRESS:
            fbw, fbh = glfw.get_framebuffer_size(window)
            wx, _ = glfw.get_window_size(window)
            scale = fbw / max(wx, 1)
            x, y = glfw.get_cursor_pos(window)
            px, py = x * scale, y * scale
            # Buttons take priority over the (larger) camera-drag area.
            for i, (_, act) in enumerate(BUTTONS):
                bx0, by0, bx1, by1 = button_rect_px(i)
                if bx0 <= px <= bx1 and by0 <= py <= by1:
                    if act == "cycle_mode":
                        state["color_mode"] = (state["color_mode"] + 1) % len(COLOR_MODES)
                    elif act == "cycle_palette":
                        state["palette"] = (state["palette"] + 1) % len(PALETTES)
                        apply_palette(state["palette"])
                    elif act == "toggle_field":
                        state["show_field"] = not state["show_field"]
                    return
            hit = -1
            for i in range(len(SLIDER_DEFS)):
                sx0, sy0, sx1, sy1 = slider_rect_px(i)
                if sx0 - 6 <= px <= sx1 + 6 and sy0 - 6 <= py <= sy1 + 6:
                    hit = i
                    break
            if hit >= 0:
                state["slider_drag"] = hit
                set_value_from_x(window, hit, x)
            else:
                state["dragging"] = True
                state["last"] = (x, y)
        else:
            state["dragging"] = False
            state["slider_drag"] = -1

    def on_cursor_pos(window, x, y):
        if state["slider_drag"] >= 0:
            set_value_from_x(window, state["slider_drag"], x)
        elif state["dragging"]:
            dx, dy = x - state["last"][0], y - state["last"][1]
            state["az"] = (state["az"] + dx * 0.3) % 360.0
            state["tilt"] = max(-89.0, min(89.0, state["tilt"] - dy * 0.3))
            state["last"] = (x, y)

    def on_scroll(window, dx, dy):
        state["zoom"] = max(0.1, min(20.0, state["zoom"] * (1.1 if dy > 0 else 1 / 1.1)))

    def on_key(window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            state["orient"] = (state["orient"] + 1) % 4
            print(f"[gpu] orient -> {state['orient']}", flush=True)
        elif key == glfw.KEY_F:
            state["az"], state["tilt"], state["zoom"] = 0.0, 0.0, 1.0
            state["pan"] = [0.0, 0.0]

    glfw.set_mouse_button_callback(win, on_mouse_button)
    glfw.set_cursor_pos_callback(win, on_cursor_pos)
    glfw.set_scroll_callback(win, on_scroll)
    glfw.set_key_callback(win, on_key)

    print(f"[gpu] {PIPE}  binning={BINNING}  {RAW_W}x{RAW_H} zones  frame={FRAME_BYTES}B", flush=True)
    start_time = time.time()
    prev_dist = None
    prev_gray = None     # previous frame's normalized depth, for optical flow
    dist_smooth = None   # EMA-smoothed depth, used for background-diff comparisons
                          # so single-frame sensor noise doesn't false-trigger as motion
    dist_bg_dynamic = None   # slow-adapting background estimate (always running, no
                              # button press needed) -- a zone that's been stable for a
                              # while gets absorbed into this and stops spawning; a zone
                              # that just changed stands out against it as foreground
    motion_smooth = None     # short EMA on the motion signal so "coming to rest" eases
                              # down instead of every particle's size/wobble snapping to
                              # the floor the instant frame-to-frame diff drops
    field_vel_smooth = None  # EMA on the debug field's per-grid-point flow velocity --
                              # particles only ever sample flow ONCE at spawn (frozen for
                              # their whole life), so they never show single-frame flow
                              # noise; the field arrows recompute every frame, which
                              # directly exposed raw Farneback jitter without this
    fps_n, fps_t0, fps = 0, time.time(), 0.0

    with open(PIPE, "rb", buffering=0) as fifo:
        nready = array.array("i", [0])
        fd = fifo.fileno()
        global _cursor

        while not glfw.window_should_close(win):
            glfw.poll_events()

            raw = read_exact(fifo, FRAME_BYTES)
            if raw is None:
                continue
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

            dist_mm = orient((linear[0:CH_LEN].view("<u2") & DIST_MASK).reshape(RAW_H, RAW_W).astype(np.float32),
                             state["orient"])

            motion_ref = slider_val[5]
            move_fx = slider_val[6] / 100.0
            density = slider_val[0] / 100.0
            max_dist = slider_val[7]
            bg_thresh = slider_val[8]

            jump_thresh = slider_val[9]
            min_vel = slider_val[10]
            edge_smooth = slider_val[11] / 100.0

            # EdgeSmooth previously only fed the Z-velocity jump/clamp
            # decision, which is why it looked like it "did nothing" -- it
            # never touched the SPAWN-triggering signal. That's also the real
            # reason edges kept emitting particles forever: a zone straddling
            # a sharp depth boundary can flicker between near/far every frame
            # from sensor mixed-pixel noise, which means its dist_smooth
            # value never truly settles, so it never gets absorbed into the
            # dynamic background and reads as permanent "foreground". Spatial
            # (neighbor-median) denoising BEFORE that comparison lets an
            # isolated flickering edge zone settle toward its stable
            # neighbors instead, so it actually converges into the
            # background like a real static zone should. Only used to feed
            # the smoothing/background pipeline -- z_for_pos and the actual
            # rendered position still come from the raw, un-blurred depth.
            if edge_smooth > 0.001:
                dist_spatial = cv2.medianBlur(dist_mm.astype(np.float32), 3)
                dist_proc = dist_mm + edge_smooth * (dist_spatial - dist_mm)
            else:
                dist_proc = dist_mm

            # EMA-smoothed depth: single-frame ToF sensor noise/jitter on an
            # otherwise-static wall can easily be tens of mm, which was enough
            # to blow past even a maxed-out BgThresh every few frames. Comparing
            # the SMOOTHED signal to the background filters that noise out while
            # still responding quickly to a real, sustained change.
            if dist_smooth is None or dist_smooth.shape != dist_proc.shape:
                dist_smooth = dist_proc.copy()
            else:
                dist_smooth = dist_smooth + 0.35 * (dist_proc - dist_smooth)

            # Dynamic background: a slow EMA that's always running, no button
            # press required. A zone that's been steady for a while gets
            # absorbed into it and stops spawning; only recent change stands
            # out as foreground. ~0.3s (@100fps) to meaningfully adapt.
            DYNAMIC_BG_ALPHA = 0.02
            if dist_bg_dynamic is None or dist_bg_dynamic.shape != dist_proc.shape:
                dist_bg_dynamic = dist_proc.copy()
            else:
                dist_bg_dynamic = dist_bg_dynamic + DYNAMIC_BG_ALPHA * (dist_proc - dist_bg_dynamic)

            old_prev_dist = prev_dist   # capture before overwriting below
            prev_dist = dist_mm

            # Mixed-pixel / flying-pixel edge noise: a zone straddling a sharp
            # depth boundary can flicker frame-to-frame between the near and
            # far surface even while staying "foreground" the whole time (so
            # the background-transition mask above doesn't catch it). Any jump
            # bigger than JumpThresh is treated as that flicker, not real
            # motion: velocity gets clamped down to MinVel, and the position
            # snaps to the NEARER of the two readings (the real object edge is
            # almost always the close surface, not the far one bleeding through).
            #
            # A single time-domain threshold can't fully tell "one isolated
            # zone flickering" from "this whole hand is genuinely moving fast
            # in Z" -- both can produce the same size frame-to-frame delta at
            # a single zone. EdgeSmooth adds a SPATIAL check ON TOP of (never
            # instead of) the magnitude check: a zone whose delta disagrees
            # sharply with its own 3x3-neighborhood median delta is an
            # isolated flying-pixel outlier and gets flagged too, even if its
            # raw magnitude alone wouldn't cross JumpThresh. This is a strict
            # OR -- it can only catch MORE flicker, never let a large raw
            # delta slip through un-clamped (an earlier version blended the
            # distance itself toward the spatial median before the magnitude
            # check, which could pull a real edge's delta below JumpThresh and
            # let it through un-clamped -- exactly backwards).
            if old_prev_dist is not None and old_prev_dist.shape == dist_mm.shape:
                raw_delta = dist_mm - old_prev_dist   # always the TRUE, un-blended delta
                is_jump = np.abs(raw_delta) > jump_thresh
                if edge_smooth > 0.001:
                    local_med = cv2.medianBlur(raw_delta.astype(np.float32), 3)
                    outlier_thresh = jump_thresh * (1.0 - edge_smooth * 0.5)
                    is_jump = is_jump | (np.abs(raw_delta - local_med) > outlier_thresh)
                z_for_pos = np.where(is_jump, np.minimum(dist_mm, old_prev_dist), dist_mm)
                # MinVel does two jobs now: (1) a deadband -- any residual
                # delta smaller than MinVel is sensor-noise-level and gets
                # zeroed, which is ALWAYS active regardless of JumpThresh, so
                # the slider has a visible effect even when JumpThresh is set
                # high enough that is_jump rarely fires; (2) still the clamp
                # ceiling for zones that DO get flagged as a flicker jump.
                deadbanded = np.where(np.abs(raw_delta) < min_vel, 0.0, raw_delta)
                clamped_delta = np.where(is_jump, np.sign(raw_delta) * min_vel, deadbanded)
            else:
                z_for_pos = dist_mm
                clamped_delta = np.zeros_like(dist_mm)

            # Manual background capture was removed -- it consistently made
            # results worse (a stale, one-shot reference drifts out of date
            # and starts reading real background as "foreground" noise). The
            # always-on, slowly-adapting dynamic background is strictly
            # better: it never goes stale and never needs a button press.
            bg = dist_bg_dynamic
            bg_active = bg is not None and bg.shape == dist_mm.shape
            if bg_active:
                # Activity = how far each zone differs from the background
                # reference. A person in front of the wall lights up whether or
                # not they're moving; the wall itself reads ~0 and is filtered.
                diff = np.abs(dist_smooth - bg)
                diff[bg <= 0] = 0.0          # zones with no reference -> ignore
                motion = diff
                # Smooth, continuous damper instead of a hard bg_thresh gate:
                # a hard cutoff means ordinary sensor noise randomly crossing
                # that line each frame flips a zone in/out of "foreground",
                # producing scattered spurious spawns ("noise spots") all over
                # an otherwise-static captured background. This ramps spawn
                # probability smoothly from 0 right at the background up to
                # full by ~2x bg_thresh. A hard zero below 15% of bg_thresh
                # (true noise-floor territory) plus a steeper cubic ramp above
                # that stops the "always slightly nonzero" trickle of
                # particles spawning on hundreds of truly-static zones and
                # just sitting there fading -- the earlier squared ramp was
                # too permissive right near zero for that to stay rare.
                diff_ratio = diff / max(bg_thresh, 1.0)
                bg_spawn_suppress = np.where(diff_ratio < 0.15, 0.0, np.clip(diff_ratio, 0.0, 1.0) ** 3)

                # A zone flipping background<->foreground is an OCCLUSION EDGE
                # event, not real depth velocity: the raw frame-to-frame delta
                # there is the whole wall-to-object jump, which was sending
                # particles rocketing off in Z far more than X/Y. Only trust a
                # delta when BOTH this frame and the last were foreground (two
                # real consecutive readings of the same moving object). A fresh
                # reveal (bg->fg) or a disappearance (fg->bg) gets zero velocity
                # instead of the spurious jump.
                if old_prev_dist is not None and old_prev_dist.shape == dist_mm.shape:
                    prev_is_bg = np.abs(old_prev_dist - bg) < bg_thresh
                else:
                    prev_is_bg = np.ones_like(dist_mm, dtype=bool)
                curr_is_bg = np.abs(dist_mm - bg) < bg_thresh
                valid_delta_mask = (~prev_is_bg) & (~curr_is_bg)
            elif old_prev_dist is not None and old_prev_dist.shape == dist_mm.shape:
                motion = np.abs(dist_mm - old_prev_dist)
                valid_delta_mask = np.ones_like(dist_mm, dtype=bool)
            else:
                motion = np.zeros_like(dist_mm)
                valid_delta_mask = np.ones_like(dist_mm, dtype=bool)

            # Short temporal EMA on the motion signal itself (not just the
            # raw depth): with no smoothing here, the instant an object's
            # frame-to-frame diff dropped, EVERY newly-spawned particle's
            # size/wobble/life fell straight to the floor with no transition
            # -- a visible snap right as motion settled, worse the lower
            # MoveFX was set (since it compresses the whole range down near
            # that floor already). This gives the "coming to rest" decay some
            # inertia instead.
            MOTION_SMOOTH_ALPHA = 0.25
            if motion_smooth is None or motion_smooth.shape != motion.shape:
                motion_smooth = motion.copy()
            else:
                motion_smooth = motion_smooth + MOTION_SMOOTH_ALPHA * (motion - motion_smooth)
            motion = motion_smooth

            # Optical flow between consecutive depth frames -- gives each
            # zone a real (dx, dy) apparent-motion vector, not just a scalar
            # magnitude. The grid is tiny (54x42 or smaller) so Farneback runs
            # in a fraction of a millisecond here.
            frame_energy = 0.0
            gray = np.clip(dist_smooth / MAX_MM * 255.0, 0, 255).astype(np.uint8)
            if prev_gray is not None and prev_gray.shape == gray.shape:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                     0.5, 2, 9, 2, 5, 1.1, 0)
            else:
                flow = np.zeros((RAW_H, RAW_W, 2), dtype=np.float32)
            prev_gray = gray

            if state["show_field"]:
                # Same per-zone velocity conversion used for spawning
                # particles, but for the WHOLE grid (not just spawning
                # zones), so the debug arrows can sample real, live motion
                # anywhere -- only computed while the field view is on.
                vel_x_full = flow[:, :, 0] * CLOUD_SCALE * FLOW_GAIN
                vel_y_full = flow[:, :, 1] * -1.0 * CLOUD_SCALE * FLOW_GAIN
                vel_z_full = clamped_delta / (CURVE_K + dist_mm) * (32.0 * ZDEPTH * CLOUD_SCALE)
                grid_vel = np.stack([vel_x_full[grid_row_idx, grid_col_idx],
                                      vel_y_full[grid_row_idx, grid_col_idx],
                                      vel_z_full[grid_row_idx, grid_col_idx]], axis=1)
                # Temporal EMA so arrows ease toward a new direction instead
                # of snapping to it every frame -- alpha=0.12 gives roughly a
                # quarter-second settling time at this framerate, enough to
                # kill single-frame Farneback jitter while still tracking
                # real, sustained motion promptly.
                if field_vel_smooth is None or field_vel_smooth.shape != grid_vel.shape:
                    field_vel_smooth = grid_vel.copy()
                else:
                    field_vel_smooth = field_vel_smooth + 0.12 * (grid_vel - field_vel_smooth)
                field_data[0::2, 4:7] = field_vel_smooth
                field_data[1::2, 4:7] = field_vel_smooth
                field_vbo.write(field_data.tobytes())

            valid = (dist_mm > 0) & (dist_mm <= max_dist)
            ys, xs = np.nonzero(valid)
            if xs.size:
                z = z_for_pos[ys, xs]
                # Small random jitter on spawn position so particles don't sit
                # in a visible lattice at exact zone centers.
                JITTER = 0.4  # object-space units, a fraction of one zone width
                x = (xs - RAW_W / 2.0 + np.random.uniform(-JITTER, JITTER, xs.size)).astype(np.float32) * CLOUD_SCALE
                y = (ys - RAW_H / 2.0 + np.random.uniform(-JITTER, JITTER, ys.size)).astype(np.float32) * -1.0 * CLOUD_SCALE
                log_z = np.log1p(z / CURVE_K)
                zc = (log_z - Z_REF_LOG) * (32.0 * ZDEPTH * CLOUD_SCALE)
                zc += np.random.uniform(-JITTER, JITTER, zc.size).astype(np.float32)

                mvals = motion[ys, xs]
                raw_ratio = (mvals / motion_ref) * move_fx   # unfloored -- for spawn probability only
                mscale = np.clip(raw_ratio, MOTION_MIN_SCALE, MOTION_MAX_SCALE).astype(np.float32)

                norm = np.clip(255.0 * (1.0 - np.clip(z, 0, MAX_MM) / MAX_MM), 0, 255).astype(np.uint8)
                bgr = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
                rgb01 = (bgr[:, ::-1].astype(np.float32) / 255.0)

                # Inherited velocity for newly-spawned particles: (fx, fy) from
                # optical flow, converted to the same object-space units as
                # x/y; fz from the actual signed depth change, converted to
                # zc's log-space units via the chain rule on log1p(z/K) so all
                # three components share one consistent scale. Zeroed out on
                # background<->foreground transition zones (valid_delta_mask)
                # since those are occlusion-edge jumps, not real velocity.
                zone_ok = valid_delta_mask[ys, xs]
                fx = flow[ys, xs, 0] * CLOUD_SCALE * FLOW_GAIN * zone_ok
                fy = flow[ys, xs, 1] * -1.0 * CLOUD_SCALE * FLOW_GAIN * zone_ok
                delta_z_mm = clamped_delta[ys, xs] * zone_ok
                fz = delta_z_mm / (CURVE_K + z) * (32.0 * ZDEPTH * CLOUD_SCALE)
                velocity = np.stack([fx, fy, fz], axis=1).astype(np.float32)

                # The actual fix for "static regions spawn tons of particles":
                # each zone's spawn PROBABILITY this frame is driven by its own
                # motion, not just a uniform per-zone rate. This floor is much
                # lower than MOTION_MIN_SCALE (which only floors the *visual*
                # size/alpha of whatever rare particle does spawn) -- otherwise
                # even an 8% per-frame chance adds up to thousands of overlapping
                # faint background particles at once across a typical scene,
                # which still reads as a bright haze under additive blending.
                # Density scales this probability uniformly on top. With a
                # background captured, the floor drops to (a small amount *
                # bgsup) so filtered-out background zones (bgsup == 0) still
                # never spawn at all -- only real foreground changes do.
                #
                # That floor scaling by bgsup matters: raw_ratio alone gets
                # divided down by MotionRef, and at MotionRef's max nearly
                # every zone's raw_ratio collapses toward zero regardless of
                # Density, so almost nothing spawns -- the rare particle that
                # does spawn then has nothing replacing it as it decays,
                # reading as a glitchy flicker rather than a smooth thinning.
                # Flooring by bgsup (not a flat constant) keeps that floor
                # tied to "is this genuinely foreground right now" so it
                # can't reintroduce spam on truly static zones.
                bgsup = bg_spawn_suppress[ys, xs] if bg_active else 1.0
                SPAWN_MIN_PROB = (0.02 * bgsup) if bg_active else 0.006
                spawn_prob = np.clip(raw_ratio * density * bgsup, SPAWN_MIN_PROB, 1.0)
                keep = np.random.random(x.size) < spawn_prob
                x, y, zc, mscale, rgb01, velocity = \
                    x[keep], y[keep], zc[keep], mscale[keep], rgb01[keep], velocity[keep]

                # Cheap system-"energy" estimate (# of spawning particles *
                # their average speed/motion) for the shader's flow-field
                # amplitude -- no GPU readback needed, just reused numbers we
                # already computed this frame for spawning.
                frame_energy = float(x.size) * float(mscale.mean()) if x.size else 0.0

                n_new = min(x.size, PARTICLE_MAX)
                if n_new > 0:
                    now = time.time() - start_time
                    rec = np.empty((n_new, RECORD_FLOATS), dtype="f4")
                    rec[:, 0] = x[:n_new]; rec[:, 1] = y[:n_new]; rec[:, 2] = zc[:n_new]
                    rec[:, 3] = now
                    rec[:, 4] = mscale[:n_new]
                    rec[:, 5:8] = rgb01[:n_new]
                    rec[:, 8:11] = velocity[:n_new]

                    idx = (np.arange(n_new) + _cursor) % PARTICLE_MAX
                    # contiguous fast path (no wrap) vs split write on wraparound
                    if idx[-1] - idx[0] == n_new - 1:
                        vbo.write(rec.tobytes(), offset=idx[0] * RECORD_FLOATS * 4)
                    else:
                        split = PARTICLE_MAX - _cursor
                        vbo.write(rec[:split].tobytes(), offset=_cursor * RECORD_FLOATS * 4)
                        vbo.write(rec[split:].tobytes(), offset=0)
                    _cursor = (_cursor + n_new) % PARTICLE_MAX

            # Smooth the raw per-frame energy estimate so the field amplitude
            # doesn't jitter with single-frame spawn-count noise.
            ENERGY_ALPHA = 0.08
            state["energy"] += ENERGY_ALPHA * (frame_energy - state["energy"])

            fbw, fbh = glfw.get_framebuffer_size(win)
            ctx.viewport = (0, 0, fbw, fbh)
            ctx.clear(0.0, 0.0, 0.0)
            ctx.blend_func = moderngl.ONE, moderngl.ONE   # additive glow for particles
            prog["u_time"].value = time.time() - start_time
            prog["u_life"].value = slider_val[4]
            prog["u_azimuth"].value = np.radians(state["az"])
            prog["u_tilt"].value = np.radians(state["tilt"])
            prog["u_zoom"].value = state["zoom"]
            prog["u_pan"].value = tuple(state["pan"])
            prog["u_canvas"].value = (float(fbw), float(fbh))
            prog["u_base_size"].value = slider_val[1]
            prog["u_speed"].value = slider_val[2]
            prog["u_freq"].value = slider_val[3]
            prog["u_noise_amp"].value = slider_val[12]
            prog["u_depth_glow"].value = slider_val[13] / 100.0
            prog["u_energy"].value = state["energy"]
            prog["u_color_mode"].value = float(state["color_mode"])
            prog["u_spawn_hsv"].value = (slider_val[14] / 360.0, slider_val[15], slider_val[16])
            prog["u_death_hsv"].value = (slider_val[17] / 360.0, slider_val[18], slider_val[19])
            prog["u_twinkle"].value = slider_val[20]
            vao.render(moderngl.POINTS, vertices=PARTICLE_MAX)

            if state["show_field"]:
                ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                field_prog["u_time"].value = time.time() - start_time
                field_prog["u_freq"].value = slider_val[3]
                field_prog["u_arrow_len"].value = 3.0
                field_prog["u_azimuth"].value = np.radians(state["az"])
                field_prog["u_tilt"].value = np.radians(state["tilt"])
                field_prog["u_zoom"].value = state["zoom"]
                field_prog["u_pan"].value = tuple(state["pan"])
                field_prog["u_canvas"].value = (float(fbw), float(fbh))
                field_prog["u_flow_vis_scale"].value = 1.0
                field_vao.render(moderngl.LINES, vertices=n_field * 2)

            # Slider panel, drawn with normal (non-additive) alpha blending.
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            for i, (name, lo, hi, _d) in enumerate(SLIDER_DEFS):
                sx0, sy0, sx1, sy1 = slider_rect_px(i)
                draw_quad(sx0, sy0, sx1, sy1, fbw, fbh, (0.22, 0.22, 0.22))
                frac = (slider_val[i] - lo) / (hi - lo)
                fill_x = sx0 + int(frac * (sx1 - sx0))
                draw_quad(sx0, sy0, fill_x, sy1, fbw, fbh, (0.0, 0.8, 1.0))
                tex, tw, th = slider_labels[i]
                ty = sy0 + (SLIDER_H - th) // 2
                draw_text(tex, tw, th, sx0 + 3, ty, fbw, fbh)

            for i, (name, act) in enumerate(BUTTONS):
                bx0, by0, bx1, by1 = button_rect_px(i)
                draw_quad(bx0, by0, bx1, by1, fbw, fbh, (0.32, 0.32, 0.32))
                if act == "cycle_mode":
                    tex, tw, th = color_mode_labels[state["color_mode"]]
                elif act == "toggle_field":
                    tex, tw, th = field_labels[int(state["show_field"])]
                else:
                    tex, tw, th = button_labels[i]
                draw_text(tex, tw, th, bx0 + 6, by0 + (BTN_H - th) // 2, fbw, fbh)

            glfw.swap_buffers(win)

            fps_n += 1
            now_t = time.time()
            if now_t - fps_t0 >= 1.0:
                fps = fps_n / (now_t - fps_t0)
                fps_n, fps_t0 = 0, now_t
                glfw.set_window_title(win, f"VL53L9CX GPU Particles  {fps:.0f} FPS  {PARTICLE_MAX} particles")
                print(f"[gpu] {fps:.0f} FPS", flush=True)

    glfw.destroy_window(win)
    glfw.terminate()
    print("[gpu] stopped.")

if __name__ == "__main__":
    main()
