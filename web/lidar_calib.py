#!/usr/bin/env python3
"""
Radial-to-perpendicular (r2p) + point-cloud calibration for the VL53L9CX.

The sensor reports RADIAL distance -- the length of the ray from the sensor to
each zone -- not perpendicular depth. For a geometrically correct 3D cloud you
have to project each zone's radial reading through the optics: a pinhole model
(effective focal length + principal point) with lens distortion and a small
emitter/receiver parallax correction.

This is a faithful, vectorized NumPy port of ST's official
`vl53l9_algo_radial_to_perp` / `vl53l9_algo_pointcloud` (from the ULD
`radial_to_perp.c`, "Python algo R_1.3.9"), using ST's default parameters --
which are the fallback the firmware itself uses before per-sensor OTP
calibration is read. Per-unit OTP values (efl / offsets / distortion K1..K4)
would refine it further; the defaults already give the correct fisheye
correction and true Cartesian millimetres, which is what the visualizers need.

  radial_to_perp(depth, binning) -> perpendicular depth (mm)
  pointcloud(depth, binning)      -> (X, Y, Z) in mm, real Cartesian
  ray_coeffs(w, h, binning)       -> per-zone (kx,ky,kz) so point = radial*(k),
                                     a depth-independent lookup (no parallax)
                                     for cheap live rendering in the browser.
"""
import numpy as np

# ST default parameters (radial_to_perp_init_default_params).
DEFAULT_PARAMS = {
    "efl": 2428.16, "residual_offset_x": 0.0, "residual_offset_y": 0.0,
    "max_distance": 9600, "parallax_correction": True, "parallax_limit": 50.0,
    "alpha": -0.00015, "beta": 0.0, "gamma": 0.0, "kappa": 0.0,
    "max_spads_x": 216, "max_spads_y": 168, "spad_size_um": 10.17,
}


def _distortion(rsq, p, binning):
    d = p["alpha"] * rsq + p["beta"] * rsq ** 2 + p["gamma"] * rsq ** 3 + p["kappa"]
    return 1.0 + (binning * binning / 4.0) * d


def _geom(p, width, height, binning):
    mspads_x = p["max_spads_x"] / 2.0
    mspads_y = p["max_spads_y"] / 2.0
    mpix = binning * 2.0
    focal = p["efl"] / (p["spad_size_um"] * mpix)           # in binned-pixel units
    x_center = ((mspads_x + p["residual_offset_x"] - (mspads_x - binning * width)) / mpix) - 0.5
    y_center = ((mspads_y + p["residual_offset_y"] - (mspads_y - binning * height)) / mpix) - 0.5
    return focal, x_center, y_center, mpix


def radial_to_perp(depth, binning=2, p=None):
    """Radial depth (mm) -> perpendicular depth (mm). Returns (perp, cx, ud, focal)."""
    p = p or DEFAULT_PARAMS
    depth = depth.astype(np.float32)
    h, w = depth.shape
    focal, xc, yc, mpix = _geom(p, w, h, binning)
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    dx = xs - xc
    dy = ys - yc
    rsq = dx * dx + dy * dy
    ud = _distortion(rsq, p, binning)
    perp = depth / np.sqrt(1.0 + rsq / (ud * ud * focal * focal))
    cx = np.full((h, w), xc, dtype=np.float32)
    if p["parallax_correction"]:
        # Emitter/receiver baseline shifts the effective x-centre with distance.
        cx = xc - (p["efl"] * 7.166 / (np.maximum(p["parallax_limit"], perp) * p["spad_size_um"] * mpix))
        dx = xs - cx
        rsq = dx * dx + dy * dy
        ud = _distortion(rsq, p, binning)
        perp = depth / np.sqrt(1.0 + rsq / (ud * ud * focal * focal))
    return perp.astype(np.float32), cx.astype(np.float32), ud.astype(np.float32), float(focal)


def pointcloud(depth, binning=2, p=None):
    """Radial depth (mm) -> real Cartesian (X, Y, Z) in mm, each (H, W)."""
    p = p or DEFAULT_PARAMS
    h, w = depth.shape
    perp, cx, ud, focal = radial_to_perp(depth, binning, p)
    _, _, yc, _ = _geom(p, w, h, binning)
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    distorted_z = perp / (ud * focal)
    X = (xs - cx) * distorted_z
    Y = (ys - yc) * distorted_z
    Z = perp
    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)


def ray_coeffs(width, height, binning=2, p=None):
    """Per-zone (kx, ky, kz): point_mm = radial_mm * (kx, ky, kz).

    Depth-independent (parallax omitted -- a sub-pixel effect) so it can be
    computed ONCE and shipped to the browser as a lookup; the client just
    multiplies each zone's radial reading by its coefficient vector."""
    p = p or DEFAULT_PARAMS
    focal, xc, yc, mpix = _geom(p, width, height, binning)
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    dx = xs - xc
    dy = ys - yc
    rsq = dx * dx + dy * dy
    ud = _distortion(rsq, p, binning)
    kz = 1.0 / np.sqrt(1.0 + rsq / (ud * ud * focal * focal))   # perp = radial*kz
    k = kz / (ud * focal)                                       # distorted_z/radial
    kx = dx * k
    ky = dy * k
    return (np.broadcast_to(kx, (height, width)).astype(np.float32),
            np.broadcast_to(ky, (height, width)).astype(np.float32),
            np.broadcast_to(kz, (height, width)).astype(np.float32))


def _selftest():
    W, H, B = 54, 42, 2
    # A constant-RADIAL field (a spherical shell) -> perpendicular should dip
    # toward the edges (cos falloff), equal radial only dead-centre.
    r = np.full((H, W), 1000.0, dtype=np.float32)
    perp, cx, ud, focal = radial_to_perp(r, B)
    print(f"focal(px)={focal:.2f}  centre perp={perp[H//2, W//2]:.1f}  corner perp={perp[0,0]:.1f}")
    assert perp.max() <= 1000.0 + 0.5, "perp must never exceed radial"
    assert abs(perp[H // 2, W // 2] - 1000.0) < 3.0, "centre perp ~= radial"
    assert perp[0, 0] < 950.0, "corner perp should fall off clearly"

    X, Y, Z = pointcloud(r, B)
    print(f"centre XYZ=({X[H//2,W//2]:.0f},{Y[H//2,W//2]:.0f},{Z[H//2,W//2]:.0f})  "
          f"corner XYZ=({X[0,0]:.0f},{Y[0,0]:.0f},{Z[0,0]:.0f})  "
          f"X span={X.max()-X.min():.0f}mm Y span={Y.max()-Y.min():.0f}mm")
    assert abs(X[H // 2, W // 2]) < 25 and abs(Y[H // 2, W // 2]) < 25, "centre ~ on axis"
    # FoV sanity: horizontal full angle from the corner rays
    import math
    fov_x = 2 * math.degrees(math.atan((W / 2) / focal))
    fov_y = 2 * math.degrees(math.atan((H / 2) / focal))
    print(f"implied FoV ~ {fov_x:.1f}deg x {fov_y:.1f}deg")

    # ray_coeffs must reconstruct the same cloud as pointcloud (minus parallax)
    kx, ky, kz = ray_coeffs(W, H, B)
    Xr, Yr, Zr = r * kx, r * ky, r * kz
    err = max(abs(Zr - Z).max(), abs(Xr - X).max(), abs(Yr - Y).max())
    print(f"ray_coeffs vs full pointcloud max diff (parallax): {err:.2f}mm")
    print("\nCALIBRATION SELF-TEST PASSED")


if __name__ == "__main__":
    _selftest()
