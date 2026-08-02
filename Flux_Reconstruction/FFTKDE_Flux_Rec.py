# -*- coding: utf-8 -*-
"""
FFTKDE_Flux_Rec.py
=====================
Heat Flux Reconstruction via Fast Fourier Transform
Kernel Density Estimation (FFTKDE) for cylindrical receivers.

Part of: A Grid-Independent Flux Mapping Approach for Cylindrical Solar Receivers
Authors: Sainath Waghmare et al., UIB, 2026. 

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage    import gaussian_filter
from KDEpy            import FFTKDE
import time
import os
import warnings
warnings.filterwarnings('ignore')


# ================================================================
#  SECTION 1 — USER PARAMETERS  (edit these before running)
# ================================================================

ASTM_FILE = "data/ASTMG173-direct.xlsx"   # path to ASTM G173-03 spectrum
HITS_FILE = "data/ray_hits.txt"            # path to OTSun ray-hit output file

OUTPUT_DIR = (
    "Results/FFTKDE"                # path to save results
)

# Geometry
receiver_outer_radius_mm = 35.0   # outer radius [mm]
receiver_length_m        = 4.0    # receiver length [m]
aperture_width_m         = 5.76   # parabolic mirror aperture width [m]
collector_length_m       = 4.0    # collector length [m]

# MCRT settings — must match the HITS_FILE
rays_per_wavelength = 20000       # rays launched per wavelength band
dlambda_nm          = 20.0        # wavelength band width [nm]
wl_min_nm           = 290.0       # ASTM G173-03 lower bound [nm]
wl_max_nm           = 3990.0      # ASTM G173-03 upper bound [nm]

# Receiver axis in the OTSun coordinate frame
axis  = "y"                       # "x", "y", or "z"
units = "mm"                      # coordinate unit of OTSun hit file

# RAM-safe chunk size for streaming large hit files
chunksize = 1_000_000

# FFTKDE internal grid — power of 2 strongly recommended
# 512 suffices for most ray counts; use 1024 for N > 5M rays
FFTKDE_GRID = 1024

# Visualisation and noise-metric grid (independent of FFTKDE_GRID)
N_VIS_THETA = 200
N_VIS_S     = 100

# CSV export resolutions — all produced from ONE FFTKDE fit at zero extra cost
EXPORT_RESOLUTIONS = [
    ( 20,   20),
    ( 80,   80),
    (300,  300),
    ( 60,  400),
    ( 80,  600),
    (100, 1000),
]

# Set True ONLY for N <= 5,000 rays/wavelength (scipy is too slow for more)
RUN_SCIPY_VALIDATION = False

# Tube centre — None triggers automatic estimation from hit normals
center1 = None
center2 = None

CMAP = cm.jet
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
#  SECTION 2 — LOAD ASTM G173-03 SPECTRUM
#
#  Builds the cumulative spectral irradiance function F(λ) [W/m²]
#  used to assign a physical power weight to every traced ray.
# ================================================================
print("=" * 62)
print("  METHOD 3 — FFTKDE FLUX RECONSTRUCTION")
print("=" * 62)
print("\n[1] Loading ASTM G173-03 direct spectrum...")

astm     = pd.read_excel(ASTM_FILE, engine="openpyxl").dropna()
wl_astm  = astm.iloc[:, 0].to_numpy(float)   # wavelength [nm]
E_astm   = astm.iloc[:, 1].to_numpy(float)   # irradiance [W/m²/nm]

# Cumulative irradiance via trapezoidal rule: F(λ) = ∫G_b dλ' [W/m²]
cumE = np.zeros_like(wl_astm)
cumE[1:] = np.cumsum(
    0.5 * (E_astm[1:] + E_astm[:-1]) * (wl_astm[1:] - wl_astm[:-1])
)

A_ap   = aperture_width_m * collector_length_m          # aperture area [m²]
R      = receiver_outer_radius_mm * 1e-3                 # outer radius [m]
period = 2.0 * np.pi * R                                # circumference [m]
scale  = 1e-3 if units.lower() == "mm" else 1.0         # hit-coord → metres

axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
cross    = [i for i in (0, 1, 2) if i != axis_idx]     # two radial axes

print(f"  Aperture area:    {A_ap:.4f} m²")
print(f"  Outer radius R:   {R*1000:.1f} mm")
print(f"  Circumference:    {period*1000:.2f} mm")


# ================================================================
#  SECTION 3 — AUTOMATED TUBE CENTRE ESTIMATION
#
#  Fits the tube centre by minimising the RMS residual between
#  each hit's measured radial distance and R, trying both inward
#  and outward surface normal projections.
# ================================================================
print("\n[2] Estimating tube centre from hit normals...")

if center1 is None or center2 is None:
    cols_local = ["energy_ray","x","y","z","px","py","pz",
                  "nx","ny","nz","wl_nm"]
    sample = pd.read_csv(
        HITS_FILE, delim_whitespace=True, comment="#",
        names=cols_local, header=None, nrows=200000
    )
    P_s = sample[["x","y","z"]].to_numpy(float) * scale
    N_s = sample[["nx","ny","nz"]].to_numpy(float)
    i1, i2 = cross[0], cross[1]

    def fit_centre(Ccand):
        c1 = Ccand[:, i1].mean()
        c2 = Ccand[:, i2].mean()
        r  = np.sqrt((P_s[:, i1]-c1)**2 + (P_s[:, i2]-c2)**2)
        return np.sqrt(np.mean((r - R)**2)), c1, c2

    err_m, c1_m, c2_m = fit_centre(P_s - R * N_s)
    err_p, c1_p, c2_p = fit_centre(P_s + R * N_s)
    center1, center2   = (c1_m, c2_m) if err_m <= err_p else (c1_p, c2_p)

print(f"  Tube centre (cross-section plane): "
      f"({center1:.6f}, {center2:.6f}) m")


# ================================================================
#  SECTION 4 — STREAM HIT FILE  →  ARRAYS (θ, s, w)
#
#  For every absorbed ray i:
#    s_i   = position along receiver axis [m]
#    θ_i   = circumferential angle ∈ [0, 2π), shifted so:
#              θ = 0°/360° → tube top  (low flux, facing sky)
#              θ = 180°    → tube bottom (peak flux, facing mirror)
#    w_i   = physical power deposited [W]
#           = ε_i × ΔG_j × A_ap / N_λ
# ================================================================
print("\n[3] Streaming OTSun hit file and computing ray weights...")

cols = ["energy_ray","x","y","z","px","py","pz","nx","ny","nz","wl_nm"]

P_abs_code = 0.0
theta_list, s_list, w_list = [], [], []

t_load_start = time.time()
reader = pd.read_csv(
    HITS_FILE, delim_whitespace=True, comment="#",
    names=cols, header=None, chunksize=chunksize
)
chunk_count = 0
for chunk in reader:
    Pos = chunk[["x","y","z"]].to_numpy(float) * scale
    s_c = Pos[:, axis_idx]
    u   = Pos[:, cross[0]] - center1
    v   = Pos[:, cross[1]] - center2

    # Angular coordinate — shift so 180° = concentration zone (bottom)
    theta = np.arctan2(v, u) - (np.pi / 2.0)
    theta = np.where(theta < 0,        theta + 2*np.pi, theta)
    theta = np.where(theta >= 2*np.pi, theta - 2*np.pi, theta)

    # Spectral energy weight from ASTM G173-03
    wl    = chunk["wl_nm"].to_numpy(float)
    a     = np.maximum(wl - 0.5*dlambda_nm, wl_min_nm)
    b     = np.minimum(wl + 0.5*dlambda_nm, wl_max_nm)
    Fa    = np.interp(a, wl_astm, cumE, left=cumE[0], right=cumE[-1])
    Fb    = np.interp(b, wl_astm, cumE, left=cumE[0], right=cumE[-1])
    P_ray = ((Fb - Fa) * A_ap) / rays_per_wavelength   # [W per ray]
    w_c   = chunk["energy_ray"].to_numpy(float) * P_ray

    P_abs_code += w_c.sum()
    theta_list.append(theta)
    s_list.append(s_c)
    w_list.append(w_c)
    chunk_count += 1
    if chunk_count % 10 == 0:
        n_so_far = sum(len(x) for x in theta_list)
        print(f"    ... {n_so_far:,} rays loaded", end="\r")

t_load_end = time.time()
theta_all  = np.concatenate(theta_list)
s_all      = np.concatenate(s_list)
w_all      = np.concatenate(w_list)
N_total    = len(theta_all)

print(f"\n  Total rays processed: {N_total:,}")
print(f"  Total absorbed power: {P_abs_code:.4f} W")
print(f"  Data loading time:    {t_load_end - t_load_start:.1f} s")


# ================================================================
#  SECTION 5 — FFTKDE FLUX RECONSTRUCTION
#
#  (a) Compute Scott's bandwidth manually:
#        h = N^{-1/(d+4)} = N^{-1/6}  for d=2
#  (b) Normalise BOTH axes to zero-mean / unit-variance so that
#      a single scalar h applies consistently in each direction.
#  (c) Pass h as a float to FFTKDE — the only bandwidth mode
#      KDEpy FFTKDE supports for multi-dimensional data.
#  (d) After evaluate(), un-normalise grid coordinates back to
#      physical metres.
#  (e) Convert normalised density → physical density [m^-2] using
#      the Jacobian of the inverse transform:
#        density_phys = density_norm / (std_x × std_y)
#  (f) Convert physical density → heat flux [W/m²]:
#        q(θ,s) = density_phys(θR, s) × P_MCRT / R
#  (g) Enforce exact energy conservation via global rescaling.
# ================================================================
print("\n[4] Building 9-fold augmented training dataset...")

x_arc = theta_all * R    # circumferential arc-length [m], range [0, 2πR]
y_ax  = s_all            # axial position [m], range [0, L]

# --- Augment: 3 θ copies (periodic BC) × 3 s copies (reflective BC) ---
x_aug_parts, y_aug_parts = [], []
for x_shift in [0.0, -period, +period]:
    for y_mirror in [y_ax, -y_ax, 2*receiver_length_m - y_ax]:
        x_aug_parts.append(x_arc + x_shift)
        y_aug_parts.append(y_mirror)

x_aug = np.concatenate(x_aug_parts)    # shape: (9N,)
y_aug = np.concatenate(y_aug_parts)    # shape: (9N,)
w_aug = np.tile(w_all, 9)              # shape: (9N,) — replicate weights

print(f"  Augmented size: {len(x_aug):,} points  (= 9 × {N_total:,})")
print(f"  x range: [{x_aug.min():.4f}, {x_aug.max():.4f}] m")
print(f"  y range: [{y_aug.min():.4f}, {y_aug.max():.4f}] m")

# ---------------------------------------------------------------
# (a)  Scott's bandwidth (d=2):  h = N_total^{-1/6}
#      Use N_total, not N_aug, because the 9 copies are exact
#      duplicates and add no new statistical information.
# ---------------------------------------------------------------
h_scott = N_total ** (-1.0 / 6.0)
print(f"\n  Scott's rule bandwidth (normalised space): {h_scott:.6f}")

# ---------------------------------------------------------------
# (b)  Normalise each axis to zero-mean / unit-variance
# ---------------------------------------------------------------
x_mean = x_arc.mean()
x_std  = x_arc.std()     # std of arc-length in [0, 2πR]
y_mean = y_ax.mean()
y_std  = y_ax.std()      # std of axial position in [0, L]

# Normalise augmented data using ORIGINAL data statistics
x_norm = (x_aug - x_mean) / x_std
y_norm = (y_aug - y_mean) / y_std

data_norm = np.column_stack([x_norm, y_norm])   # (9N, 2), ~N(0,1) per axis
w_norm    = w_aug / w_aug.sum()                  # normalise weights

# Effective bandwidths in physical (metric) space for reporting
h_eff_x = h_scott * x_std    # [m] circumferential direction
h_eff_y = h_scott * y_std    # [m] axial direction
h_eff   = h_scott * np.sqrt(x_std * y_std)   # [m] geometric mean

print(f"  Scott's rule bandwidth:")
print(f"    h (normalised):        {h_scott:.6f}")
print(f"    x_std (original data): {x_std*1000:.3f} mm")
print(f"    y_std (original data): {y_std*1000:.1f} mm")
print(f"    h_x (circumferential): {h_eff_x*1000:.3f} mm")
print(f"    h_y (axial):           {h_eff_y*1000:.1f} mm")
print(f"    h_eff (geom. mean):    {h_eff*1000:.3f} mm")


# ---------------------------------------------------------------
# (c)  Fit FFTKDE with scalar bandwidth on normalised data
# ---------------------------------------------------------------
print(f"\n[5] Fitting FFTKDE (grid={FFTKDE_GRID}×{FFTKDE_GRID}, "
      f"h={h_scott:.4f})...")
t_kde_start = time.time()

kde = FFTKDE(kernel='gaussian', bw=h_scott).fit(data_norm, weights=w_norm)

# ---------------------------------------------------------------
# (d)  Evaluate on FFTKDE_GRID × FFTKDE_GRID regular grid
#
#  KDEpy FFTKDE.evaluate(n) for 2D data returns:
#    (grid_pts, density)
#    grid_pts : shape (n^2, 2) — flat array of normalised grid points
#    density  : shape (n^2,)  — density in normalised units [1/norm²]
#
#  The grid spans the data range ± a few bandwidths in each axis.
# ---------------------------------------------------------------
eval_result = kde.evaluate(FFTKDE_GRID)

if not (isinstance(eval_result, tuple) and len(eval_result) == 2):
    raise RuntimeError(
        "FFTKDE.evaluate() did not return a (grid, density) tuple.\n"
        "Please upgrade KDEpy:  pip install --upgrade KDEpy"
    )

grid_pts_norm, density_norm_flat = eval_result
# grid_pts_norm     : (FFTKDE_GRID^2, 2)  in normalised coordinates
# density_norm_flat : (FFTKDE_GRID^2,)    in [1/norm_unit^2]

t_kde_end   = time.time()
t_kde_total = t_kde_end - t_kde_start
print(f"  FFTKDE fit + evaluate time: {t_kde_total:.2f} s")

# ---------------------------------------------------------------
# (d-ii)  Extract unique axis values and reshape to 2D
# ---------------------------------------------------------------
x_norm_unique = np.unique(grid_pts_norm[:, 0])   # FFTKDE_GRID values
y_norm_unique = np.unique(grid_pts_norm[:, 1])   # FFTKDE_GRID values
density_norm_2d = density_norm_flat.reshape(
    len(x_norm_unique), len(y_norm_unique)
)

# ---------------------------------------------------------------
# (d-iii)  Un-normalise grid coordinates back to physical metres
#   x_phys = x_norm × x_std + x_mean
#   y_phys = y_norm × y_std + y_mean
# ---------------------------------------------------------------
x_kde_m = x_norm_unique * x_std + x_mean   # [m], arc-length axis
y_kde_m = y_norm_unique * y_std + y_mean   # [m], axial axis

print(f"  Physical grid x range: [{x_kde_m.min():.4f}, {x_kde_m.max():.4f}] m")
print(f"  Physical grid y range: [{y_kde_m.min():.4f}, {y_kde_m.max():.4f}] m")

# ---------------------------------------------------------------
# (e)  Convert normalised density → physical density [m^-2]
#
#  Jacobian of the inverse linear transform:
#    dx = x_std · dx_norm,   dy = y_std · dy_norm
#  Therefore:
#    density_phys(x,y) = density_norm(x_norm,y_norm) / (x_std × y_std)
#
#  Verification: ∫density_phys dx dy
#              = ∫density_norm/(x_std·y_std) · x_std·dx_norm · y_std·dy_norm
#              = ∫density_norm dx_norm dy_norm = 1  ✓
# ---------------------------------------------------------------
density_phys_2d = density_norm_2d / (x_std * y_std)   # [m^-2]

# ---------------------------------------------------------------
#  Crop to physical domain  x ∈ [0, 2πR],  y ∈ [0, L]
# ---------------------------------------------------------------
mask_x       = (x_kde_m >= 0.0) & (x_kde_m <= period)
mask_y       = (y_kde_m >= 0.0) & (y_kde_m <= receiver_length_m)
x_phys       = x_kde_m[mask_x]
y_phys       = y_kde_m[mask_y]
theta_phys   = x_phys / R                                     # radians [0,2π]
density_phys = density_phys_2d[np.ix_(mask_x, mask_y)]       # [m^-2]

print(f"  Physical domain grid after crop: "
      f"{len(x_phys)} × {len(y_phys)} points")

# ---------------------------------------------------------------
# (f)  Convert physical density → heat flux [W/m²]
#
#  q(θ,s) = ρ_phys(θR, s) × P_MCRT / R
#
#  Derivation: ∫q·R·dθ·ds = ∫ρ_phys·dx·dy × P_MCRT = P_MCRT  ✓
#  (since ∫ρ_phys dx dy = 1 over the full domain)
# ---------------------------------------------------------------
flux_phys = np.clip(density_phys * P_abs_code / R, 0.0, None)

# ---------------------------------------------------------------
# (g)  Exact energy conservation rescaling
#
#  The 9-fold augmentation and cropping introduce a small imbalance.
#  A single global factor enforces:  Σ q_i · A_i = P_MCRT  exactly.
# ---------------------------------------------------------------
dtheta_p        = (theta_phys[-1] - theta_phys[0]) / (len(theta_phys) - 1)
ds_p            = (y_phys[-1]     - y_phys[0])     / (len(y_phys)     - 1)
dA_p            = R * dtheta_p * ds_p
P_recon         = (flux_phys * dA_p).sum()
conserv_err_pct = (P_recon - P_abs_code) / P_abs_code * 100.0
flux_phys      *= (P_abs_code / P_recon)   # exact closure

print(f"\n  Conservation error (before rescaling): {conserv_err_pct:.4f} %")
print(f"  Reconstructed power (after rescaling): "
      f"{(flux_phys * dA_p).sum():.4f} W  "
      f"(target: {P_abs_code:.4f} W)")
print(f"  Max flux on physical FFT grid:         {flux_phys.max():.1f} W/m²")

# ---------------------------------------------------------------
#  Build physical-domain interpolator for all downstream use
#  Inputs:  (θ [rad], s [m])    Output: q [W/m²]
# ---------------------------------------------------------------
interp_phys = RegularGridInterpolator(
    (theta_phys, y_phys), flux_phys,
    method='linear', bounds_error=False, fill_value=0.0
)

# Store effective bandwidth for reporting
h_bandwidth = h_eff   # geometric-mean effective bandwidth [m]


# ================================================================
#  SECTION 6 — NOISE AND VOLATILITY METRICS
#  (same framework as k-NN and scipy KDE codes for fair comparison)
# ================================================================
print("\n[6] Computing noise and volatility metrics...")

# Evaluate on standard 200×100 visualisation grid
theta_vis = np.linspace(0, 2*np.pi, N_VIS_THETA)
s_vis     = np.linspace(0, receiver_length_m, N_VIS_S)
TV, SV    = np.meshgrid(theta_vis, s_vis, indexing='ij')
vis_pts   = np.column_stack([TV.ravel(), SV.ravel()])
flux_vis  = np.clip(
    interp_phys(vis_pts).reshape(N_VIS_THETA, N_VIS_S), 0.0, None
)

# Re-enforce energy conservation on vis grid
dA_vis   = R * (2*np.pi / N_VIS_THETA) * (receiver_length_m / N_VIS_S)
flux_vis *= P_abs_code / (flux_vis * dA_vis).sum()

q_max_vis = flux_vis.max()
q_avg_vis = (flux_vis * dA_vis).sum() / (2*np.pi * R * receiver_length_m)

# Focal zone mask: faces with > 10% of peak flux
focal_mask = flux_vis > (0.10 * q_max_vis)

# Global Surface Roughness Index (RI):
#   macro trend  = Gaussian-filtered flux (sigma=3 bins, periodic in θ)
#   micro ripple = flux − macro trend   (Monte Carlo noise)
#   RI           = RMS(ripple in focal zone) / q_max × 100%
macro_trend  = gaussian_filter(flux_vis, sigma=3.0, mode='wrap')
micro_ripple = flux_vis - macro_trend
rms_noise    = np.sqrt(np.mean(micro_ripple[focal_mask]**2))
RI           = (rms_noise / q_max_vis) * 100.0

# Spatial gradient volatility (RMS of |∇q| in focal zone)
df_dtheta            = np.zeros_like(flux_vis)
df_dtheta[1:-1, :]   = (flux_vis[2:, :] - flux_vis[:-2, :]) / 2.0
df_dtheta[0,    :]   = (flux_vis[1,  :] - flux_vis[-1, :]) / 2.0  # periodic
df_dtheta[-1,   :]   = (flux_vis[0,  :] - flux_vis[-2, :]) / 2.0  # periodic
df_ds      = np.gradient(flux_vis, axis=1)
grad_mag   = np.sqrt(df_dtheta**2 + df_ds**2)
volatility = np.sqrt(np.mean(grad_mag[focal_mask]**2))

# Theoretical noise floor for FFTKDE (2D, Scott's bandwidth):
#   σ ∝ 1 / √(N · h_eff²)
noise_floor_pct = (1.0 / np.sqrt(N_total * h_bandwidth**2)) * 100.0

t_total = (t_load_end - t_load_start) + t_kde_total

print(f"\n{'='*62}")
print(f"  METHOD 3 (FFTKDE) — METRIC VERIFICATION SUMMARY")
print(f"{'='*62}")
print(f"  Rays per wavelength:                {rays_per_wavelength:,}")
print(f"  Total rays (N):                     {N_total:,}")
print(f"  Total absorbed power (P_MCRT):      {P_abs_code:.4f} W")
print(f"  Scott's h (normalised space):       {h_scott:.6f}")
print(f"  h_eff (geom. mean, physical):       {h_bandwidth*1000:.3f} mm")
print(f"  Theoretical noise floor 1/√(N·h²): {noise_floor_pct:.4f} %")
print(f"  Global Roughness Index (RI):        {RI:.4f} %")
print(f"  Spatial gradient volatility:        {volatility:.2f} W/m² per node")
print(f"  Max flux  q_max (200×100 grid):     {q_max_vis:.2f} W/m²")
print(f"  Mean flux q_avg (200×100 grid):     {q_avg_vis:.2f} W/m²")
print(f"  Data loading time:                  {t_load_end-t_load_start:.1f} s")
print(f"  FFTKDE fit + evaluate time:         {t_kde_total:.2f} s")
print(f"  Total computation time:             {t_total:.1f} s  "
      f"({t_total/60:.2f} min)")
print(f"{'='*62}\n")


# ================================================================
#  SECTION 7 — 3D CYLINDRICAL VISUALISATION
# ================================================================
print("[7] Generating 3D cylinder visualisation...")

theta_w = np.append(theta_vis, theta_vis[0] + 2*np.pi)
flux_w  = np.vstack([flux_vis, flux_vis[0:1, :]])
TW, SW  = np.meshgrid(theta_w, s_vis, indexing='ij')

# Invert the −π/2 angle shift applied during hit processing
X3D = center1 + R * np.cos(TW + np.pi/2)
Y3D = SW
Z3D = center2 + R * np.sin(TW + np.pi/2)

norm3d   = Normalize(vmin=flux_w.min(), vmax=flux_w.max())
colors3d = CMAP(norm3d(flux_w))

fig3d = plt.figure(figsize=(8, 6))
ax3d  = fig3d.add_subplot(111, projection='3d')
ax3d.plot_surface(X3D, Y3D, Z3D, facecolors=colors3d,
                  rstride=1, cstride=1, linewidth=0,
                  antialiased=False, shade=False)
sm3d = cm.ScalarMappable(norm=norm3d, cmap=CMAP)
sm3d.set_array(flux_w)
cb3d = fig3d.colorbar(sm3d, ax=ax3d, shrink=0.6, pad=0.05)
cb3d.ax.set_title("q''\n(W/m²)", fontsize=10, pad=8)
ax3d.set_xlabel("Diameter (m)",  labelpad=4,  fontsize=10)
ax3d.set_ylabel("Length (m)",    labelpad=4,  fontsize=10)
ax3d.set_zlabel("Height (m)",    labelpad=8,  fontsize=10)
ax3d.set_yticks(np.arange(0, receiver_length_m + 1, 1))
ax3d.view_init(elev=20, azim=-60)
ax3d.set_box_aspect((1, 3, 1))
plt.tight_layout()

path_3d = os.path.join(OUTPUT_DIR, "flux_receiver_3D_FFTKDE.jpg")
fig3d.savefig(path_3d, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {path_3d}")


# ================================================================
#  SECTION 8 — 2D CONTINUOUS CONTOUR MAP
# ================================================================
print("[8] Generating 2D contour map...")

plt.rcParams.update({
    "font.family": "serif",  "font.size": 10,
    "axes.labelsize": 11,    "axes.titlesize": 11,
    "xtick.labelsize": 10,   "ytick.labelsize": 10,
})

levels = np.linspace(0, np.ceil(q_max_vis / 5000) * 5000, 11)
fig2d, ax2d = plt.subplots(figsize=(8, 5), dpi=300)

cf2d = ax2d.contourf(s_vis, np.degrees(theta_vis), flux_vis,
                      levels=levels, cmap='jet')
cl2d = ax2d.contour( s_vis, np.degrees(theta_vis), flux_vis,
                      levels=levels, colors='black',
                      linewidths=0.4, alpha=0.6)
ax2d.clabel(cl2d, inline=True, fmt='%1.0f', fontsize=8, colors='black')

ax2d.set_xlabel("Receiver Longitudinal Axis, $s$ (m)",
                labelpad=8, fontweight='bold')
ax2d.set_ylabel("Receiver Circumferential Angle, $\\theta$ (deg)",
                labelpad=8, fontweight='bold')
ax2d.set_title(
    "2D Continuous Heat Flux ($q''$ in W/m²)\n"
    "Reconstructed via Fast Fourier Transform KDE (FFTKDE)\n"
    f"[N = {N_total:,} rays  |  RI = {RI:.4f}%  |  "
    f"h_eff = {h_bandwidth*1000:.2f} mm  |  time = {t_kde_total:.1f} s]",
    pad=12, fontweight='bold'
)
ax2d.set_xlim(0, receiver_length_m)
ax2d.set_ylim(0, 360)
ax2d.set_yticks([0, 90, 180, 270, 360])
ax2d.grid(True, linestyle=':', alpha=0.4, color='#cccccc')

cb2d = fig2d.colorbar(cf2d, ax=ax2d, pad=0.03, shrink=0.9)
cb2d.set_label("Absorbed Heat Flux, $q''$ (W/m²)",
               labelpad=10, fontweight='bold')
cb2d.set_ticks(levels)
plt.tight_layout()

path_2d = os.path.join(OUTPUT_DIR, "flux_contour_2D_FFTKDE.png")
fig2d.savefig(path_2d, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_2d}")


# ================================================================
#  SECTION 9 — SPATIAL ROUGHNESS MAP
# ================================================================
print("[9] Generating spatial roughness map...")

local_rough  = gaussian_filter(np.abs(micro_ripple), sigma=1.5, mode='wrap')
rough_map    = (local_rough / q_max_vis) * 100.0
rough_masked = np.where(focal_mask, rough_map, 0.0)
rough_masked = np.clip(rough_masked, 0.0, 5.0)

fig_n, ax_n = plt.subplots(figsize=(8, 5), dpi=300)
cn = ax_n.contourf(s_vis, np.degrees(theta_vis), rough_masked,
                   levels=np.linspace(0, 5.0, 11), cmap='viridis_r')
cb_n = fig_n.colorbar(cn, ax=ax_n, pad=0.03, shrink=0.9)
cb_n.set_label("Local Roughness Intensity (%)", labelpad=10, fontweight='bold')
ax_n.set_xlabel("Receiver Longitudinal Axis, $s$ (m)",
                labelpad=8, fontweight='bold')
ax_n.set_ylabel("Receiver Circumferential Angle, $\\theta$ (deg)",
                labelpad=8, fontweight='bold')
ax_n.set_title(
    f"Method 3 (FFTKDE) — Spatial Roughness Map\n"
    f"[Global RI = {RI:.4f}%  |  "
    f"h_eff = {h_bandwidth*1000:.2f} mm  |  N = {N_total:,}]",
    pad=12, fontweight='bold'
)
ax_n.set_xlim(0, receiver_length_m)
ax_n.set_ylim(0, 360)
ax_n.set_yticks([0, 90, 180, 270, 360])
ax_n.grid(True, linestyle=':', alpha=0.4, color='#cccccc')
plt.tight_layout()

path_noise = os.path.join(OUTPUT_DIR, "Method3_FFTKDE_Noise_Map.png")
fig_n.savefig(path_noise, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_noise}")


# ================================================================
#  SECTION 10 — CSV EXPORT AT ALL REQUESTED RESOLUTIONS
#
#  KEY ADVANTAGE OVER SCIPY KDE:
#  The FFTKDE is fitted ONCE. Every export resolution is obtained
#  by bilinear interpolation from the fine FFT grid — zero
#  additional KDE computation regardless of how many resolutions
#  are requested.
# ================================================================
print("\n[10] Exporting CSV files at all requested resolutions...")
print(f"  {'Resolution':>12}  {'Cells':>10}  "
      f"{'q_max (W/m²)':>14}  {'P_recv (W)':>12}  Filename")
print(f"  {'-'*70}")

def export_csv(N_th, N_s, interp, R, L, P_target, out_dir):
    """
    Evaluate interp_phys on a (N_th × N_s) cell-centred grid.
    Enforces exact energy conservation and writes standard 4-column CSV.
    Column order matches Method 1 (BB) and Method 2 (k-NN).
    """
    dtheta_e  = 2 * np.pi / N_th
    ds_e      = L / N_s
    cell_area = R * dtheta_e * ds_e

    theta_c = np.linspace(dtheta_e/2, 2*np.pi - dtheta_e/2, N_th)
    s_c     = np.linspace(ds_e/2,     L - ds_e/2,            N_s)
    TC, SC  = np.meshgrid(theta_c, s_c, indexing='ij')
    pts     = np.column_stack([TC.ravel(), SC.ravel()])

    flux = np.clip(interp(pts), 0.0, None)
    flux *= P_target / (flux.sum() * cell_area)   # exact energy conservation
    # ── Compute RI on this specific export grid ──────────────────
    from scipy.ndimage import gaussian_filter as gf
    flux_2d      = flux.reshape(N_th, N_s)
    macro        = gf(flux_2d, sigma=3.0, mode='wrap')
    ripple       = flux_2d - macro
    focal        = flux_2d > (0.10 * flux_2d.max())
    rms_ripple   = np.sqrt(np.mean(ripple[focal]**2)) if focal.any() else 0.0
    RI_export    = (rms_ripple / flux_2d.max() * 100.0) if flux_2d.max() > 0 else 0.0
    Noise_export = (np.sqrt(np.mean(ripple**2)) / flux_2d.mean() * 100.0) \
                   if flux_2d.mean() > 0 else 0.0

    # ─────────────────────────────────────────────────────────────

    df_out = pd.DataFrame({
        'CircumferentialAngle_deg' : np.round(np.degrees(TC.ravel()) % 360, 6),
        'LongitudinalAxis_m'       : np.round(SC.ravel(),  6),
        'CellArea_m2'              : np.round(cell_area,   8),
        'HeatFlux_W_m2'            : np.round(flux,        4),
    })
    fname = os.path.join(out_dir, f"Method3_FFTKDE_{N_th}x{N_s}.csv")
    df_out.to_csv(fname, index=False)

    q_mean_export = flux.sum() * cell_area / (2 * np.pi * R * L)
    print(f"  {N_th:>5}×{N_s:<6}  "
          f"max={flux.max():>10,.1f}  "
          f"mean={q_mean_export:>10,.1f}  "
          f"RI={RI_export:.3f}%  "
          f"Noise={Noise_export:.3f}%  "
          f"{os.path.basename(fname)}")


for (N_th, N_s) in EXPORT_RESOLUTIONS:
    export_csv(N_th, N_s, interp_phys, R,
               receiver_length_m, P_abs_code, OUTPUT_DIR)


# ================================================================
#  SECTION 11 — OPTIONAL SCIPY VALIDATION
#  Only run at N <= 5,000 rays/wavelength (scipy too slow for more)
# ================================================================
if RUN_SCIPY_VALIDATION:
    print("\n[11] Running scipy gaussian_kde validation comparison...")
    from scipy.stats import gaussian_kde as scipy_kde

    t_sp_start  = time.time()
    k_sp        = scipy_kde(np.vstack([theta_all, s_all]), weights=w_all)
    flux_sp_flat = k_sp.evaluate(np.vstack([TV.ravel(), SV.ravel()]))
    flux_sp     = np.clip(
        flux_sp_flat.reshape(N_VIS_THETA, N_VIS_S) * P_abs_code / R,
        0.0, None
    )
    flux_sp *= P_abs_code / (flux_sp * dA_vis).sum()
    t_sp_end    = time.time()

    abs_diff    = np.abs(flux_vis - flux_sp)
    max_rel     = (abs_diff.max()              / q_max_vis) * 100.0
    rms_rel     = (np.sqrt(np.mean(abs_diff**2)) / q_max_vis) * 100.0

    print(f"  scipy time:               {t_sp_end - t_sp_start:.1f} s")
    print(f"  FFTKDE time:              {t_kde_total:.2f} s")
    print(f"  Speed-up factor:          {(t_sp_end-t_sp_start)/t_kde_total:.1f}×")
    print(f"  Max relative difference:  {max_rel:.3f} %")
    print(f"  RMS relative difference:  {rms_rel:.3f} %")

    # Side-by-side validation figure
    fig_v, axv = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    vmax_kw = max(q_max_vis, flux_sp.max()) / 1000
    for ax, data, ttl in zip(
        axv,
        [flux_sp/1000, flux_vis/1000, abs_diff/1000],
        ["scipy KDE (reference)",
         "FFTKDE (this work)",
         f"| difference |  max={max_rel:.2f}%"]
    ):
        im = ax.imshow(data, aspect='auto', origin='lower',
                       cmap='jet', vmin=0, vmax=vmax_kw,
                       extent=[0, receiver_length_m, 0, 360])
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("s (m)"); ax.set_ylabel("θ (°)")
        plt.colorbar(im, ax=ax, label="kW/m²", shrink=0.85)

    plt.suptitle(
        f"FFTKDE vs scipy validation  [N={N_total:,} rays]  "
        f"Speed-up: {(t_sp_end-t_sp_start)/t_kde_total:.0f}×  "
        f"Max diff: {max_rel:.2f}%",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    pv = os.path.join(OUTPUT_DIR, "Method3_FFTKDE_validation.png")
    plt.savefig(pv, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pv}")


# ================================================================
#  SECTION 12 — WRITE METRICS SUMMARY FILE (for paper table)
# ================================================================
print("\n[12] Saving metrics summary to text file...")

metrics_path = os.path.join(OUTPUT_DIR, "Method3_FFTKDE_metrics.txt")
with open(metrics_path, 'w') as f:
    f.write("METHOD 3 — FFTKDE METRIC VERIFICATION SUMMARY\n")
    f.write("=" * 55 + "\n")
    f.write(f"Rays per wavelength:             {rays_per_wavelength:,}\n")
    f.write(f"Total rays (N):                  {N_total:,}\n")
    f.write(f"Total absorbed power:            {P_abs_code:.4f} W\n")
    f.write(f"Scott's h (normalised space):    {h_scott:.6f}\n")
    f.write(f"h_x effective (physical):        {h_eff_x*1000:.3f} mm\n")
    f.write(f"h_y effective (physical):        {h_eff_y*1000:.2f} mm\n")
    f.write(f"h_eff geom-mean (physical):      {h_bandwidth*1000:.3f} mm\n")
    f.write(f"Theoretical noise 1/sqrt(N*h²):  {noise_floor_pct:.4f} %\n")
    f.write(f"Global RI (roughness index):     {RI:.4f} %\n")
    f.write(f"Spatial gradient volatility:     {volatility:.2f} W/m² per node\n")
    f.write(f"Max flux (200×100 vis grid):     {q_max_vis:.2f} W/m²\n")
    f.write(f"Mean flux (200×100 vis grid):    {q_avg_vis:.2f} W/m²\n")
    f.write(f"Data loading time:               {t_load_end-t_load_start:.1f} s\n")
    f.write(f"FFTKDE fit + evaluate time:      {t_kde_total:.2f} s\n")
    f.write(f"Total computation time:          {t_total:.1f} s\n")
    f.write("=" * 55 + "\n")
    f.write("\nCSV files exported:\n")
    for (N_th, N_s) in EXPORT_RESOLUTIONS:
        f.write(f"  Method3_FFTKDE_{N_th}x{N_s}.csv  ({N_th*N_s:,} cells)\n")

print(f"  Saved: {metrics_path}")
print(f"\n{'='*62}")
print(f"  METHOD 3 (FFTKDE) — COMPLETE")
print(f"  Total time: {t_total:.1f} s  ({t_total/60:.2f} min)")
print(f"  All outputs → {OUTPUT_DIR}")
print(f"{'='*62}\n")
