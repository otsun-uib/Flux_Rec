# -*- coding: utf-8 -*-
"""
BB_Flux_Rec.py
Bin-based (BB) spectral MCRT flux reconstruction for cylindrical receivers.

Part of: A Grid-Independent Flux Mapping Approach for Cylindrical Solar Receivers

Authors: Sainath Waghmare et al., UIB, 2026. 
"""
import time 

start_time = time.perf_counter()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ============================================================
# 1) EDIT THESE PARAMETERS
# ============================================================

ASTM_FILE = "data/ASTMG173-direct.xlsx"   # path to ASTM G173-03 spectrum
HITS_FILE = "data/ray_hits.txt"            # path to OTSun ray-hit output file

# Receiver geometry (EDITABLE)
receiver_outer_radius_mm = 35.0     # outer radius [mm]  (your case: 35 mm)
receiver_length_m = 4.0            # receiver length [m] (your case: 10 m = 10000 mm)

# Collector geometry for ABSOLUTE POWER (EDITABLE)
aperture_width_m = 5.76             # [m]
collector_length_m = 4.0           # [m] (often same as receiver length)

# MCRT settings (EDITABLE)
rays_per_wavelength = 500          # rays emitted per wavelength (OTSun setting)
dlambda_nm = 20.0                   # wavelength step used in simulation [nm]

wl_min_nm = 290.0
wl_max_nm = 3990.0

# Coordinates in OTSun file (EDITABLE)
axis = "y"                          # receiver axis: "x", "y", or "z" (your case: "y")
units = "mm"                        # OTSun coordinates are usually in "mm" for LS-3

# Plot / binning (EDITABLE)
N_theta = 40                    # bins around tube
N_s = 600                           # bins along tube
chunksize = 1_000_000               # for big files

# Tube center in cross-section plane (OPTIONAL)
# If you know it, set it for perfect geometry; else keep None (auto-estimate from hit points)
center1 = None
center2 = None

# Colormap (to match your sample, use "jet")
CMAP = cm.jet
# ============================================================


# ============================================================
# 2) LOAD ASTM SPECTRUM
# ============================================================
astm = pd.read_excel(ASTM_FILE, engine="openpyxl").dropna()
wl_astm = astm.iloc[:, 0].to_numpy(float)     # wavelength (nm)
E_astm  = astm.iloc[:, 1].to_numpy(float)     # spectral irradiance (W/m²/nm) [1](https://seos-project.eu/earthspectra/documents/solar-spectra/nrel_ASTM_G-173/Solar%20Spectral%20Irradiance%20%20Air%20Mass%201.5.htm)

# --------- NEW (only change): cumulative integral of ASTM ---------
# cumE(λ) = ∫ E_astm dλ  -> units W/m²
cumE = np.zeros_like(wl_astm)
cumE[1:] = np.cumsum(0.5 * (E_astm[1:] + E_astm[:-1]) * (wl_astm[1:] - wl_astm[:-1]))

# Aperture area for absolute power
A_ap = aperture_width_m * collector_length_m  # m²

# Receiver radius in meters
R = receiver_outer_radius_mm * 1e-3

# Unit conversion for hit coordinates
scale = 1e-3 if units.lower() == "mm" else 1.0

# Axis index
axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
cross = [i for i in (0, 1, 2) if i != axis_idx]  # cross-section coordinate indices


# ============================================================
# 3) ESTIMATE TUBE CENTER (IF NOT PROVIDED)
#    Robust: use normals and choose correct sign automatically
# ============================================================
if center1 is None or center2 is None:

    # local column names (avoids NameError: cols not defined)
    cols_local = ["energy_ray","x","y","z","px","py","pz","nx","ny","nz","wl_nm"]

    sample = pd.read_csv(
        HITS_FILE,
        delim_whitespace=True,
        comment="#",
        names=cols_local,
        header=None,
        nrows=200000
    )

    P = sample[["x", "y", "z"]].to_numpy(float) * scale      # points (m)
    N = sample[["nx", "ny", "nz"]].to_numpy(float)           # normals (unitless)  [1](https://cpp.openfoam.org/v13/classFoam_1_1PatchPostProcessing.html)

    # cross-section indices depending on axis
    if axis.lower() == "y":       # tube along Y -> cross plane X-Z
        i1, i2 = 0, 2
    elif axis.lower() == "x":     # tube along X -> cross plane Y-Z
        i1, i2 = 1, 2
    else:                         # tube along Z -> cross plane X-Y
        i1, i2 = 0, 1

    def fit_center(Ccand):
        c1 = Ccand[:, i1].mean()
        c2 = Ccand[:, i2].mean()
        r = np.sqrt((P[:, i1] - c1)**2 + (P[:, i2] - c2)**2)
        err = np.sqrt(np.mean((r - R)**2))
        return err, c1, c2

    # Try both normal directions (normals may be inward or outward)
    err_minus, c1_minus, c2_minus = fit_center(P - R * N)
    err_plus,  c1_plus,  c2_plus  = fit_center(P + R * N)

    if err_minus <= err_plus:
        center1, center2 = c1_minus, c2_minus
    else:
        center1, center2 = c1_plus, c2_plus


# ============================================================
# 4) SET UP BINS (theta around tube, s along tube)
# ============================================================
theta_edges = np.linspace(0, 2*np.pi, N_theta + 1)
s_edges     = np.linspace(0, receiver_length_m, N_s + 1)

power_map = np.zeros((N_theta, N_s), float)  # stores POWER (W) accumulated in each surface bin


# ============================================================
# 5) STREAM THE OTSUN HITS FILE AND ACCUMULATE POWER
#    File columns (11): energy_ray, x,y,z, px,py,pz, nx,ny,nz, wavelength_nm
# ============================================================
cols = ["energy_ray","x","y","z","px","py","pz","nx","ny","nz","wl_nm"]

power_map = np.zeros((N_theta, N_s), float)
P_abs_code = 0.0

reader = pd.read_csv(HITS_FILE, delim_whitespace=True, comment="#",
                     names=cols, header=None, chunksize=chunksize)

for chunk in reader:
    P = chunk[["x", "y", "z"]].to_numpy(float) * scale
    s = P[:, axis_idx]

    u = P[:, cross[0]] - center1
    v = P[:, cross[1]] - center2
    theta = np.arctan2(v, u) - (np.pi / 2.0)
    theta[theta < 0] += 2*np.pi

    wl = chunk["wl_nm"].to_numpy(float)  # wavelength (nm) from OTSun output [2](https://www.nlr.gov/grid/solar-resource/spectra)

    # --------- NEW (only change): band-integrated irradiance ---------
    # E_band(wl) = ∫_{wl-Δλ/2}^{wl+Δλ/2} E(λ) dλ   [W/m²]

        
    a = np.maximum(wl - 0.5 * dlambda_nm, wl_min_nm)
    b = np.minimum(wl + 0.5 * dlambda_nm, wl_max_nm)
    
    Fa = np.interp(a, wl_astm, cumE, left=cumE[0], right=cumE[-1])
    Fb = np.interp(b, wl_astm, cumE, left=cumE[0], right=cumE[-1])
    E_band = Fb - Fa  # W/m² (integrated over the wavelength bin)

    # Power per emitted ray at that wavelength bin (absolute)
    P_ray = (E_band * A_ap) / rays_per_wavelength

    w = chunk["energy_ray"].to_numpy(float) * P_ray

    P_abs_code += w.sum()

    H, _, _ = np.histogram2d(theta, s, bins=[theta_edges, s_edges], weights=w)
    power_map += H

print("Absorbed power from rays (sum of w) [W]:", P_abs_code)
print("Absorbed power from binned map (power_map.sum) [W]:", power_map.sum())


# ============================================================
# 6) CONVERT POWER MAP -> FLUX MAP (W/m²) ON RECEIVER SURFACE
# ============================================================
dtheta = 2*np.pi / N_theta
ds = receiver_length_m / N_s
dA = R * dtheta * ds  # surface area of one (theta,s) bin on cylinder

flux_map = power_map / dA

print("Max flux (W/m²):", flux_map.max())
print("Mean flux (W/m²):", flux_map.mean())


# ============================================================
# 7) 3D CYLINDER PLOT (CARTESIAN) + SEAM FIX
# ============================================================
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])   # (N_theta,)
s_centers = s_edges     # (N_s,)

Theta, S = np.meshgrid(theta_centers, s_centers, indexing="ij")  # (N_theta, N_s)

# ---- SEAM FIX: wrap theta=0 row to theta=2pi ----
flux_wrap  = np.vstack([flux_map, flux_map[0:1, :]])
Theta_wrap = np.vstack([Theta, Theta[0:1, :] + 2*np.pi])
S_wrap     = np.vstack([S,     S[0:1, :]])

# Invert 3D coordinates wrapping rotation to match the current selection (+pi/2)
X = center1 + R * np.cos(Theta_wrap + (np.pi / 2.0))
Y = S_wrap
if axis.lower() == "y":
    Z = center2 + R * np.sin(Theta_wrap + (np.pi / 2.0))
elif axis.lower() == "x":
    X = S_wrap
    Y = center1 + R * np.cos(Theta_wrap + (np.pi / 2.0))
    Z = center2 + R * np.sin(Theta_wrap + (np.pi / 2.0))
else:
    Z = S_wrap

# ---- Colors from flux ----
norm = Normalize(vmin=flux_wrap.min(), vmax=flux_wrap.max())
colors = CMAP(norm(flux_wrap))

fig = plt.figure(figsize=(8, 4.5))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    X, Y, Z,
    facecolors=colors,
    rstride=1, cstride=1,
    linewidth=0,
    antialiased=False,
    shade=False
)

#colour bar
m = cm.ScalarMappable(norm=norm, cmap=CMAP)
m.set_array(flux_wrap)

cb = fig.colorbar(m, ax=ax, shrink=0.65, pad=0.03)  # smaller pad => closer
cb.ax.set_title("$q$ (W/m²)", fontsize=12, pad=10, ha="left")  # vertical label, spacing
cb.ax.tick_params(labelsize=10)

ax.set_xlim(np.min(X), np.max(X))
ax.set_ylim(np.min(Y), np.max(Y))
ax.set_zlim(np.min(Z), np.max(Z))

ax.set_yticks(np.arange(0, receiver_length_m + 1, 1))

ax.set_xlabel("Receiver diameter, D (m)", labelpad=4, fontsize=11)
ax.set_ylabel("Receiver length, L (m)", labelpad=4, fontsize=11)
ax.set_zlabel("Receiver height, H (m)", labelpad=10, fontsize=11)

#ax.set_title("Flux on Receiver Tube (3D Cartesian Cylinder)")
ax.view_init(elev=20, azim=-60)


for lbl in ax.get_xticklabels():
    lbl.set_rotation(30)
    lbl.set_horizontalalignment('right')  # helps avoid overlap
    lbl.set_verticalalignment('center')

# --- bring tick numbers closer to the axis lines ---
ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=-2)
ax.tick_params(axis='z', pad=4)

plt.tight_layout()
ax.set_box_aspect((1, 3, 1))  # X:Y:Z visual scaling (make Y look 4x longer)
plt.show()

fig.subplots_adjust(top=1.15, bottom=-0.15, left=-0.05, right=0.85)

fig.savefig("flux_receiver_3D.jpg", dpi=300, bbox_inches="tight", facecolor="white")

# ============================================================
# NEW === 8) TRUE LENGTH-INDEPENDENT 2D CONTOUR MAPPING === NEW
# ============================================================
print("\nGenerating true length-independent 2D checkerboard map...")

# 1. Update font and presentation styling for publication-ready layout
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# 2. Initialize the 2D Canvas
fig2d, ax2d = plt.subplots(figsize=(7.5, 4.8), dpi=300)

# Establish clean colorbar intervals matching your exact color layout bounds
max_flux_val = flux_map.max()
levels = np.linspace(0, np.ceil(max_flux_val / 5000) * 5000, 11)

# Generate raw checkerboard grid blocks that fill the entire frame
contour_filled = ax2d.pcolormesh(s_edges, np.degrees(theta_edges), flux_map, cmap=CMAP, vmin=0, vmax=levels[-1])

# Formatted Axis Labels and Parameters
ax2d.set_xlabel("Receiver Longitudinal Axis, $s$ (m)", labelpad=8)
ax2d.set_ylabel("Receiver Circumferential Angle, $\\theta$ (deg)", labelpad=8)
#ax2d.set_title("Method 1: 2D Discrete Heat Flux Topography ($q''$ in W/m²)\nRepresented via Raw Numerical Bins", pad=15, fontweight="bold")

# Force the plot to lock exactly onto the full physical receiver limits
ax2d.set_xlim(0, receiver_length_m)
ax2d.set_ylim(0, 360)
ax2d.set_xticks(np.arange(0, receiver_length_m + 0.1, 0.5))
ax2d.set_yticks([0, 90, 180, 270, 360])
ax2d.grid(True, linestyle=":", alpha=0.4, color="#cccccc")

# Map the colorbar parameters
import matplotlib.colors as mcolors
import matplotlib.cm as cm
norm_match = mcolors.Normalize(vmin=flux_wrap.min(), vmax=flux_wrap.max())
m2d = cm.ScalarMappable(norm=norm_match, cmap="jet")
m2d.set_array(flux_map)

# 2. Draw the colorbar using the 3D dimensions (shrink=0.6) and formatting
cb2d = fig2d.colorbar(m2d, ax=ax2d, shrink=0.9, pad=0.05)
cb2d.ax.set_title("$q$ (W/m²)", fontsize=11, pad=10, ha="left")

plt.tight_layout()
fig2d.savefig("flux_contour_bands_2D_Perfect_Edges.png", dpi=300, bbox_inches="tight")
print("Saved complete independent 2D checkerboard plot as: 'flux_contour_bands_2D_Perfect_Edges.png'")
plt.show()

# ============================================================
# EXTRA: MCRT NOISE CALCULATION (POISSON STATISTICAL ERROR)
# ============================================================
print("\n--- Method 1 MCRT Noise Metrics ---")

# 1. Local Relative Noise Map (%)
noise_map_m1 = np.zeros_like(power_map, dtype=float)
mask = power_map > 0
noise_map_m1[mask] = (1.0 / np.sqrt(power_map[mask])) * 100 

# 2. Global Metric
focal_zone_mask = flux_map > (0.10 * flux_map.max())

# Calculate physical ray counts inside the zone
total_power_in_zone = power_map[focal_zone_mask].sum()
mean_ray_power = np.mean(P_ray)
total_rays_in_zone = total_power_in_zone / mean_ray_power
num_bins_in_zone = np.sum(focal_zone_mask)

average_focal_rays = float(total_rays_in_zone / num_bins_in_zone)
representative_noise_m1 = (1.0 / np.sqrt(average_focal_rays)) * 100
print(f"Representative Focal Line Noise Level: {representative_noise_m1:.2f}%")

# 3. Use RectBivariateSpline instead of the deprecated interp2d
from scipy.interpolate import RectBivariateSpline

theta_centers_native = 0.5 * (theta_edges[:-1] + theta_edges[1:])
s_centers_native     = 0.5 * (s_edges[:-1] + s_edges[1:])

# Safe check for single-bin edge cases
if N_s == 1:
    s_centers_native = np.array([0.0, receiver_length_m])
    noise_reconstructed = np.hstack([noise_map_m1, noise_map_m1])
else:
    noise_reconstructed = noise_map_m1

# Set up the matching fine grid array sizes (250 elements to match S_fine / Theta_fine)
theta_fine_deg = np.linspace(0, 360, 250)
s_fine_m       = np.linspace(0, receiver_length_m, 250)
S_fine_grid, Theta_fine_grid = np.meshgrid(s_fine_m, theta_fine_deg)

# Generate interpolation object
spline_noise = RectBivariateSpline(np.degrees(theta_centers_native), s_centers_native, noise_reconstructed, kx=1, ky=1)
noise_plot_grid = spline_noise(theta_fine_deg, s_fine_m)

# ==============================================================================
# ADDED === EXTRA: UNIFIED SURFACE ROUGHNESS INDEX (RI) ANALYSIS === ADDED
# ==============================================================================
print("\n--- Method 1 Unified Surface Roughness Metrics ---")
from scipy.ndimage import gaussian_filter

# 1. Retain the same 10% threshold to isolate the active illumination envelope directly from raw bins
focal_mask_ri = flux_map > (0.10 * flux_map.max())

# 2. Extract physical macro trend using a periodic wrap across the 0-360 boundaries on raw bins
# (Sigma is scaled down to 1.5 to perfectly match the coarser 20x20 grid structure)
macro_trend_m1 = gaussian_filter(flux_map, sigma=1.5, mode='wrap')

# 3. Isolate high-frequency micro-stochastic ripples (pure numerical cell-to-cell block variations)
micro_ripples_m1 = flux_map - macro_trend_m1

# 4. Compute the global Surface Roughness Index (RI) percentage inside the active zone
rms_noise_m1 = np.sqrt(np.mean(micro_ripples_m1[focal_mask_ri]**2))
surface_roughness_index_m1 = (rms_noise_m1 / flux_map.max()) * 100
print(f"Global Method 1 Surface Roughness Index (RI): {surface_roughness_index_m1:.4f}%")

# 5. Local Spatial Roughness Map Masked to Active Flux Area
local_roughness_m1 = gaussian_filter(np.abs(micro_ripples_m1), sigma=0.5, mode='wrap')
m1_ri_map = (local_roughness_m1 / flux_map.max()) * 100

# Apply mask to eliminate zero-flux background clutter
m1_ri_map_masked = np.zeros_like(m1_ri_map)
m1_ri_map_masked[focal_mask_ri] = m1_ri_map[focal_mask_ri]

# Set a clean visual ceiling for plotting comparison (Matches your 5.0% ceiling)
max_visible_ri_limit = 5.0
m1_ri_map_masked = np.clip(m1_ri_map_masked, 0, max_visible_ri_limit)


# ==============================================================================
# ADDED === PROVISION: EXPORT STRUCTURED CFD FLUX BOUNDARY CONDITION (BB) ===
# ==============================================================================
# Compute midpoints of the spatial bins for coordinate generation
theta_centers_export = 0.5 * (theta_edges[:-1] + theta_edges[1:])
s_centers_export = 0.5 * (s_edges[:-1] + s_edges[1:])

# Generate 2D meshgrids matching the (N_theta, N_s) indexing of the flux_map
Theta_export_mesh, S_export_mesh = np.meshgrid(theta_centers_export, s_centers_export, indexing="ij")

# Flatten the matrices to create 1D coordinate and value columns
flat_theta_deg = np.degrees(Theta_export_mesh).flatten()
flat_s_m = S_export_mesh.flatten()
flat_flux = flux_map.flatten()
flat_area = np.full_like(flat_flux, dA)  # Uses the uniform cell area dA from Section 6

# Assemble structured text dataframe
cfd_export_data = np.column_stack((flat_theta_deg, flat_s_m, flat_area, flat_flux))
    
# Write file out with structured header tracking coordinates
csv_filename = f"BB_{N_theta}x{N_s}_{rays_per_wavelength}.csv"    #Put directory to save results. 
np.savetxt(
        csv_filename, 
        cfd_export_data, 
        delimiter=",", 
        fmt=["%.4f", "%.4f", "%.8f", "%.2f"],
        header="CircumferentialAngle_deg,LongitudinalAxis_m,CellArea_m2,HeatFlux_W_m2",
        comments=""
    )
print(f" -> SUCCESS: Exported clean boundary condition data file to: {csv_filename}")

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print(f" Elapsed time: {elapsed_time} seconds")