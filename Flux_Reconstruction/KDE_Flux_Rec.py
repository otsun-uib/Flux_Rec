# -*- coding: utf-8 -*-
"""
KDE_Flux_Rec.py
Kernel Density Estimation (KDE) spectral MCRT flux reconstruction for cylindrical receivers.

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
from scipy.stats import gaussian_kde

# ============================================================
# 1) EDIT THESE PARAMETERS (Matches Your Exact Setup)
# ============================================================

ASTM_FILE = "data/ASTMG173-direct.xlsx"   # path to ASTM G173-03 spectrum
HITS_FILE = "data/ray_hits.txt"            # path to OTSun ray-hit output file

receiver_outer_radius_mm = 35.0     # outer radius [mm]
receiver_length_m = 4.0            # receiver length [m]
aperture_width_m = 5.76             # [m]
collector_length_m = 4.0           # [m]

rays_per_wavelength = 500        # OTSun setting
dlambda_nm = 20.0                   # wavelength step [nm]
wl_min_nm, wl_max_nm = 290.0, 3990.0

# enter grid size (EDITABLE)
N_eval_theta = 20
N_eval_s = 20

axis = "y"                          # receiver axis: "x", "y", or "z"
units = "mm"                        # OTSun coordinates unit
chunksize = 1_000_000               # RAM safety cushion

center1 = None                      # Auto-calculated below
center2 = None
CMAP = cm.jet

# ============================================================
# 2) LOAD ASTM SPECTRUM & PRE-CALCULATE DIRECT RADIATION BOUNDS
# ============================================================
astm = pd.read_excel(ASTM_FILE, engine="openpyxl").dropna()
wl_astm = astm.iloc[:, 0].to_numpy(float)
E_astm  = astm.iloc[:, 1].to_numpy(float)

cumE = np.zeros_like(wl_astm)
cumE[1:] = np.cumsum(0.5 * (E_astm[1:] + E_astm[:-1]) * (wl_astm[1:] - wl_astm[:-1]))

A_ap = aperture_width_m * collector_length_m
R = receiver_outer_radius_mm * 1e-3
scale = 1e-3 if units.lower() == "mm" else 1.0

axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
cross = [i for i in (0, 1, 2) if i != axis_idx]

# ============================================================
# 3) AUTOMATED TUBE CENTER ESTIMATION
# ============================================================
if center1 is None or center2 is None:
    cols_local = ["energy_ray","x","y","z","px","py","pz","nx","ny","nz","wl_nm"]
    sample = pd.read_csv(HITS_FILE, delim_whitespace=True, comment="#",
                         names=cols_local, header=None, nrows=200000)
    P = sample[["x", "y", "z"]].to_numpy(float) * scale
    N = sample[["nx", "ny", "nz"]].to_numpy(float)
    
    i1, i2 = cross[0], cross[1]
    def fit_center(Ccand):
        c1 = Ccand[:, i1].mean()
        c2 = Ccand[:, i2].mean()
        r = np.sqrt((P[:, i1] - c1)**2 + (P[:, i2] - c2)**2)
        return np.sqrt(np.mean((r - R)**2)), c1, c2

    err_minus, c1_minus, c2_minus = fit_center(P - R * N)
    err_plus,  c1_plus,  c2_plus  = fit_center(P + R * N)
    center1, center2 = (c1_minus, c2_minus) if err_minus <= err_plus else (c1_plus, c2_plus)

# ============================================================
# 4) STREAM HITS FILE & COLLECT 2D CURVILINEAR COORDINATES
# ============================================================
cols = ["energy_ray","x","y","z","px","py","pz","nx","ny","nz","wl_nm"]
P_abs_code = 0.0

# Pre-allocate temporary python lists to safely collect coordinates across chunks
theta_collected = []
s_collected = []
weights_collected = []

print("Streaming OTSun file and converting coordinates into 2D manifold space...")
reader = pd.read_csv(HITS_FILE, delim_whitespace=True, comment="#",
                     names=cols, header=None, chunksize=chunksize)

for chunk in reader:
    P = chunk[["x", "y", "z"]].to_numpy(float) * scale
    s = P[:, axis_idx]
    
    u = P[:, cross[0]] - center1
    v = P[:, cross[1]] - center2
    theta = np.arctan2(v, u) - (np.pi / 2.0)
    theta[theta < 0] += 2*np.pi
    
    wl = chunk["wl_nm"].to_numpy(float)
    
#    wl_min_nm, wl_max_nm = 290.0, 3990.0
    a = np.maximum(wl - 0.5 * dlambda_nm, wl_min_nm)
    b = np.minimum(wl + 0.5 * dlambda_nm, wl_max_nm)
    
    Fa = np.interp(a, wl_astm, cumE, left=cumE[0], right=cumE[-1])
    Fb = np.interp(b, wl_astm, cumE, left=cumE[0], right=cumE[-1])
    E_band = Fb - Fa
    
    P_ray = (E_band * A_ap) / rays_per_wavelength
    w = chunk["energy_ray"].to_numpy(float) * P_ray
    P_abs_code += w.sum()
    
    # Store information for KDE fitting
    theta_collected.append(theta)
    s_collected.append(s)
    weights_collected.append(w)

# Flatten memory chunks into continuous 1D arrays
theta_all = np.concatenate(theta_collected)
s_all = np.concatenate(s_collected)
w_all = np.concatenate(weights_collected)

print(f"Total Absorbed Power from streamed ray datasets: {P_abs_code:.4f} W")

# ============================================================
# 5) METHOD 3 EXECUTION: SCIPY COMPARED CONTINUUM (ALL RAYS)
# ============================================================
print("\nProcessing complete ray population...")
theta_train = np.concatenate(theta_collected)
s_train = np.concatenate(s_collected)
w_train = np.concatenate(weights_collected)
N_rays_total = len(theta_train)

print(f"Fitting continuous 2D Kernel Density Estimation function field for {N_rays_total} rays...")
kernel_data = np.vstack([theta_train, s_train])

# Initialize SciPy's internal optimized estimator
kde_estimator = gaussian_kde(kernel_data, weights=w_train)

# ============================================================
# 6) HIGH-SPEED BLOCKED MULTI-DIMENSIONAL 2D EVALUATION
# ============================================================
print("Executing high-speed blocked 2D continuum evaluation (No Separable Assumptions)...")



theta_axis = np.linspace(0, 2*np.pi, N_eval_theta)
s_axis = np.linspace(0, receiver_length_m, N_eval_s)
Theta_grid, S_grid = np.meshgrid(theta_axis, s_axis, indexing="ij")

T_flat = Theta_grid.ravel()
S_flat = S_grid.ravel()
N_eval_total = len(T_flat)

# Define an optimal evaluation block size to maximize CPU cache performance
block_size = 2000
density_total = np.zeros(N_eval_total)

for idx in range(0, N_eval_total, block_size):
    end_idx = min(idx + block_size, N_eval_total)
    T_sub = T_flat[idx:end_idx]
    S_sub = S_flat[idx:end_idx]
    
    # Core coordinate matrices
    eval_real = np.vstack([T_sub, S_sub])
    eval_left = np.vstack([T_sub, -S_sub])
    eval_right = np.vstack([T_sub, 2 * receiver_length_m - S_sub])
    
    # Compute full 2D multi-variable interactions in fast blocks
    density_total[idx:end_idx] = (
        kde_estimator.evaluate(eval_real) + 
        kde_estimator.evaluate(eval_left) + 
        kde_estimator.evaluate(eval_right)
    )

raw_density_field = density_total.reshape(N_eval_theta, N_eval_s)

# Transform raw probability metrics directly into physical absolute heat flux [W/m²]
flux_map_kde = (raw_density_field * P_abs_code) / R

# Verify Energy Conservation via Integration
# ============================================================
# VERIFY ENERGY CONSERVATION VIA PROPER TRAPEZOIDAL INTEGRATION
# ============================================================
# Instead of an unweighted np.sum, use the trapezoidal rule along both axes
# to correctly scale the boundary edges and periodic seams.
from scipy.integrate import trapezoid

# Integrate along the s axis (axis=1) first, then along the theta axis (axis=0)
integral_s = trapezoid(flux_map_kde, x=s_axis, axis=1)
P_reconstructed_kde = trapezoid(integral_s, x=theta_axis, axis=0) * R

conservation_error_internal = ((P_reconstructed_kde - P_abs_code) / P_abs_code) * 100
# ============================================================

# Estimating the Global Average Heat Flux
A_tube_physical = 2 * np.pi * R * receiver_length_m
average_flux_kde = P_abs_code / A_tube_physical

print("\n=== VERIFIED BLOCKED 2D METHOD 3 RESULTS ===")
print(f"Processed Ray Count:                           {N_rays_total} rays")
print(f"Baseline Code Input Power (P_abs_code):       {P_abs_code:.4f} W")
print(f"Reconstructed Continuous Integral Power:       {P_reconstructed_kde:.4f} W")
print(f"Max Continuous Flux Reconstructed:             {flux_map_kde.max():.2f} W/m²")
print(f"Mean Continuous Flux Reconstructed:            {average_flux_kde:.2f} W/m²")


# ============================================================
# 7) CONTINUOUS 3D PLOTTING (Seam-Wrapped for Separable Method)
# ============================================================
print("Constructing 3D geometric surface mesh for visualization...")

# Reconstruct the coordinate grids from the 1D evaluation axes
Theta_grid, S_grid = np.meshgrid(theta_axis, s_axis, indexing="ij")

# ---- SEAM FIX: wrap theta=0 row to theta=2pi ----
flux_wrap  = np.vstack([flux_map_kde, flux_map_kde[0:1, :]])
Theta_wrap = np.vstack([Theta_grid, Theta_grid[0:1, :] + 2*np.pi])
S_wrap     = np.vstack([S_grid, S_grid[0:1, :]])

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

# ---- Colors from continuous flux field ----
norm = Normalize(vmin=flux_wrap.min(), vmax=flux_wrap.max())
colors = CMAP(norm(flux_wrap))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    X, Y, Z,
    facecolors=colors,
    rstride=1, cstride=1,
    linewidth=0,
    antialiased=False,
    shade=False
)

# Colorbar integration
m = cm.ScalarMappable(norm=norm, cmap=CMAP)
m.set_array(flux_wrap)

cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.05)
cb.ax.set_title("Flux (W/m²)", fontsize=11, pad=10)
cb.ax.tick_params(labelsize=10)

ax.set_xlim(np.min(X), np.max(X))
ax.set_ylim(np.min(Y), np.max(Y))
ax.set_zlim(np.min(Z), np.max(Z))

ax.set_yticks(np.arange(0, receiver_length_m + 1, 1))

ax.set_xlabel("Receiver diameter (m)", labelpad=4, fontsize=11)
ax.set_ylabel("Receiver length (m)", labelpad=4, fontsize=11)
ax.set_zlabel("Receiver height (m)", labelpad=10, fontsize=11)

ax.view_init(elev=20, azim=-60)

for lbl in ax.get_xticklabels():
    lbl.set_rotation(30)
    lbl.set_horizontalalignment('right')
    lbl.set_verticalalignment('center')

ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=-2)
ax.tick_params(axis='z', pad=4)

plt.tight_layout()
ax.set_box_aspect((1, 3, 1))
plt.show()

# Export publication-quality image file
fig.savefig("flux_receiver_3D_KDE_fast.jpg", dpi=600, bbox_inches="tight", facecolor="white")
print("Saved high-resolution flux_receiver_3D_KDE_fast.jpg")

# ============================================================
# 8) EXPORT CONTINUOUS FLUX MAP FOR OPENFOAM MAPPING
# ============================================================
data = np.column_stack([
    np.degrees(T_flat),             # Centered theta (deg) [0 to 360]
    S_flat,                         # s (m)
    flux_map_kde.ravel()            # q'' (W/m^2)
])

# ==============================================================================
# FIXED: EXPORTING CLEAN CSV WITHOUT COORDINATE COLLAPSE
# ==============================================================================
dtheta_csv = 2 * np.pi / N_eval_theta
ds_csv     = receiver_length_m / N_eval_s
cell_area  = R * dtheta_csv * ds_csv  

df_kde_out = pd.DataFrame({
    # REMOVED '% 360' so 360 degrees stays distinct from 0 degrees
    'CircumferentialAngle_deg': np.degrees(T_flat), 
    'LongitudinalAxis_m'      : S_flat,
    'CellArea_m2'             : cell_area,
    'HeatFlux_W_m2'           : flux_map_kde.ravel()
})

df_kde_out.to_csv(f"KDE_{N_eval_theta}x{N_eval_s}_{rays_per_wavelength}.csv", index=False)
print(" -> CSV successfully generated with unique spatial indices.")
# ==============================================================================

# ============================================================
# 9) 2D CONTINUOUS ISOFLUX CONTOUR BAND MAPPING (UNSMOOTHED)
# ============================================================
print("Generating 2D unsmoothed checkerboard map...")

if 'flux_map_kde' in locals() and 'Theta_grid' in locals() and 'S_grid' in locals():
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    max_flux = flux_map_kde.max()
    levels = np.linspace(0, np.ceil(max_flux / 5000) * 5000, 11) 

    # 2. Plot the raw, block-wise evaluated KDE field using 'nearest' for identical grid shapes
    contour_filled = ax.pcolormesh(
        S_grid, np.degrees(Theta_grid), flux_map_kde,
        cmap="jet",
        vmin=0,
        vmax=levels[-1],
        shading='nearest'  # FIXED: Handles cell-centered grids of matching dimensions
    )

    # NOTE: The continuous ax.contour and ax.clabel overlays have been removed 
    # to maintain a pure, un-interpolated discrete block layout.

    # Format Axes to match the cylindrical receiver unrolled geometry
    ax.set_xlabel("Receiver Longitudinal Axis, $s$ (m)", labelpad=8, fontweight="bold")
    ax.set_ylabel("Receiver Circumferential Angle, $\\theta$ (deg)", labelpad=8, fontweight="bold")
    ax.set_title("2D Discrete Heat Flux Topography ($q''$ in W/m²)\nEvaluated via Kernel Density Estimation Node Points", pad=15, fontweight="bold")

    # Scale ticks to map out the full surface area of the receiver tube
    ax.set_xlim(0, receiver_length_m)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.grid(True, linestyle=":", alpha=0.4, color="#cccccc")

    # Integrate a colorbar to quantify the color blocks
    cb = fig.colorbar(contour_filled, ax=ax, pad=0.03, shrink=0.9)
    cb.set_label("Absorbed Heat Flux, $q''$ (W/m²)", labelpad=10, fontweight="bold")
    cb.set_ticks(levels)

    plt.tight_layout()

    # Save the high-resolution image file
    fig.savefig("flux_contour_bands_2D_KDE.png", dpi=300, bbox_inches="tight")
    print("Successfully saved 2D un-interpolated KDE representation: 'flux_contour_bands_2D_KDE.png'")
    plt.show()

else:
    print("Error: Calculation variables not found. Make sure to run this block at the end of your main KDE execution script.")
# ============================================================
# FIXED: MASKED CYLINDRICAL PERIODIC KDE SURFACE VOLATILITY
# ============================================================
print("\n" + "="*50)
print("=== METHOD 3: KDE METRIC VERIFICATION SUMMARY ===")
print("="*50)

flux_contour_grid = flux_map_kde  
focal_mask = flux_contour_grid > (0.10 * flux_contour_grid.max())

# 1. ADDED: Compute Explicit Multivariate Theoretical Noise Floor [ 1/sqrt(N * h^2) ]
h_bandwidth = kde_estimator.factor  # Internal multi-variable scaling factor factor
theoretical_noise_floor_kde = (1.0 / np.sqrt(N_rays_total * (h_bandwidth**2))) * 100

# 2. Compute True Cylindrical Periodic Derivatives
df_dtheta = np.zeros_like(flux_contour_grid)
df_dtheta[1:-1, :] = (flux_contour_grid[2:, :] - flux_contour_grid[:-2, :]) / 2.0
df_dtheta[0, :] = (flux_contour_grid[1, :] - flux_contour_grid[-1, :]) / 2.0  
df_dtheta[-1, :] = (flux_contour_grid[0, :] - flux_contour_grid[-2, :]) / 2.0 

df_ds = np.gradient(flux_contour_grid, axis=1)
gradient_magnitude = np.sqrt(df_dtheta**2 + df_ds**2)

# 3. Raw Absolute Volatility (RMS Gradient)
spatial_volatility_m3 = np.sqrt(np.mean(gradient_magnitude[focal_mask]**2))

# 4. Scientific Surface Roughness Index (RI) with Periodic Wrap
from scipy.ndimage import gaussian_filter
macro_trend = gaussian_filter(flux_contour_grid, sigma=3.0, mode='wrap')
micro_stochastic_ripples = flux_contour_grid - macro_trend

rms_noise = np.sqrt(np.mean(micro_stochastic_ripples[focal_mask]**2))
surface_roughness_index = (rms_noise / flux_contour_grid.max()) * 100

# Unified Console Metrics Output
print(f"Total Rays Processed (N):                     {N_rays_total} rays")
print(f"Extracted Bandwidth Factor (h):               {h_bandwidth:.4f}")
print(f"Base Sample Theoretical Noise [1/sqrt(N*h²)]: {theoretical_noise_floor_kde:.2f} %")
print(f"Global Method 3 Surface Roughness Index (RI): {surface_roughness_index:.4f} %")
print(f"KDE Absolute Topography Volatility:           {spatial_volatility_m3:.2f} W/m² per node")
print("="*50 + "\n")

# 5. FIXED: Local Spatial Roughness Map Masked to Active Flux Area
local_roughness_magnitude = gaussian_filter(np.abs(micro_stochastic_ripples), sigma=1.5, mode='wrap')
kde_volatility_map = (local_roughness_magnitude / flux_contour_grid.max()) * 100

kde_volatility_map_masked = np.zeros_like(kde_volatility_map)
kde_volatility_map_masked[focal_mask] = kde_volatility_map[focal_mask]

max_visible_contour_limit = 5.0 
kde_volatility_map_masked = np.clip(kde_volatility_map_masked, 0, max_visible_contour_limit)

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print(f" Elapsed time: {elapsed_time} seconds")