# -*- coding: utf-8 -*-
"""
k-NN_Flux_Rec.py
k-Neartest Neighbor (k-NN) spectral MCRT flux reconstruction for cylindrical receivers.

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
from scipy.spatial import cKDTree

# ============================================================
# 1) PARAMETERS (Identical to Your Method 3 Framework)
# ============================================================
ASTM_FILE = "data/ASTMG173-direct.xlsx"   # path to ASTM G173-03 spectrum
HITS_FILE = "data/ray_hits.txt"            # path to OTSun ray-hit output file

receiver_outer_radius_mm = 35.0     # outer radius [mm]
receiver_length_m = 4.0            # receiver length [m]
aperture_width_m = 5.76             # [m]
collector_length_m = 4.0           # [m]

rays_per_wavelength = 10000         # OTSun setting
dlambda_nm = 20.0                   # wavelength step [nm]
wl_min_nm, wl_max_nm = 290.0, 3990.0

# Construct matching evaluation grids (EDITABLE)
N_eval_theta = 60
N_eval_s = 400

# Enter k-value (EDITABLE)
k_neighbors = 500 
boundary_tolerance_m = 0.25  # Tolerance at the start and end of pipe. 

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
    
    # SHIFTED TO MATCH IMAGE_11345E: Shifting by -pi/2 maps the peak concentrated
    # solar flux directly to the 0 and 360 degree boundaries (bottom of pipe).
    theta = np.arctan2(v, u) - (np.pi / 2.0)
    
    # Bound mapping space cleanly between [0, 2pi]
    theta = np.where(theta < 0, theta + 2*np.pi, theta)
    theta = np.where(theta >= 2*np.pi, theta - 2*np.pi, theta)
    
    wl = chunk["wl_nm"].to_numpy(float)
    
    a = np.maximum(wl - 0.5 * dlambda_nm, wl_min_nm)
    b = np.minimum(wl + 0.5 * dlambda_nm, wl_max_nm)
    
    Fa = np.interp(a, wl_astm, cumE, left=cumE[0], right=cumE[-1])
    Fb = np.interp(b, wl_astm, cumE, left=cumE[0], right=cumE[-1])
    E_band = Fb - Fa
    
    P_ray = (E_band * A_ap) / rays_per_wavelength
    w = chunk["energy_ray"].to_numpy(float) * P_ray
    P_abs_code += w.sum()
    
    theta_collected.append(theta)
    s_collected.append(s)
    weights_collected.append(w)

# Flatten memory chunks into continuous 1D arrays
theta_train = np.concatenate(theta_collected)
s_train = np.concatenate(s_collected)
w_train = np.concatenate(weights_collected)
N_rays_total = len(theta_train)

theta_left  = theta_train - 2 * np.pi
theta_right = theta_train + 2 * np.pi
theta_periodic = np.concatenate([theta_left,  theta_train, theta_right])
s_periodic     = np.tile(s_train, 3)
w_periodic     = np.tile(w_train, 3)

# ============================================================
# 5) METHOD 2 EXECUTION: HIGH-SPEED k-NN DENSITY INTERPOLATION
# ============================================================
print(f"\nBuilding spatial KD-Tree for {N_rays_total} rays...")
X_train = np.column_stack([theta_periodic * R, s_periodic])
ray_tree = cKDTree(X_train)


theta_axis = np.linspace(0, 2*np.pi, N_eval_theta)
s_axis = np.linspace(0, receiver_length_m, N_eval_s)
Theta_grid, S_grid = np.meshgrid(theta_axis, s_axis, indexing="ij")

T_flat = Theta_grid.ravel()
S_flat = S_grid.ravel()

block_size = 4000
flux_flat_knn = np.zeros(len(T_flat))

print(f"Evaluating k-NN continuum field using k={k_neighbors} neighbors with boundary corrections...")

for idx in range(0, len(T_flat), block_size):
    end_idx = min(idx + block_size, len(T_flat))
    T_sub = T_flat[idx:end_idx]
    S_sub = S_flat[idx:end_idx]
    
    X_eval_real = np.column_stack([T_sub * R, S_sub])
    distances_real, _ = ray_tree.query(X_eval_real, k=k_neighbors, workers=-1)
    radius_real = np.maximum(distances_real[:, -1], 1e-5)
    
    # Boundary Correction: Left Mirror (s = 0)
    X_eval_left = np.column_stack([T_sub * R, -S_sub])
    distances_left, _ = ray_tree.query(X_eval_left, k=k_neighbors, workers=-1)
    radius_left = np.maximum(distances_left[:, -1], 1e-5)
    
    # Boundary Correction: Right Mirror (s = L)
    X_eval_right = np.column_stack([T_sub * R, 2 * receiver_length_m - S_sub])
    distances_right, _ = ray_tree.query(X_eval_right, k=k_neighbors, workers=-1)
    radius_right = np.maximum(distances_right[:, -1], 1e-5)
    
    flux_real = (k_neighbors * (P_abs_code / N_rays_total)) / (np.pi * radius_real**2)
    flux_left = (k_neighbors * (P_abs_code / N_rays_total)) / (np.pi * radius_left**2)
    flux_right = (k_neighbors * (P_abs_code / N_rays_total)) / (np.pi * radius_right**2)
    
    weight_left = np.exp(-S_sub / boundary_tolerance_m)  
    weight_right = np.exp(-(receiver_length_m - S_sub) / boundary_tolerance_m)
    
    flux_flat_knn[idx:end_idx] = flux_real + (flux_left * weight_left) + (flux_right * weight_right)

flux_map_knn = flux_flat_knn.reshape(N_eval_theta, N_eval_s)

# Power scaling normalization via surface integration
dtheta_eval = 2 * np.pi / (N_eval_theta - 1)
ds_eval = receiver_length_m / (N_eval_s - 1)
dA_eval = R * dtheta_eval * ds_eval

P_reconstructed_knn = np.sum(flux_map_knn * dA_eval)
flux_map_knn *= (P_abs_code / P_reconstructed_knn)  # Perfect energy closure enforcement

# --- SPATIAL CELL-TO-CELL GRADIENT RMS NOISE ---
peak_profile_idx = np.argmax(np.mean(flux_map_knn, axis=1))
peak_longitudinal_flux = flux_map_knn[peak_profile_idx, :]
mean_peak_value = np.mean(peak_longitudinal_flux)
gradients_along_s = np.diff(peak_longitudinal_flux)
spatial_rms_noise_pct = (np.sqrt(np.mean(gradients_along_s**2)) / mean_peak_value) * 100

total_surface_area = 2 * np.pi * R * receiver_length_m
P_reconstructed_final = np.sum(flux_map_knn * dA_eval)
avg_flux_knn = P_reconstructed_final / total_surface_area

print("\n" + "="*50)
print("=== METHOD 2: k-NN METRIC VERIFICATION SUMMARY ===")
print("="*50)
print(f"Total Rays Processed:                         {N_rays_total} rays")
print(f"Raw Ray-Trace Absorbed Power (P_abs_code):    {P_abs_code:.4f} W")
print(f"Reconstructed Integrated Power (Absorber):    {P_reconstructed_final:.4f} W")
print(f"Average Surface Heat Flux (q''_avg):          {avg_flux_knn:.2f} W/m²")
print(f"Maximum Local Heat Flux (q''_max):            {flux_map_knn.max():.2f} W/m²")
print(f"Longitudinal Spatial RMS Noise:               {spatial_rms_noise_pct:.2f} %")
print("="*50 + "\n")

# ============================================================
# 6) CONTINUOUS 3D CYLINDRICAL VISUALIZATION
# ============================================================
print("Constructing 3D geometric surface mesh...")
flux_wrap  = np.vstack([flux_map_knn, flux_map_knn[0:1, :]])
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

norm = Normalize(vmin=flux_wrap.min(), vmax=flux_wrap.max())
colors = CMAP(norm(flux_wrap))

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

m = cm.ScalarMappable(norm=norm, cmap=CMAP)
m.set_array(flux_wrap)
cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.05)
cb.ax.set_title("$q$ (W/m²)", fontsize=11, pad=10, ha="left")

ax.set_xlim(np.min(X), np.max(X))
ax.set_ylim(np.min(Y), np.max(Y))
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_yticks(np.arange(0, receiver_length_m + 1, 1))

for lbl in ax.get_xticklabels():
    lbl.set_rotation(30)
    lbl.set_horizontalalignment('right')  # helps avoid overlap
    lbl.set_verticalalignment('center')
    
# --- bring tick numbers closer to the axis lines ---
ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=-2)
ax.tick_params(axis='z', pad=4)

ax.set_xlabel("Receiver diameter, D (m)", labelpad=4, fontsize=11)
ax.set_ylabel("Receiver length, L (m)", labelpad=4, fontsize=11)
ax.set_zlabel("Receiver height, H (m)", labelpad=10, fontsize=11)
ax.view_init(elev=20, azim=-60)
ax.set_box_aspect((1, 3, 1))
plt.tight_layout()
fig.savefig("flux_receiver_3D_KNN.jpg", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()

# ============================================================
# 7) 2D DISCRETE ISOFLUX TOPOGRAPHY MAPPING (UNSMOOTHED)
# ============================================================
print("Generating 2D unsmoothed k-NN checkerboard map...")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

fig2d, ax2d = plt.subplots(figsize=(8, 5), dpi=300)
 
levels = np.linspace(0, np.ceil(flux_map_knn.max() / 5000) * 5000, 11) 

# Plot raw grid blocks directly using 'nearest' shading to prevent contour blending
contour_filled = ax2d.pcolormesh(
    S_grid, np.degrees(Theta_grid), flux_map_knn, 
    cmap="jet",
    vmin=0,
    vmax=levels[-1],
    shading='nearest'
)

# NOTE: Continuous line overlays (ax2d.contour & ax2d.clabel) have been removed
# to preserve the raw block layout.

ax2d.set_xlabel("Receiver Longitudinal Axis, $s$ (m)", labelpad=8)
ax2d.set_ylabel("Receiver Circumferential Angle, $\\theta$ (deg)", labelpad=8)
#ax2d.set_title("2D Discrete Heat Flux Topography ($q''$ in W/m²)\nEvaluated via k-Nearest Neighbors (k-NN) Node Points", pad=15, fontweight="bold")
ax2d.set_xlim(0, receiver_length_m)
ax2d.set_ylim(0, 360)
ax2d.set_yticks([0, 90, 180, 270, 360])
ax2d.grid(True, linestyle=":", alpha=0.4, color="#cccccc")

cb2d = fig2d.colorbar(contour_filled, ax=ax2d, pad=0.05, shrink=0.9)
cb2d.set_label("", labelpad=10)
cb2d.ax.set_title("$q$ (W/m²)", pad=10, fontsize=11, ha="left")
cb2d.set_ticks(levels)

plt.tight_layout()
fig2d.savefig("flux_discrete_grid_2D_KNN.png", dpi=600, bbox_inches="tight")
print("Saved 2D un-interpolated block representation: 'flux_discrete_grid_2D_KNN.png'")
plt.show()

# ============================================================
# EXTRA: METHOD 2 (k-NN) MCRT NOISE & LOCAL GRADIENT MAP
# ============================================================
print("\n--- Method 2 MCRT Noise Metrics ---")

print(f"Base Sample Statistical Noise (1/sqrt(k)): {(1.0 / np.sqrt(k_neighbors)) * 100:.2f}%")
print(f"Empirical Longitudinal Spatial RMS Noise: {spatial_rms_noise_pct:.2f}%")

df_dtheta, df_ds = np.gradient(flux_map_knn)
focal_mask = flux_map_knn > (0.10 * flux_map_knn.max())
spatial_volatility = np.sqrt(np.mean(df_dtheta[focal_mask]**2 + df_ds[focal_mask]**2))
print(f"Spatial Contour Roughness Metric (RMS Gradient): {spatial_volatility:.2f} W/m² per node")

# ==============================================================================
# ADDED === EXTRA: UNIFIED SURFACE ROUGHNESS INDEX (RI) ANALYSIS === ADDED
# ==============================================================================
print("\n--- Method 2 Unified Surface Roughness Metrics ---")
from scipy.ndimage import gaussian_filter

# 1. Retain the same 10% threshold to isolate the active illumination envelope
focal_mask_ri = flux_map_knn > (0.10 * flux_map_knn.max())

# 2. Extract physical macro trend using a periodic wrap across the 0-360 boundaries
macro_trend_knn = gaussian_filter(flux_map_knn, sigma=3.0, mode='wrap')

# 3. Isolate high-frequency micro-stochastic ripples (pure numerical noise)
micro_ripples_knn = flux_map_knn - macro_trend_knn

# 4. Compute the global Surface Roughness Index (RI) percentage inside the active zone
rms_noise_knn = np.sqrt(np.mean(micro_ripples_knn[focal_mask_ri]**2))
surface_roughness_index_knn = (rms_noise_knn / flux_map_knn.max()) * 100
print(f"Global Method 2 Surface Roughness Index (RI): {surface_roughness_index_knn:.4f}%")

# 5. Local Spatial Roughness Map Masked to Active Flux Area
local_roughness_knn = gaussian_filter(np.abs(micro_ripples_knn), sigma=1.5, mode='wrap')
knn_ri_map = (local_roughness_knn / flux_map_knn.max()) * 100

# Apply mask to eliminate zero-flux background clutter
knn_ri_map_masked = np.zeros_like(knn_ri_map)
knn_ri_map_masked[focal_mask_ri] = knn_ri_map[focal_mask_ri]

# Set a clean visual ceiling for plotting comparison (Matches your 5.0% ceiling)
max_visible_ri_limit = 5.0
knn_ri_map_masked = np.clip(knn_ri_map_masked, 0, max_visible_ri_limit)

# ==============================================================================
# PROVISION: EXPORT STRUCTURED CFD FLUX BOUNDARY CONDITION (k-NN METHOD)
# ==============================================================================
print("\nAssembling structured data frame for CFD solver injection...")

# Flatten 2D evaluation coordinate grids and corresponding flux values to 1D columns
flat_theta_deg = np.degrees(Theta_grid).flatten()
flat_s_m = S_grid.flatten()
flat_flux = flux_map_knn.flatten()

# Calculate the matching cell element area based on grid intervals
dtheta_export = 2 * np.pi / (N_eval_theta)
ds_export = receiver_length_m / (N_eval_s)
dA_export = R * dtheta_export * ds_export

# Fill a matching array with the uniform spatial cell area
flat_area = np.full_like(flat_flux, dA_export)

# Assemble structured text dataframe
cfd_export_data = np.column_stack((flat_theta_deg, flat_s_m, flat_area, flat_flux))
    
# Define fallback output directory if not specified globally
OUTPUT_DIR = "" 
csv_filename = f"KNN_{N_eval_theta}x{N_eval_s}_{rays_per_wavelength}.csv"   #Put directory to save results. 

# Write file out with structured header tracking coordinates
np.savetxt(
    csv_filename, 
    cfd_export_data, 
    delimiter=",", 
    fmt=["%.4f", "%.4f", "%.8f", "%.2f"],
    header="CircumferentialAngle_deg,LongitudinalAxis_m,CellArea_m2,HeatFlux_W_m2",
    comments=""
)
print(f" -> SUCCESS: Exported clean boundary condition data file to: {csv_filename}")
# ==============================================================================

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print(f" Elapsed time: {elapsed_time} seconds")