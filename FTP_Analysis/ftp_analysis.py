# -*- coding: utf-8 -*-
"""
ftp_analysis_v2.py
==================
Flux Transfer Performance (FTP) Analysis
Updated for 6 MCRT resolutions × 6 CFD meshes × 3 methods = 108 combinations.

Resolutions: 20×20, 80×80, 300×300, 60×400, 80×600, 100×1000

Metrics computed for every CSV × mesh combination:
  ETA  — Energy Transfer Accuracy      (%)
  PTA  — Peak Transfer Accuracy        (%)
  SSI  — Spatial Staircase Index       (%) ← solver difficulty
  ILR  — Information Loss Ratio        (%) ← spatial pattern loss
  RCR  — Resolution Compatibility Ratio
  FTP  — Composite score (0–1)

Outputs:
  FTP_Results/FTP_all_results.csv          all 108 rows
  FTP_Results/FTP_matrix_BB.csv/png        6×6 heatmap for bin-based
  FTP_Results/FTP_matrix_kNN.csv/png       6×6 heatmap for k-NN
  FTP_Results/FTP_matrix_KDE.csv/png       6×6 heatmap for KDE
  FTP_Results/FTP_metric_bars.png          per-metric comparison bars
  FTP_Results/FTP_histograms.png           flux distribution comparison
  FTP_Results/FTP_solver_difficulty.png    adjacent-face jump distributions

Usage:
  python3 ftp_analysis.py

Author: MCRT-CFD Flux Mapping Research
"""

import numpy as np
import pandas as pd
import re, os, json, time, warnings
from scipy.spatial     import cKDTree
from scipy.interpolate import RegularGridInterpolator
from scipy.stats       import entropy as scipy_entropy
from scipy.ndimage     import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
warnings.filterwarnings('ignore')


# ================================================================
#  SECTION 1 — USER SETTINGS
#  Edit the paths to match your directory structure.
# ================================================================

BASE   = "/home/uib/OneDrive/CFD_Studies"
CSV_DIR  = os.path.join(BASE, "FTP_Analysis/CSV")
MESH_DIR = os.path.join(BASE, "FTP_Analysis/Meshes")
ADJ_DIR  = os.path.join(BASE, "FTP_Analysis/adjacency_cache")
OUT_DIR  = os.path.join(BASE, "FTP_Analysis/FTP_Results")

# Physical constants — same geometry for all meshes
P_MCRT_W = 16506.5   # W  total absorbed power from MCRT (your validated value)
R_OUTER  = 0.035      # m  outer tube radius
L_TUBE   = 4.0        # m  tube length
PIPE_AXIS = "y"       # axis along which pipe runs in the mesh

# Patch name on CFD mesh outer surface
REGION = "Tube"
PATCH  = "Tube_outer"   # ← change if Salome named it differently

# Number of nearest neighbours for adjacency detection (4 for structured hex)
K_ADJ = 4

# Number of histogram bins for ILR metric
N_HIST_BINS = 60

# ── MCRT CSV files (18 total: 6 resolutions × 3 methods) ──────────
MCRT_FILES = {
    # Bin-based method
    "BB_20x20"     : os.path.join(CSV_DIR, "BB/BB_20x20_10000.csv"),
    "BB_80x80"     : os.path.join(CSV_DIR, "BB/BB_80x80_10000.csv"),
    "BB_300x300"   : os.path.join(CSV_DIR, "BB/BB_300x300_10000.csv"),
    "BB_60x400"    : os.path.join(CSV_DIR, "BB/BB_60x400_10000.csv"),
    "BB_80x600"    : os.path.join(CSV_DIR, "BB/BB_80x600_10000.csv"),
    "BB_100x1000"  : os.path.join(CSV_DIR, "BB/BB_100x1000_10000.csv"),
    # k-Nearest Neighbour (k=500)
    "kNN_20x20"    : os.path.join(CSV_DIR, "kNN/KNN_20x20_10000.csv"),
    "kNN_80x80"    : os.path.join(CSV_DIR, "kNN/KNN_80x80_10000.csv"),
    "kNN_300x300"  : os.path.join(CSV_DIR, "kNN/KNN_300x300_10000.csv"),
    "kNN_60x400"   : os.path.join(CSV_DIR, "kNN/KNN_60x400_10000.csv"),
    "kNN_80x600"   : os.path.join(CSV_DIR, "kNN/KNN_80x600_10000.csv"),
    "kNN_100x1000" : os.path.join(CSV_DIR, "kNN/KNN_100x1000_10000.csv"),
    # KDE (FFTKDE, Scott's bandwidth)
    "KDE_20x20"    : os.path.join(CSV_DIR, "KDE/KDE_20x20_10000.csv"),
    "KDE_80x80"    : os.path.join(CSV_DIR, "KDE/KDE_80x80_10000.csv"),
    "KDE_300x300"  : os.path.join(CSV_DIR, "KDE/KDE_300x300_10000.csv"),
    "KDE_60x400"   : os.path.join(CSV_DIR, "KDE/KDE_60x400_10000.csv"),
    "KDE_80x600"   : os.path.join(CSV_DIR, "KDE/KDE_80x600_10000.csv"),
    "KDE_100x1000" : os.path.join(CSV_DIR, "KDE/KDE_100x1000_10000.csv"),
}

# ── CFD mesh directories (6 meshes) ───────────────────────────────
CFD_MESHES = {
    "Mesh_20x20"    : os.path.join(MESH_DIR, "20_20"),
    "Mesh_80x80"    : os.path.join(MESH_DIR, "80_80"),
    "Mesh_300x300"  : os.path.join(MESH_DIR, "300_300"),
    "Mesh_60x400"   : os.path.join(MESH_DIR, "60_400"),
    "Mesh_80x600"   : os.path.join(MESH_DIR, "80_600"),
    "Mesh_100x1000" : os.path.join(MESH_DIR, "100_1000"),
}

# Resolution labels for matrix axes (order matters — keeps matrix sorted)
RES_ORDER = ["20x20", "80x80", "300x300", "60x400", "80x600", "100x1000"]
METHOD_ORDER = ["BB", "kNN", "KDE"]

for d in [ADJ_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)


# ================================================================
#  SECTION 2 — MESH READING UTILITIES
# ================================================================

def _strip_foam_comments(text):
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text

def read_foam_points(case_dir, region):
    fpath = os.path.join(case_dir, "constant", region, "polyMesh", "points")
    with open(fpath) as f:
        c = _strip_foam_comments(f.read())
    m = re.search(r'^\s*(\d+)\s*\n\s*\(', c, re.MULTILINE)
    n = int(m.group(1))
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', c[m.end():])
    return np.array(nums[:n*3], dtype=float).reshape(n, 3)

def get_patch_face_range(case_dir, region, patch):
    """Return (start_face, n_faces) for the named patch."""
    bpath = os.path.join(case_dir, "constant", region, "polyMesh", "boundary")
    with open(bpath) as f:
        bc = _strip_foam_comments(f.read())

    # Try nFaces before startFace
    m = re.search(
        patch + r'\s*\{[^}]*nFaces\s+(\d+)[^}]*startFace\s+(\d+)',
        bc, re.DOTALL)
    if m:
        return int(m.group(2)), int(m.group(1))  # start, n

    # Try startFace before nFaces
    m = re.search(
        patch + r'\s*\{[^}]*startFace\s+(\d+)[^}]*nFaces\s+(\d+)',
        bc, re.DOTALL)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Try partial patch name match (in case Salome added a suffix)
    m = re.search(
        r'(\w*' + re.escape(patch) + r'\w*)\s*\{[^}]*nFaces\s+(\d+)'
        r'[^}]*startFace\s+(\d+)', bc, re.DOTALL)
    if m:
        print(f"    WARNING: patch '{patch}' matched as '{m.group(1)}'")
        return int(m.group(3)), int(m.group(2))

    raise ValueError(
        f"Patch '{patch}' not found in {bpath}.\n"
        f"Run:  grep -A5 'outer' {bpath}"
    )

def read_patch_face_centres(case_dir, region, patch):
    """
    Returns (face_angles_deg, face_s_m, face_area, n_faces).
    Reads mesh topology directly from polyMesh files.
    """
    points = read_foam_points(case_dir, region)
    start, n_faces = get_patch_face_range(case_dir, region, patch)

    fpath = os.path.join(case_dir, "constant", region, "polyMesh", "faces")
    with open(fpath) as f:
        c = _strip_foam_comments(f.read())
    m = re.search(r'^\s*\d+\s*\n\s*\(', c, re.MULTILINE)
    face_strs = re.compile(r'\d*\s*\(([^)]+)\)').findall(c[m.end():])
    patch_fs  = face_strs[start : start + n_faces]

    centres = np.zeros((n_faces, 3))
    for i, fs in enumerate(patch_fs):
        verts = list(map(int, fs.split()))
        centres[i] = points[verts].mean(axis=0)

    ax_map = {"x": 0, "y": 1, "z": 2}
    ax  = ax_map[PIPE_AXIS.lower()]
    r1, r2 = [i for i in range(3) if i != ax]

    face_s   = centres[:, ax]
    face_ang = (np.degrees(np.arctan2(centres[:, r2], centres[:, r1]))) % 360
    face_area = 2 * np.pi * R_OUTER * L_TUBE / n_faces

    return face_ang, face_s, face_area, n_faces


# ================================================================
#  SECTION 3 — ADJACENCY COMPUTATION (cached per mesh)
# ================================================================

def build_adjacency(face_angles_deg, face_s_m, mesh_label, k=K_ADJ):
    """
    Find adjacent face pairs using KD-Tree in arc-length space.
    Handles angular periodicity.  Caches result as JSON.
    O(N log N) — fast even for 100,000-face meshes.
    """
    cache = os.path.join(ADJ_DIR, f"adj_{mesh_label}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            pairs = [tuple(p) for p in json.load(f)]
        print(f"    Adjacency: {len(pairs):,} pairs (loaded from cache)")
        return pairs

    print(f"    Building adjacency (N={len(face_angles_deg):,} faces)...",
          end=" ", flush=True)
    t0 = time.time()

    period = 2 * np.pi * R_OUTER
    arc    = np.radians(face_angles_deg) * R_OUTER

    # Augment with periodic copies to handle the 0°/360° seam
    X_base  = np.column_stack([arc,          face_s_m])
    X_left  = np.column_stack([arc - period, face_s_m])
    X_right = np.column_stack([arc + period, face_s_m])
    X_all   = np.vstack([X_base, X_left, X_right])
    n = len(face_angles_deg)

    tree = cKDTree(X_all)
    _, idxs = tree.query(X_base, k=k+1, workers=-1)

    pairs = set()
    for i in range(n):
        for j_raw in idxs[i, 1:]:
            j = int(j_raw) % n
            if j != i:
                pairs.add((min(i, j), max(i, j)))

    pairs = list(pairs)
    with open(cache, 'w') as f:
        json.dump(pairs, f)
    print(f"{len(pairs):,} pairs  ({time.time()-t0:.1f} s)")
    return pairs


# ================================================================
#  SECTION 4 — MAP CSV FLUX ONTO MESH FACE CENTRES
# ================================================================

def load_csv(csv_path):
    df      = pd.read_csv(csv_path)
    angles  = sorted(df['CircumferentialAngle_deg'].unique())
    z_vals  = sorted(df['LongitudinalAxis_m'].unique())
    flux_2d = df.pivot(index='CircumferentialAngle_deg',
                       columns='LongitudinalAxis_m',
                       values='HeatFlux_W_m2').values
    c_area  = df['CellArea_m2'].iloc[0]
    return np.array(angles), np.array(z_vals), flux_2d, len(df), c_area

def map_csv_to_faces(csv_path, face_angles_deg, face_s_m, face_area):
    angles, z_vals, flux_2d, n_csv, _ = load_csv(csv_path)
    label = os.path.basename(csv_path)

    interp_method = 'nearest' if (label.startswith("BB") or label.startswith("Method1")) else 'linear'

    interp = RegularGridInterpolator(
        (angles, z_vals), flux_2d,
        method=interp_method, bounds_error=False, fill_value=None
    )
    pts = np.column_stack([face_angles_deg, face_s_m])
    
    # 1. Pure interpolated flux directly from the data mapping
    flux_raw = np.clip(interp(pts), 0.0, None)
    
    # 2. Calculate true diagnostic mapping energy before scaling
    P_raw_mapped = flux_raw.sum() * face_area
    diagnostic_ETA = (P_raw_mapped / P_MCRT_W) * 100.0
    
    # 3. Scale the profile to enforce hard energy closure for the CFD solver
    flux_corrected = flux_raw.copy()
    if P_raw_mapped > 0:
        flux_corrected *= (P_MCRT_W / P_raw_mapped)
        
    return flux_corrected, flux_2d, n_csv, diagnostic_ETA


# ================================================================
#  SECTION 5 — FTP METRIC FUNCTIONS
# ================================================================

def metric_ETA(flux_faces, face_area):
    """Energy Transfer Accuracy (%) — how much of P_MCRT is received."""
    return (flux_faces.sum() * face_area / P_MCRT_W) * 100.0
'''
def metric_PTA(flux_faces, flux_2d_csv):
    """Peak Transfer Accuracy (%) — how much of q_max survives transfer."""
    q_max_csv  = flux_2d_csv.max()
    q_max_mesh = flux_faces.max()
    return (q_max_mesh / q_max_csv * 100.0) if q_max_csv > 0 else 0.0
'''
# PTA must compare against physical truth, not own CSV:
# q_ref = k-NN k=500 at max rays on the 60×400 mesh — your best estimate of reality
Q_MAX_PHYSICAL_REFERENCE = 44500  # W/m²  
# Set this from your k-NN at 50,000 rays/λ on 60×400 mesh

def metric_PTA(flux_faces):
    """Compare to physical reference, not own CSV."""
    return (flux_faces.max() / Q_MAX_PHYSICAL_REFERENCE) * 100.0

def metric_SSI(flux_faces, adj_pairs):
    """
    Vectorized Spatial Staircase Index.
    Expects adj_pairs to be a 2D NumPy array of shape (N, 2).
    """
    if len(adj_pairs) == 0:
        return 0.0
    
    # Convert to numpy array once during loading if not already done
    adj_array = np.asarray(adj_pairs)
    
    # Fast vectorized element index slicing
    jumps = np.abs(flux_faces[adj_array[:, 0]] - flux_faces[adj_array[:, 1]])
    
    return (jumps.mean() / flux_faces.mean()) * 100.0

def metric_ILR(flux_faces, flux_2d_csv, n_bins=N_HIST_BINS):
    """
    Information Loss Ratio (%) — Shannon entropy comparison.
    ILR = (1 - H_mesh / H_MCRT) × 100%.
    No interpolation; compares distributions directly.
    """
    q_all = np.concatenate([flux_faces.ravel(), flux_2d_csv.ravel()])
    bins  = np.linspace(q_all.min(), q_all.max() + 1, n_bins + 1)

    p_csv,  _ = np.histogram(flux_2d_csv.ravel(), bins=bins)
    p_mesh, _ = np.histogram(flux_faces.ravel(),  bins=bins)

    p_csv  = p_csv  / p_csv.sum()  + 1e-10
    p_mesh = p_mesh / p_mesh.sum() + 1e-10

    H_csv  = scipy_entropy(p_csv,  base=2)
    H_mesh = scipy_entropy(p_mesh, base=2)

    return max(0.0, (1.0 - H_mesh / H_csv)) * 100.0 if H_csv > 0 else 0.0

def metric_RCR(n_mcrt_cells, n_mesh_faces):
    """
    Resolution Compatibility Ratio — pure geometry, no flux data needed.
    RCR = 1.0 for k-NN/KDE (mesh-independent); varies for BB.
    """
    return min(n_mcrt_cells, n_mesh_faces) / max(n_mcrt_cells, n_mesh_faces)

def FTP_composite(ETA, PTA, SSI, ILR, RCR):
    ETA_n = np.clip(ETA  / 100.0, 0, 1)
    PTA_n = np.clip(1.0  - abs(PTA - 100.0) / 100.0, 0, 1)
    SSI_n = np.clip(1.0  - SSI / 100.0, 0, 1)
    ILR_n = np.clip(1.0  - ILR / 100.0, 0, 1)
    RCR_n = np.clip(RCR, 0, 1)
    # SSI weighted double — most critical for solver behaviour
    return (ETA_n + PTA_n + 2*SSI_n + ILR_n + RCR_n) / 6.0


# ================================================================
#  SECTION 6 — MAIN BATCH LOOP (18 CSVs × 6 meshes = 108 runs)
# ================================================================

def run_all():
    print("=" * 65)
    print("  FLUX TRANSFER PERFORMANCE (FTP) ANALYSIS  v2")
    print(f"  {len(MCRT_FILES)} CSV files  ×  {len(CFD_MESHES)} meshes"
          f"  =  {len(MCRT_FILES)*len(CFD_MESHES)} combinations")
    print("=" * 65)

    # ── Pre-load all meshes (face centres + adjacency) ─────────────
    print("\n[Step 1] Loading all CFD meshes and building adjacency...")
    mesh_data = {}
    for mesh_label, mesh_dir in CFD_MESHES.items():
        print(f"\n  {mesh_label}")
        if not os.path.isdir(mesh_dir):
            print(f"    SKIP — directory not found: {mesh_dir}")
            continue
        try:
            face_ang, face_s, face_area, n_faces = \
                read_patch_face_centres(mesh_dir, REGION, PATCH)
            adj_pairs = build_adjacency(face_ang, face_s, mesh_label)
            mesh_data[mesh_label] = {
                'face_ang'  : face_ang,
                'face_s'    : face_s,
                'face_area' : face_area,
                'n_faces'   : n_faces,
                'adj_pairs' : adj_pairs,
            }
            print(f"    Faces: {n_faces:,}  |  "
                  f"Angle: {face_ang.min():.1f}°–{face_ang.max():.1f}°  |  "
                  f"s: {face_s.min():.3f}–{face_s.max():.3f} m")
        except Exception as e:
            print(f"    ERROR: {e}")

    if not mesh_data:
        print("\nNo meshes loaded. Check paths in CFD_MESHES.")
        return

    # ── Main loop: every CSV × every mesh ──────────────────────────
    print(f"\n[Step 2] Computing FTP metrics "
          f"({len(MCRT_FILES)} × {len(mesh_data)} combinations)...")

    all_results = []
    combo_count = 0
    total_combos = len(MCRT_FILES) * len(mesh_data)

    for csv_label, csv_path in MCRT_FILES.items():
        if not os.path.exists(csv_path):
            print(f"  SKIP {csv_label}: file not found")
            continue

        # Parse method and resolution from label
        parts  = csv_label.split('_', 1)
        method = parts[0]            # BB / kNN / KDE
        mcrt_res = parts[1] if len(parts) > 1 else csv_label

# Define the independent analytical target peak flux to avoid circular evaluation
        Q_MAX_BENCHMARK = 44500.0  # W/m2 (Adjust based on your independent benchmark)

        for mesh_label, md in mesh_data.items():
            combo_count += 1
            tag = f"[{combo_count:>3}/{total_combos}]"

            try:
                # 1. Unpack mapping outputs
                flux_faces, flux_2d, n_csv, diagnostic_ETA = map_csv_to_faces(
                    csv_path,
                    md['face_ang'], md['face_s'], md['face_area']
                )

                # 2. Extract Energy Conservation
                ETA = diagnostic_ETA 
                
                # 3. Calculate Peak Accuracy using the independent benchmark
                PTA = (flux_faces.max() / Q_MAX_PHYSICAL_REFERENCE) * 100.0
                
                # 4. Compute original spatial indices
                SSI = metric_SSI(flux_faces, md['adj_pairs'])
                ILR = metric_ILR(flux_faces, flux_2d)
                
                
                # 5. Calculate UPDATED Scale-Dependent RCR*
                RCR_base = metric_RCR(n_csv, md['n_faces'])
                
                # Safe check: Use .mean() if it's an array, otherwise use the float directly
                if hasattr(md['face_area'], 'mean'):
                    delta_s = np.sqrt(md['face_area'].mean())
                else:
                    delta_s = np.sqrt(md['face_area'])
                    
                D_receiver = 2 * R_OUTER                   # Uses your existing R_OUTER constant
                RCR_updated = RCR_base * (1.0 - (delta_s / D_receiver))

                # 6. Normalize components to a 0.0 - 1.0 scale for Geometric Mean
                ETA_idx  = ETA / 100.0
                SSI_idx  = (100.0 - SSI) / 100.0
                ILR_idx  = (100.0 - ILR) / 100.0
                RCR_idx  = RCR_updated
                
                # Symmetric Peak Penalty: Penalizes both unphysical overshoots and undershoots
                PTA_accuracy = 1.0 - min(abs(PTA - 100.0) / 100.0, 1.0)

                # 7. Composite FTP Score — equal-weight arithmetic mean (5 components, weight = 0.2 each)
                FTP = (0.2 * ETA_idx) + (0.2 * PTA_accuracy) + (0.2 * SSI_idx) + (0.2 * ILR_idx) + (0.2 * RCR_idx)

                # Store the updated results
                row = {
                    'Method'      : method,
                    'MCRT_res'    : mcrt_res,
                    'Mesh'        : mesh_label.replace('Mesh_', ''),
                    'MCRT_cells'  : n_csv,
                    'Mesh_faces'  : md['n_faces'],
                    'P_recv_W'    : round(flux_faces.sum() * md['face_area'], 3),
                    'q_max_csv'   : round(flux_2d.max(),   1),
                    'q_max_mesh'  : round(flux_faces.max(), 1),
                    'ETA_%'       : round(ETA, 3),
                    'PTA_%'       : round(PTA, 1),
                    'SSI_%'       : round(SSI, 3),
                    'ILR_%'       : round(ILR, 3),
                    'RCR'         : round(RCR_updated, 4),  # Saved as RCR_updated
                    'FTP_score'   : round(FTP, 4),          # New Geometric Mean Score
                    '_flux_faces' : flux_faces,
                    '_flux_2d'    : flux_2d,
                }
                all_results.append(row)

                print(f"  {tag} {csv_label:<18} → {mesh_label:<18} "
                      f"ETA={ETA:.1f}% PTA={PTA:.0f}% "
                      f"SSI={SSI:.1f}% ILR={ILR:.1f}% "
                      f"RCR={RCR_updated:.3f} FTP={FTP:.3f}")

            except Exception as e:
                print(f"  {tag} ERROR: {csv_label} → {mesh_label}: {e}")

    if not all_results:
        print("No results computed.")
        return

    # ── Save flat results table ─────────────────────────────────────
    df_all = pd.DataFrame(all_results)
    df_export = df_all.drop(columns=['_flux_faces', '_flux_2d'], errors='ignore')
    csv_out = os.path.join(OUT_DIR, "FTP_all_results.csv")
    df_export.to_csv(csv_out, index=False)
    print(f"\n  Full results saved: {csv_out}")

    # ── Print summary table ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(df_export[['Method','MCRT_res','Mesh',
                      'ETA_%','PTA_%','SSI_%','ILR_%',
                      'RCR','FTP_score']].to_string(index=False))

    # ── Build and save 6×6 FTP matrices (one per method) ───────────
    print("\n[Step 3] Building FTP score matrices...")
    for method in METHOD_ORDER:
        sub = df_export[df_export['Method'] == method]
        if sub.empty:
            continue
        mat = pd.DataFrame(index=RES_ORDER, columns=RES_ORDER, dtype=float)
        for _, row in sub.iterrows():
            mr = row['MCRT_res']
            me = row['Mesh']
            if mr in RES_ORDER and me in RES_ORDER:
                mat.loc[mr, me] = row['FTP_score']
        mat_path = os.path.join(OUT_DIR, f"FTP_matrix_{method}.csv")
        mat.to_csv(mat_path)
        print(f"  {method} matrix saved: {mat_path}")

    # ================================================================
    #  SECTION 7 — VISUALISATIONS
    # ================================================================
    print("\n[Step 4] Generating visualisations...")
    _plot_ftp_matrices(df_export)
    _plot_metric_bars(df_export)
    _plot_histograms(all_results)
    _plot_solver_difficulty(all_results)
    _print_layman_summary(df_export)
    print(f"\n  All outputs → {OUT_DIR}")


# ================================================================
#  SECTION 7a — FTP SCORE HEATMAPS (one per method)
# ================================================================

def _plot_ftp_matrices(df_exp):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0f1117')

    for ax, method in zip(axes, METHOD_ORDER):
        sub = df_exp[df_exp['Method'] == method]
        mat = np.full((len(RES_ORDER), len(RES_ORDER)), np.nan)
        for _, row in sub.iterrows():
            mr = row['MCRT_res'];  me = row['Mesh']
            if mr in RES_ORDER and me in RES_ORDER:
                mat[RES_ORDER.index(mr), RES_ORDER.index(me)] = row['FTP_score']

        im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1,
                       aspect='auto', interpolation='none')
        ax.set_xticks(range(len(RES_ORDER))); ax.set_xticklabels(RES_ORDER, rotation=35, ha='right', fontsize=8, color='white')
        ax.set_yticks(range(len(RES_ORDER))); ax.set_yticklabels(RES_ORDER, fontsize=8, color='white')
        ax.set_xlabel("CFD Mesh Resolution", color='white', fontsize=10)
        ax.set_ylabel("MCRT Resolution", color='white', fontsize=10)
        ax.set_title(f"FTP Score — {method}", color='white',
                     fontsize=12, fontweight='bold', pad=8)
        ax.set_facecolor('#1a1a2e')

        # Annotate cells with score values
        for i in range(len(RES_ORDER)):
            for j in range(len(RES_ORDER)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}",
                            ha='center', va='center',
                            fontsize=7, color='black', fontweight='bold')

        # Mark diagonal (matched resolution)
        for k in range(len(RES_ORDER)):
            ax.add_patch(plt.Rectangle(
                (k-0.5, k-0.5), 1, 1,
                fill=False, edgecolor='cyan', linewidth=2
            ))

        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("FTP score", color='white', fontsize=9)
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    fig.suptitle(
        "FTP Score Matrix  (MCRT Resolution × CFD Mesh Resolution)\n"
        "Diagonal (cyan) = matched resolution.  "
        "k-NN and KDE rows should be uniformly green.",
        color='white', fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout(pad=1.5)
    path = os.path.join(OUT_DIR, "FTP_score_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  SECTION 7b — PER-METRIC BAR COMPARISON
# ================================================================

def _plot_metric_bars(df_exp):
    # Show only the matched-resolution cases for clarity
    matched = df_exp[df_exp['MCRT_res'] == df_exp['Mesh']].copy()
    if matched.empty:
        # Fall back to 60×400 as a representative resolution
        matched = df_exp[df_exp['MCRT_res'] == '60x400'].copy()

    metrics = ['ETA_%','PTA_%','SSI_%','ILR_%','RCR','FTP_score']
    titles  = ['ETA — Energy Accuracy (%)\nTarget > 99.5%',
               'PTA — Peak Accuracy (%)\nTarget 80–120%',
               'SSI — Staircase Index (%)\nLower = better',
               'ILR — Info Loss (%)\nLower = better',
               'RCR — Resolution Match\nTarget 1.0',
               'FTP — Composite Score\nHigher = better']
    colors  = ['#00ff88','#ffaa00','#ff4444','#ff8844','#4488ff','#aa88ff']

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor('#0f1117')

    labels = (matched['Method'] + '\n' + matched['MCRT_res']).tolist()
    x      = np.arange(len(labels))

    for ax, metric, title, color in zip(
            axes.flatten(), metrics, titles, colors):
        vals = matched[metric].values
        bars = ax.bar(x, vals, color=color, alpha=0.8, width=0.65)
        ax.set_title(title, color='white', fontsize=9,
                     fontweight='bold', pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right',
                           fontsize=7, color='white')
        ax.tick_params(colors='white', labelsize=7)
        ax.set_facecolor('#1a1a2e')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=6, color='white')

    plt.suptitle('FTP Metrics — Matched Resolution Cases (MCRT res = Mesh res)',
                 color='white', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(pad=1.5)
    path = os.path.join(OUT_DIR, "FTP_metric_bars.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  SECTION 7c — FLUX HISTOGRAMS (60×400 mesh, all methods)
# ================================================================

def _plot_histograms(all_results):
    # Select the 60×400 mesh results for comparison
    target_mesh = '60x400'
    subset = [r for r in all_results if r['Mesh'] == target_mesh]
    if not subset:
        target_mesh = list({r['Mesh'] for r in all_results})[0]
        subset = [r for r in all_results if r['Mesh'] == target_mesh]
    if not subset:
        return

    colors = ['#ff4444','#ff8844','#ffaa00','#ffff44',
              '#00ff88','#4488ff','#aa88ff','#ff44ff',
              '#44ffff','#ff4488','#88ff44','#4444ff',
              '#ff8844','#44ff88','#8844ff','#ff4444',
              '#88ffff','#ffff88']

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1a2e')

    for r, color in zip(subset, colors):
        flux = r['_flux_faces']
        label = f"{r['Method']}_{r['MCRT_res']} (ILR={r['ILR_%']:.1f}%)"
        ax.hist(flux/1000, bins=60, alpha=0.45,
                label=label, color=color, density=True)

    ax.set_xlabel('Heat Flux (kW/m²)', color='white', fontsize=11)
    ax.set_ylabel('Probability Density', color='white', fontsize=11)
    ax.set_title(
        f'Flux Distribution on CFD Faces ({target_mesh} mesh)\n'
        'Narrower or shifted curves = information lost in transfer',
        color='white', fontsize=11, fontweight='bold'
    )
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    ax.legend(facecolor='#1a1a2e', labelcolor='white',
              fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "FTP_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  SECTION 7d — SOLVER DIFFICULTY PLOT (adjacent face jumps)
# ================================================================

def _plot_solver_difficulty(all_results):
    # Select the 60×400 mesh for comparison
    target_mesh = '60x400'
    subset = [r for r in all_results if r['Mesh'] == target_mesh]
    if not subset:
        return

    # Load adjacency for target mesh
    adj_cache = os.path.join(ADJ_DIR, f"adj_Mesh_{target_mesh}.json")
    if not os.path.exists(adj_cache):
        print(f"  SKIP solver difficulty plot: no adjacency cache for {target_mesh}")
        return
    with open(adj_cache) as f:
        adj_pairs = [tuple(p) for p in json.load(f)]

    colors = ['#ff4444','#ff8844','#ffaa00','#ffff44',
              '#00ff88','#4488ff','#aa88ff','#ff44ff',
              '#44ffff','#ff4488','#88ff44','#4444ff',
              '#ff8844','#44ff88','#8844ff','#ff4444',
              '#88ffff','#ffff88']

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1a2e')

    for r, color in zip(subset, colors):
        flux  = r['_flux_faces']
        jumps = np.array([abs(flux[i] - flux[j])
                          for i, j in adj_pairs]) / 1000.0
        label = f"{r['Method']}_{r['MCRT_res']} (SSI={r['SSI_%']:.1f}%)"
        ax.hist(jumps, bins=80, alpha=0.45, density=True,
                label=label, color=color)

    ax.set_xlabel('Adjacent Face Flux Jump (kW/m²)', color='white', fontsize=11)
    ax.set_ylabel('Probability Density', color='white', fontsize=11)
    ax.set_title(
        f'Solver Difficulty: Flux Discontinuities Between Adjacent Faces '
        f'({target_mesh} mesh)\n'
        'Large jumps = solver must smooth artificial steps = slower convergence',
        color='white', fontsize=11, fontweight='bold'
    )
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    ax.legend(facecolor='#1a1a2e', labelcolor='white',
              fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "FTP_solver_difficulty.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
#  SECTION 8 — LAYMAN INTERPRETATION SUMMARY
# ================================================================

def _print_layman_summary(df_exp):
    print("\n" + "=" * 65)
    print("  LAYMAN INTERPRETATION")
    print("=" * 65)

    # Best and worst FTP scores overall
    best = df_exp.loc[df_exp['FTP_score'].idxmax()]
    worst = df_exp.loc[df_exp['FTP_score'].idxmin()]

    print(f"\n  Best  transfer: {best['Method']}_{best['MCRT_res']} "
          f"→ {best['Mesh']} mesh  (FTP = {best['FTP_score']:.3f})")
    print(f"  Worst transfer: {worst['Method']}_{worst['MCRT_res']} "
          f"→ {worst['Mesh']} mesh  (FTP = {worst['FTP_score']:.3f})")

    print("\n  Per-method summary (diagonal / matched resolution):")
    for method in METHOD_ORDER:
        sub = df_exp[(df_exp['Method'] == method) &
                     (df_exp['MCRT_res'] == df_exp['Mesh'])]
        if sub.empty:
            continue
        print(f"\n  {method}:")
        for _, row in sub.iterrows():
            ssi_v = ("smooth → easy for solver" if row['SSI_%'] < 10
                     else "moderate" if row['SSI_%'] < 20
                     else "STAIRCASE → hard for solver")
            ilr_v = ("minimal loss" if row['ILR_%'] < 20
                     else "moderate" if row['ILR_%'] < 50
                     else "HEAVY LOSS")
            print(f"    {row['MCRT_res']:>12}:  "
                  f"Energy={row['ETA_%']:.1f}%  "
                  f"Peak={row['PTA_%']:.0f}%  "
                  f"SSI={row['SSI_%']:.1f}% ({ssi_v})  "
                  f"Info loss={row['ILR_%']:.1f}% ({ilr_v})  "
                  f"FTP={row['FTP_score']:.3f}")

    print(f"\n  Full results table: {OUT_DIR}/FTP_all_results.csv")
    print(f"  FTP matrices:       {OUT_DIR}/FTP_matrix_<Method>.csv")
    print("=" * 65 + "\n")


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    run_all()