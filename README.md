# Supplementary Material for manuscript "Bin-Resolution-Independent Radiative Flux Mapping and Assessment for Cylindrical Solar Receivers."

## By Sainath A. Waghmare, Gabriel Cardona, and Ramón Pujol-Nadal

# Contents of the repository

## 1. `PTC_design`
Folder containing the PTC designs.

- **`70_1740_23040000_Simple.FCStd`**:
PTC geometry file constructed in FreeCAD.
- **`Spectral_materials.zip`**:
Spectral optical material files for all four components, in OTSunWebApp format:
  - Reflector (mirror): `T_Ag_E1` (Ag-BK7)
  - Absorber: `thinfilm_absorber_TiAlN_TiAlON_Si3N4` (TiAlN-TiAlON-Si₃N₄)
  - Glass envelope: `T_Glass` (BK7)
  - Anti-reflective coating: `AR1` (MgF₂)
- **`ASTMG173-direct.xlsx`**:
Spectral direct normal irradiance from ASTM G173-03, 290–3990 nm at 20 nm intervals, used for per-ray energy weighting.

To reproduce the ray-hit dataset, upload `70_1740_23040000_Simple.FCStd` and `Spectral_materials.zip` to [OTSunApp](https://otsunapp.uib.es/OTSunApp/), set incident angle = 0, the desired Nλ, and CSR = 0.05, and obtain the ray data file (`Th_points_absorber`) per Nλ.

---

## 2. `MCRT_RayHit_Data`
Folder containing the raw ray-hit output files from OTSun spectral MCRT simulations of the LS-3 receiver, at solar incidence angle = 0°, Buie circumsolar model CSR = 0.05, DNI = 900.07 W/m². Each file contains absorbed ray records: position (x, y, z), surface normal, wavelength, and attenuation factor (ε) per ray.

Contains 1 folder, **`Delta_20_10000`** (Nλ = 10000), with the following files:

- **`source_wavelengths.txt`**: MCRT input data.
- **`Th_integral_spectrum.txt`**: Power to absorber from source, input irradiance, and optical efficiency.
- **`Th_points_absorber_D_20_R_10000.txt`**: Ray dataset. (Provided as `Th_points_absorber_D_20_R_10000.txt.gz`, compressed with gzip.)
- **`Th_spectral_efficiency.txt`**: Spectral optical efficiency data.

**Note:** Ray-hit data of only the working ray count (Nλ = 10,000, N_rays = 1,118,896) is provided. All other ray-hit data files used in the study (Nλ = 500 to 100,000) are too large and will be available from the corresponding author on request. These files can also be regenerated using `70_1740_23040000_Simple.FCStd` with OTSun.

---

## 3. `Flux_Reconstruction`
Folder containing Python scripts implementing the four flux reconstruction methods. All scripts read from `MCRT_RayHit_Data` and write standardized CSV output files with four columns: `CircumferentialAngle_deg`, `LongitudinalAxis_m`, `CellArea_m2`, `HeatFlux_W_m2`.

Edit the `OUTPUT_DIR`, `HITS_FILE`, and `rays_per_wavelength` variables at the top of each script before running.

- **`BB_Flux_Rec.py`**: Bin-based histogram reconstruction code.
Exports `BB_<NθxNs>_<Nλ>.csv` for each resolution and ray count.
- **`k-NN_Flux_Rec.py`**: k-Nearest Neighbor reconstruction code.
Exports `KNN_<NθxNs>_<Nλ>.csv` for each resolution and ray count.
- **`KDE_Flux_Rec.py`**: Gaussian KDE reconstruction code.
Exports `KDE_<NθxNs>_<Nλ>.csv` for each resolution and ray count.
- **`FFTKDE_Flux_Rec.py`**: Fast Fourier Transform KDE reconstruction code.
Exports `FFTKDE_<NθxNs>_<Nλ>.csv` for each resolution and ray count.

The output of the `Flux_Reconstruction` scripts provides:

- Average flux (q_avg) — *Figure 3*
- Power to absorber (E_MCRT) — *Figure 3*
- Maximum flux (q_max) — *Figure 4(a), Figure 6(a), Figure 7(a) and Figure 5* (by varying the k value in `k-NN_Flux_Rec.py`)
- Noise and RI (%) — *Figure 4(b), Figure 6(b), Figure 7(b)*
- Flux contour plots for the cylindrical and unrolled surface — *Figure 8(a), (b), (c)*
- Reconstruction time (seconds) — *Table 5*

Run each script individually to obtain the above results for a given method; for example, to reproduce this list for the BB method at 20×20 resolution with Nλ = 2000, run `BB_Flux_Rec.py` and provide the data accordingly. Run all four scripts to obtain the results needed to plot Figures 3 to 8.

**Dependencies:** `numpy`, `pandas`, `scipy`, `KDEpy`, `matplotlib`

```
pip install KDEpy --break-system-packages
```

**Note:** Reconstruction time (seconds) may vary depending on the computer configuration and the number of CPU cores/units.

---

## 4. `Flux_Rec_Plot_Data`
Folder containing Excel files with the raw data behind all figures in the manuscript, enabling reproduction of each figure in Excel or matplotlib without re-running the reconstruction scripts.

- **`Energy_Avg_Flux.xlsx`** — Data to generate Figure 3
- **`BB_Plot_data.xlsx`** — Data to generate Figure 4
- **`k-NN_Plot_data.xlsx`** — Data to generate Figures 5 and 6
- **`KDE_Plot_data.xlsx`** — Data to generate Figure 7
- **`Validation.csv`** — Data to generate Figure 4

---

## 5. `CFD_Meshes`
Folder containing the OpenFOAM `polyMesh` directories for the six receiver tube outer-surface meshes, generated in Salome and converted with `ideasUnvToFoam`.

- **`20_20.unv`**
- **`80_80.unv`**
- **`300_300.unv`**
- **`60_400.unv`**
- **`80_600.unv`**
- **`100_1000.unv`**
- **`Mesh_Conversion_info.txt`**: Steps and script used to convert the meshes to OpenFOAM `polyMesh` format.

---

## 6. `FTP_Analysis`
Folder containing Python scripts implementing the Flux Transfer Performance (FTP) assessment.

- **`CSV/`**: Contains three subfolders (`BB`, `kNN`, `KDE`) with the output `.csv` files obtained from `Flux_Reconstruction` for the 60×400 grid.
- **`ftp_analysis.py`**: Main analysis script. Reads all 18 source CSV files (6 MCRT resolutions × 3 methods) and evaluates all 108 combinations (18 × 6 CFD meshes), computing five FTP metrics per combination. Outputs `FTP_all_results.csv`, `FTP_matrix_BB/kNN/KDE.csv`, and four PNG figures.
- **`plot_ftp_matrix.py`**: Standalone script to regenerate the FTP heatmap figures from the CSV matrix files, for users who do not wish to rerun the full analysis. Style (font, colours, DPI) is fully editable.

Data files:

- **`FTP_all_results.csv`** — 108 rows, all metric values; Figure 9
- **`FTP_matrix_BB.csv`** — 6×6 heatmap, BB method; Figure 10, Tables 3 and 4
- **`FTP_matrix_kNN.csv`** — 6×6 heatmap, k-NN method; Figure 10, Tables 3 and 4
- **`FTP_matrix_KDE.csv`** — 6×6 heatmap, KDE method; Figure 10, Tables 3 and 4
- **`plot_data_solver_difficulty.csv`** — Figure 11
- **`plot_data_metrics_summary.csv`** — Table 4, Figure 9
- FTP Results for RunA, RunB, and RunC and the three additional MCRT realizations used for uncertainty quantification (S9) are available on request. 


To run `ftp_analysis.py`:

- Provide all 18 CSV files for BB, k-NN and KDE, at resolutions 20×20, 80×80, 300×300, 60×400, 80×600 and 100×1000, in the `CSV` folder.
- Provide all 6 mesh files from the `CFD_Meshes` folder.
- Manually set the total absorbed power from MCRT, `Q_MAX_PHYSICAL_REFERENCE` (or keep the default value in the code).
- Edit the paths in the USER SETTINGS block (Section 1 of the script) to point to your CSV and mesh directories.

---

## 7. `Supplementary_Material`
Folder containing supplementary information.

- **`Supplementary_Information.pdf`**
