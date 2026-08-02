# Supplementary Material for manuscript "Characterization of Circumferential Flux Distributions in Parabolic Trough Receivers Using Adaptive Gaussian Representations"

## By Sainath A. Waghmare, Gabriel Cardona, and Ramón Pujol-Nadal**

# Contents of the repository

## 1. `PTC_design`
Folder containing the PTC designs.

- **`PTC_Create_Scene.py`**: 
Script for generating the geometries of the Parabolic Trough Collector, based on six geometric parameters, for its analysis in OTSunApp.
- **`Wavelength-independent_material.zip`**: 
Optical materials that are independent of wavelength (constant property) for the constructed PTC designs (in OTSunWebApp file format) for the simulations.
- **`Wavelength-dependent_material.zip`**: 
Optical materials that are independent of wavelength (variable property) for the constructed PTC designs (in OTSunWebApp file format) for the simulations.

---

## 2. `Spectral_Analysis_Results`
Folder containing multiple folders which contain results files obtained from MCRT spectral analysis on OTSun and a Python code to estimate flux distribution. The results need to be arranged in ascending order for wavelengths. *“4.1 Ray Tracing”*

- **`Flux_Distribution_code.py`**: 
Python script for calculating flux distribution on the tubular receiver. The inputs are `ASTMG173.csv` and OTSun output files.
- **`ASTMG173.csv`**: Spectral direct solar radiation extracted from the main file `ASTMG173-direct.txt` from 290 to 3990 nm with step 20nm.

Folders:

- **`No_of_Rays_Simulation`** — Output of the simulation for number of rays  
- **`CSR_Vary`** — Output of the simulation for varying CSR values  
- **`Constant_material_Validation`** — Output of the simulation for a wavelength-independent material  
- **`Variable_material_Validation`** — Output of the simulation for a wavelength-dependent material  
- **`Receiver_Diameter`** — Output of the simulation for varying receiver diameter  
- **`Receiver_Height`** — Output of the simulation for varying receiver height  
- **`Rim_Angle`** — Output of the simulation for variable rim angles  

---

### 3. `Flux_Distribution_Results`
Folder containing Python code for the Gaussian curve fitting algorithm and results files (receiver circumferential angle vs LCR) obtained from `Flux_Distribution_code` and OTSun output files. *“5 Result and discussion”*

- **`Gaussian_Characterization_Algorithm`**: Script used to generate Gaussian curves for receiver circumferential angle vs LCR results. The inputs are flux distribution results. *“Figures 3 and 4”*

Data files:

- **`P_No_of_Rays.csv`** — Data to generate Figure 5  
- **`P_CSR_vary.csv`** — Data to generate Figure 7  
- **`P_Constant_Material_PTC.csv`** — Data to generate Figure 8(a)  
- **`P_Variable_Material_PTC.csv`** — Data to generate Figure 8(b)  
- **`P_Variable_Receiver_Diameter_LCR.csv`** — Data to generate Figure 9  
- **`P_Variable_Receiver_Height_LCR`** — Data to generate Figures 10(a) & 10(b)  
- **`P_Variable_Rim_Angle_LCR.csv`** — Data to generate Figures 11(a) & 11(b)  

---

## 4. `Data_From_Literature`
Folder containing files related to data from previous researchers for validation.  
The WebPlotDigitizer tool (https://automeris.io/) is used to extract results from already published literature.

- **`Altarawneh_et_al_Point_Source.csv`** — Data taken from Altarawneh et al., 2020, for validation in Figure 8(a)  
- **`Altarawneh_et_al_Sun_Shape.csv`** — Data taken from Altarawneh et al., 2020, for validation in Figure 8(a)  
- **`Jeter_et_al_Data_Sun_Shape.csv`** — Data taken from Jeter et al., 1986, for validation in Figures 3, 4, 6, and 8(a)  
- **`Rodriguez-Sanchez_and_Rosengarten.csv`** — Data taken from Rodriguez-Sanchez and Rosengarten., 2024, for validation in Figure 8(b) 

## 5. `Supplementary_Material`
Folder containing files of supplementary information and values of k (Gaussian parameters)

- **`Supplementary_Information.pdf`**: Figure S1 for spectral irradiance modelling using ASTM G173-03. Tables S1 and S2 for statistical indicator values for Jeter's curve at 0◦ incidence angle.
- **`Parameter_Values.xlsx`**: Gaussian parameters, k used in this characterization study to generate Figures: 6, 7, 8, 9, 10 and 11. 


---

# References

- **Rodriguez-Sanchez, D.; Rosengarten, G.**  *Optical efficiency of parabolic troughs with a secondary flat reflector; effects of non-ideal primary mirrors.*  Energy 288 (2024) 129521.

- **Altarawneh, I.; Batiha, M.; Rawadieh, S.; Alnaief, M.; Tarawneh, M.**  *Solar desalination under concentrated solar flux and reduced pressure conditions.*  Solar Energy 206 (2020) 983–996.

- **Jeter, S. M.**  *Calculation of the concentrated flux density distribution in parabolic trough collectors by a semifinite formulation.*    Solar Energy 37(5) (1986) 335–345.
