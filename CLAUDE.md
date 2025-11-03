# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains imaging processing pipelines for analyzing decompressive craniectomy patients using structural MRI (T1) and diffusion tensor imaging (DTI). The analysis focuses on quantifying brain herniation through contour analysis and investigating white matter microstructural changes using DTI metrics (FA, MD).

**Key Cohorts:**
- LEGACY cohort: Decompressive craniectomy patients (numeric IDs)
- CENTER cohort: Decompressive craniectomy patients (alphanumeric IDs)
- Both cohorts are combined for analysis purposes
- Data locations vary by cohort (see DTI Processing section)

## Environment Setup

### Python Environment
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `nibabel` for NIfTI file handling
- `opencv-python` for contour detection
- `matplotlib`, `seaborn` for visualization
- `scipy`, `statsmodels`, `pymer4` for statistical analysis
- `rpy2` for R integration (LME models)
- `neuroCombat` for scanner harmonization
- `pandas`, `numpy` for data manipulation

### FSL and Cluster Usage
- Scripts are designed to run on Cambridge HPC (SLURM scheduler)
- FSL tools required: `fsl`, `bet`, `flirt`, `fast`, `fslmaths`
- Submit jobs using: `sbatch sbatch.sh`
- Example SLURM script in `sbatch.sh` shows ROI extraction job configuration

### R Requirements
Some statistical analyses require R packages:
- `lme4`, `lmerTest` for linear mixed effects models
- `emmeans` for marginal means
- `pbkrtest` for p-value corrections
- `PMCMRplus` for Jonckheere-Terpstra trend tests

## Code Architecture

### 1. Structural MRI Processing Pipeline (`Image_Processing_Scripts/`)

**Brain extraction workflow (run sequentially):**
```bash
# 1. Bias field correction
./Image_Processing_Scripts/step1biascorr.sh

# 2. Register to first timepoint
./Image_Processing_Scripts/step2reg.sh

# 3. Brain extraction (manual per patient/timepoint)
./Image_Processing_Scripts/step3bet.sh -p <patient_id> -t <timepoint> -b '<bet params>' -c <crop_voxels>

# 4. Manual refinement (optional)
./Image_Processing_Scripts/manualeditbet.sh -p <patient_id> -t <timepoint> -f <segtocut.nii.gz>
./Image_Processing_Scripts/manualeditbet.sh -p <patient_id> -t <timepoint> -a <segtoadd.nii.gz>
```

**Contour extraction and analysis workflow (run sequentially after BET):**
```bash
# 1. Extract contours from brain masks
python Image_Processing_Scripts/contour_plot_main.py [<patient_id>]

# 2. Fit ellipses to contours
python Image_Processing_Scripts/ellipse_plot_main.py

# 3. Analyze ellipse goodness of fit
python Image_Processing_Scripts/ellipse_fit_analysis.py

# 4. Calculate herniation areas
python Image_Processing_Scripts/area_main.py

# 5. Longitudinal area analysis
python Image_Processing_Scripts/longitudinal_main.py

# 6. Statistical analysis (area changes over time)
python Image_Processing_Scripts/stats_main.py

# 7. Ellipse-based area analysis
python Image_Processing_Scripts/area_main_ellipse.py

# 8. Final statistical analysis with mixed effects
python Image_Processing_Scripts/stats_main_h.py
```

**Key data files:**
- `included_patient_info.csv`: Patient selection, skull endpoints, timepoint inclusion/exclusion
- `data_entries.csv`: Extracted contour points (deformed, baseline, reflected)
- `ellipse_data.csv`: Fitted ellipse parameters and transformed coordinates
- `area_data.csv`: Calculated herniation areas

### 2. DTI Processing Pipeline (`DTI_Processing_Scripts/`)

**Data extraction and registration:**
```bash
# Extract and register DTI to T1 space
./DTI_Processing_Scripts/main.sh
```

**Region of Interest (ROI) analysis:**
```bash
# Main ROI extraction with concentric spheres
./DTI_Processing_Scripts/roi_main_newcoords.sh --num_bins=<N> --bin_size=<voxels> --filter_fa_values=<true/false> --overwrite=<true/false>

# Create white matter specific ROIs
./DTI_Processing_Scripts/roi_WM_segmentation.sh

# Extract WM metrics
./DTI_Processing_Scripts/roi_extract_wm.sh
```

**Common configurations:**
- `--num_bins=5 --bin_size=4`: 5 rings of 4 voxel radius each (uses `_NEW` suffix)
- `--num_bins=10 --bin_size=2`: 10 rings of 2 voxel radius each (uses `_NEW` suffix)
- Coordinates from: `DTI_Processing_Scripts/LEGACY_DTI_coords_fully_manual.csv`

**DTI harmonization and analysis:**
```python
# Harmonize scanner effects using neuroCombat
python DTI_Processing_Scripts/harmonisation.py

# Plotting and statistical analysis
python DTI_Processing_Scripts/dti_results_plotting_main.py
```

**Output locations:**
- Individual ROIs: `/home/cmb247/rds/hpc-work/April2025_DWI/<patient_id>/<timepoint>/roi_files_<num_bins>x<bin_size>vox/`
- WM ROIs: Same path + `/WM_mask_DTI_space/`
- CSV results: `DTI_Processing_Scripts/results/`
- Plots: `DTI_Processing_Scripts/dti_plots/`

### 3. Utility Scripts (`python_scripts/`)

Small helper scripts for data visualization and management:
- `main.py`: Creates patient timeline matrix visualization
- `data_filter.py`: Filters patients by scan availability
- `read_data.py`: CSV data loading utilities
- `plots.py`: Matrix plotting functions

## Important Implementation Details

### Contour Extraction Process
1. **Coordinate system**: Skull endpoints manually marked in ITK-SNAP at corpus callosum midline
2. **Points of interest**:
   - `(x1, y1)`: Anterior skull endpoint
   - `(x2, y2)`: Posterior skull endpoint
   - `(x3, y1)`: Anterior baseline (contralateral)
   - `(x4, y2)`: Posterior baseline (contralateral)
3. **Contour detection**: Uses OpenCV to extract brain surface between skull endpoints
4. **Trimming**: Removes contour overlap into falx cerebri using Euclidean distance-based endpoints
5. **Reflection**: Creates symmetric baseline by reflecting contralateral side across midline

### Ellipse Fitting Transformation
Ellipses fit to form: $y = h\sqrt{a^2 - x^2}$ (with optional skew parameter $b$)

**Coordinate transformation steps:**
1. Translate posterior skull point to origin $(0, 0)$
2. Rotate about origin so anterior point lies on $y=0$
   - Left craniectomy: anticlockwise rotation
   - Right craniectomy: clockwise rotation
3. Center horizontally about $x=0$ using mean of deformed contour
4. Resample points at regular intervals to avoid overfitting
5. Fit ellipse using `scipy.optimize.curve_fit` with bounded parameters

### DTI ROI Architecture
- **Concentric rings**: Start from manually picked anatomical coordinates, expand outward
- **Masking**: All ROIs constrained to brain mask (ANTS mask in DTI space)
- **WM specificity**: Optional multiplication with white matter segmentation from T1 (registered to DTI space via ANTs transformations)
- **Filtering**: FA values >1 or <0.05 replaced with NA (physically implausible)
- **File locking**: Master CSV uses locking for concurrent write safety

### Statistical Methods

**Mixed Effects Models:**
- Account for repeated measures (multiple timepoints per patient)
- Random intercepts for patient ID
- FDR correction (Benjamini-Hochberg) for multiple comparisons over FWER due to small sample size
- Implemented in R via `rpy2` for DTI analysis

**Timepoint Recategorization:**
Original timepoints consolidated based on days since injury:
- `ultra-fast`: 0-2 days
- `fast`: 2-8 days
- `acute`: 8-42 days
- `3-6mo`: 42-278 days (combines 3mo + 6mo)
- `12-24mo`: 278+ days (combines 12mo + 24mo)

**Trend Analysis:**
- Jonckheere-Terpstra test for ordered alternatives (increasing/decreasing trends)
- Used to detect FA/MD gradients across concentric rings from craniectomy site

### Data Imputation Policy
Missing data NOT imputed due to:
1. High sparsity would make results overly dependent on imputation method
2. Difficulty distinguishing between MCAR, MAR, and MNAR mechanisms
3. Potential for bias in MNAR cases (e.g., patient mortality related to herniation severity)

## Common Development Tasks

### Testing a Single Patient Analysis
```bash
# Contour extraction for specific patient
python Image_Processing_Scripts/contour_plot_main.py <patient_id>

# DTI ROI extraction for specific patient (modify roi_main_newcoords.sh to filter CSV)
```

### Adding a New Timepoint Category
1. Update `map_timepoint_to_string()` in `longitudinal_main.py`
2. Update `process_timepoint_data()` in `dti_results_plotting_main.py`
3. Modify timepoint ordering lists in statistical analysis scripts

### Modifying ROI Analysis Parameters
When changing `num_bins` or `bin_size`:
1. Scripts automatically add `_NEW` suffix for 5×4 or 10×2 configurations
2. Check file locking in `roi_extract.sh` and `roi_extract_wm.sh`
3. Update harmonization script input filenames accordingly

### Publication Plots
- Style set via `set_publication_style()` function (Times New Roman, 300 DPI)
- High-res outputs saved to: `../Thesis/phd-thesis-template-2.4/Chapter6/Figs/`
- Standard outputs in respective `*_plots/` directories

## Data Locations

**LEGACY cohort:**
- DTI: `/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/<patient_id>/<timepoint>/`
- Previously processed by Stefan Winzeck (see References)

**CENTER cohort:**
- MRI: `/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI/<patient_id>/`

**Processed outputs:**
- DTI ROIs: `/home/cmb247/rds/hpc-work/April2025_DWI/`
- Freesurfer: `/home/cmb247/Desktop/Project_3/Freesurfer/`

## Key Functions and Their Locations

**Contour extraction:**
- `load_nifti()`: `contour_plot_main.py` - Load NIfTI files with nibabel
- `auto_boundary_detect()`: Detects contours using OpenCV
- `trim_contours()`: Removes falx overlap regions
- `reflect_across_line()`: Creates symmetric baseline

**Ellipse fitting:**
- `transform_points()`: Translate to origin
- `rotate_points()`: Apply rotation matrix
- `center_points()`: Center about x=0
- `fit_ellipse()`: Main fitting orchestration
- `filter_fitted_values()`: Remove discontinuities and imaginary values

**DTI analysis:**
- `process_timepoint_data()`: Standardize timepoint categories
- `parameter_differences()`: Calculate control - craniectomy differences
- `create_timepoint_boxplot_LME_dti()`: Visualization with model overlays
- `jt_test()`: Jonckheere-Terpstra trend analysis

**Harmonization:**
- `process_metrics_file()`: Apply neuroCombat batch correction
- `average_rings()`: Average metrics across specified ring ranges
- Replaces FA < 0.0001 and MD < 0 with NaN

## References

[1] Winzeck, S. (2020). Methods for Data Management in Multi-Centre MRI Studies and Applications to Traumatic Brain Injury. https://doi.org/10.17863/CAM.71122
