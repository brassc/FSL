This repo contains commands and scripts used to process brain extraction for a cohort of 25 patients. There is no patient data contained in this repo. 

# Image Processing Scripts #
The `Image_Processing_Scripts` folder contains bets script for processing original images. This processing pipeline includes bias field correction using FSL fast, then registering each patient scan to first time point available using FLIRT. 

Note: absolute paths specified so runs from repo location.


### Pipeline: ###
1. **FSL fast with -B option to bias correct raw images**
- output bias corrected images to directory `OG_Scans_bias_corr`. 
- log file (`bias_correction_log.txt`) located at `BET_Extractions` directory level
- checks log file to see whether item has already been processed. i.e. once running do not delete
- bash script: `step1biascorr.sh`

2. **FSL flirt to register bias corrected images to first time point available for each patient.** First time point selected dynamically. 
- takes `*restore.nii.gz` files from `OG_Scans_bias_corr` directory. 
- output registered images to `T1_time1_bias_corr_registered_scans` directory.
- checks output directory to see if `*$timepoint*registered.nii.gz` file is already present before proceeding with flirt
- log file (`bias_reg_log.txt`) located at `BET_Extractions` directory level
- bash script: `step2reg.sh`

3. **FSL bet. Extract brain, create mask. Manual, case by case basis.** 
- Takes output from `T1_time1_bias_corr_registered_scans` directory.
- log file for bet params (`bias_bet_reg_log.txt`) located at `BET_Extractions` directory level
- bash script: `step3bet.sh`
- implementation: `./step3bet.sh -p <patient_id> -t <timepoint>  -b '<bet parameters e.g. -f 0.55'' -c <crop voxel dimn>`
- recorded in `tracking_doc.csv`

4. Check bet output. If not satisfactory, run [optional] manual modification script.
- Create `segtocut.nii.gz` file - this is a binary mask of region to be removed. Do in ITKSNAP or similar. OR
- Create `segtoadd.nii.gz` file - this is a binary mask of region to be added. Do in ITKSNAP or similar.
- outputs modified mask.nii.gz and modified bet.nii.gz with original basename & `maskmodified` or `modified` appended.
- bash script: `manualeditbet.sh` 
- implementation for cutting: `./manualeditbet.sh -p <patient id> -t <timepoint> -f <segmented area to be removed .nii.gz file>` 
- implementation for adding: `./manualeditbet.sh -p <patient id> -t <timepoint> -a <segmented area to be added .nii.gz file>`
- recorded in `bet_tracking_doc.csv`


# Contour points extraction #
Takes .csv file of marked skull end points in voxel coordinates and extracts the contour on the BET brain mask between these points using openCV python library. 

Patients selected according to patient selection criteria, outlined below.

To redo the analysis after changes in bet, the following should be ran:
- `contour_plot_main.py` (on cluster only)
- `ellipse_plot_main.py`
- `ellipse_fit_analysis.py`
- `area_main.py`
- `longitudinal_main.py`
- `stats_main.py`
- `area_main_ellipse.py`
- `stats_main_h.py`


### Patient selection criteria: ###
- 'Free' bulging i.e. no fluid accumulation, brain allowed to expand
- Not too much other pathology e.g. 13198 fast vs. 3 mo scan
- Easily identifiable skull 'end' points

**Patient ID, timepoint, skull end point coordinates for all eligible patients documented in** `included_patient_info.csv`

### Slice selection and skull point extraction process ###
1. Go to midline
2. Select high point on inferior side of corpus callosum arch
3. Select skull end points if visible. If not visible, exclude patient from analysis.
- `x1, y1` = anterior skull end point
- `x2, y2` = posterior skull end point
4. Use these y coordinates to pick point on the skull on the opposite side. Assuming the patient's head is quasisymmetrical, these form an estimation of where should the brain surface be. 
- `x3, y1` = anterior baseline skull point
- `x4, y2` = posterior baseline skull point


### Contour points extraction using contour_plot_main.py ###
This program extracts the contour line and creates an array of points that lie on this line. It works recursively per patient and timepoint, extracted from `included_patient_info.csv`. 

Usage: `python contour_plot_main.py <patient id>` or `python contour_plot_main.py` to default to all included patients. 

1. Import `included_patient_info.csv`
2. Filter according to exclusion flag (first column)
3. Iterate over each patient and timepoint to produce contour points array for deformed and baseline sides, and also a reflected array.

For each patient id and timepoint iteration, the steps are as follows:
1. Find bet image and mask dynamically according to patient id and timepoint
2. Load files using `load_nifti` bespoke function for each file of interest. `load_nifti` loads .nii.gz using `nibabel` library
3. Extract point and side information from filtered `included_patient_info.csv`
4. Extract bet image and mask slices according to recorded z coordinate using `extract_and_display_slice`
5. Extract contour from mask edge using `auto_boundary_detect` for each contour of interest i.e. twice per patient id and timepoint.
    - uses `flipside_func` to flip recorded side prior to calling `auto_boundary_detect` the second time for baseline
6. Trim contours where necessary e.g. where they overlap into the center falx fold. 
    - `find_contour_ends` function calculates pairwise Euclidean distances between all points on a contour, returns the indices of the maximum distance found
    - `trim_contours` function calls `find_contour_ends` function, identifies anterior and posterior ends of function by comparison to known/input skull end point values, adds point to trimmed array if it satisfies the condition of being on the correct side of supplied anterior or posterior end point x value, and is outside the area defined by the `threshold` parameter supplied to `trim_contours` function. Returns trimmed x and y points as two 1D numpy arrays. 
7. Create reflected contour - a reflection of the extracted (trimmed) baseline contour. Uses `get_mirror_line` function and `reflect_across_line` function. 
    - `get_mirror_line` uses first and last points extracted from contour to draw a line centrally located.  
    - `reflect_across_line` function handles both 1 and 2 dimensional reflections according to whether the mirror line extracted via `get_mirror_line` has a gradient or not. 
    - `trim_reflected` function removes any portion of reflected contour that is below the posterior skull end point y value.
8. Slice is plotted and saved to each patient's directory on the cluster. 
9. Local patient id, timepoint and corresponding contour points are saved to global dictionary `data_entries` using `add_data_entry` function. Key format is `data_entry_{patient_id}_{timepoint}`, key dynamically updates. 
10. `data_entries` is saved as `data_entries.csv` file. 


# Ellipse fitting #
Program `ellipse_plot_main.py` takes contour data stored in `data_entries.csv`, transforms each whole contour such that anterior and posterior points lie on $y=0$ and are symmetrical about $x=0$, fits and returns ellipse in `transformed_df` according to $y=h\sqrt(a^2-x^2)$ defined in `func` function. There is an option to include skew, defined in `funcb` function and represented by the $b$ parameter: $y=h\sqrt(a^2-(1+\frac{b}{a}x))$.

### Ellipse points extraction using ellipse_plot_main.py ###

1. Import data from `data_entries.csv` and import and extract side data from `included_patient_info.csv`. Create one pandas DataFrame. 
2. Convert contour arrays from pd.Series to numpy type, convert y to be horizontal and x to be vertical.
3. Create empty DataFrame `transformed_df`. This is mutable and ready to be accessed within the following for loop. 
4. Loop through each DataFrame line by line and extract the fitted ellipse data, added to each line as four new columns per ellipse fit, or eight in total (`h_def_rot`, `v_def_rot`, `ellipse_h_def`, `ellipse_v_def`, `h_ref_rot`, `v_ref_rot`, `ellipse_h_ref`, `ellipse_v_ref`). Append this new line to DataFrame `transformed_df`.
5. Save DataFrame as `ellipse_data.csv`. 

The loop in step 4 performs the following processes:
1. Get copy of line of DataFrame as slice containing only data for a patient ID at a specific timepoint. 
2. Transform data points contained in slice such that posterior point lies at $(0, 0)$ using `transform_points` function
    - `transform_points` takes copy of slice of DataFrame, creates four new columns to store translated contours in.
    - It then computes where the points should be moved to such that the last point in the contour lies at $0$ and applies it using a lambda function. 
    - These new points are stored in the four new columns created at beginning of function (`h_def_tr`, `v_def_tr`, `h_ref_tr`, `v_ref_tr`), and the new slice is returned. 
3. Rotate about $(0,0)$ so that anterior point lies on $y=0$ using `rotate_points` function
    - `rotate_points` takes the new slice from `transform_points` and creates four new columns to store rotated contours in in addition to an `angle` column. 
    - Angle is computed via trigonometry, direction adjusted according to which side the craniectomy is on. Note: L craniectomy rotates anticlockwise, R craniectomy rotates clockwise, both about $(0,0)$. 
    - Rotation angle is applied using a two dimensional rotation matrix for deformed and reference coordinates (pairs of contours) multiplied together using a lambda function. 
    - These new points are stored in the four new columns created at beginning of function (`h_def_rot`, `v_def_rot`, `h_ref_rot`, `v_ref_rot`), and this new slice is returned.
4. Center these points about $x=0$ using `center_points` function
    - `center_points` takes data slice from `rotate_points` function and computes average of horizontal deformed contour. 
    - This average is applied to both reference and deformed horizontal contour points to center about $x=0$.
    - The centered horizontal contours are stored in `h_def_cent` and `h_ref_cent`, new columns `v_def_cent` and `v_ref_cent` to store corresponding `v_def_rot` and `v_ref_rot` data in a cohesive naming convention are created.
    - The data slice is returned.
5. Resample and downsample - the contour lines for `def` and `ref` contours are reconstructed (linear interpolation) and the points cloud resampled at regular $x$ ($h$) intervals whilst preserving the end points. This avoids overfitting and clusters of points having increased influence over the ellipse that is fit. 
6. Fit ellipse using least squares method in `fit_ellipse` function
    - For each set of contours marked by `def` and `ref`, the `fit_ellipse` function takes a given data slice and returns a new data slice with added `ellipse_h_def`, `ellipse_v_def`, `ellipse_h_ref`, `ellipse_v_ref` columns as `ellipse_data`.
        - The columns are initialised using `initialize_columns` function. 
        - Data is fitted using `fit_data` function, returning `params` (an array of $h$ and $a$ values). 
            - `fit_data` gets initial guesses using `get_fit_params` function
                - `get_fit_params` returns starting estimates by extracting key data e.g. highest and lowest values of horizontal contour to estimate upper and lower bounds for $a$, uses `find_intersection_height` function to extract an estimate for $h$. 
                    - `find_intersection_height` function finds the height at which a linear interpolation between two h_coords either side of the y axis cuts the y axis.
            - `fit_data` returns $h$ and $a$ as `initial_guess` array, upper and lower bounds for $a$ as `bounds` array.
            - Curve fitting is performed using `curve_fit` from `scipy.optimize` python library. 
                - `curve_fit` takes `func`, a user-defined function defining the ellipse curve in square root form $y=h\sqrt(x^2-a^2)$, `p0=initial_guesses` and `bounds` and returns `params` and `covariance`. 
        - `params` is returned to `fit_ellipse` function, checked using `check_params`function
            - `check_params` prints the $h$, $a$ and optionally $b$ parameters according to the length of the `params` array
        - Data and `params` are used by `calculate_fitted_values` function to return `h_values`, a linear spaced array between minimum and maximum values in the horizontal contour, and `v_fitted`, where the ellipse function defined in `func` is used to calculate a corresponding vertical array from `h_values`. 
        - Due to the square root inherent to the ellipse formula, sometimes `func` would return imaginary values, represented as flat, horizontal trailing lines either side of the ellipse. These values were trimmed using `filter_fitted_values` function, ensuring to retain the starting and ending $0$ in the `v_fitted` array and corresponding `h_values` at the relevant index. 
            - To prevent discontinuities, a filter was also applied for the vertical direction. Regions of the ellipse with very high gradient discontinuous with the rest of the ellipse were found by evaluating the second derivative of the ellipse for all `x` values of the fitted ellipse curve. These regions were removed so that only a smooth region of the ellipse remained. 
        - Data slice is updated using `update_dataframe` function, adding the filtered data to `ellipse_h_def`, `ellipse_v_def`, `ellipse_h_ref`, `ellipse_v_ref`.
    - The data is then returned to main program as `ellipse_data`.
6. `ellipse_data` is appended to `transformed_df` DataFrame.

### Ellipse analysis ###
Analysis of ellipse parameters is completed in script `longitudinal_data.py`. This imports the `ellipse_data.csv` data, converts from wide to long format, creates global plots of `h_optimal` for both deformed and reference configurations grouped by patient ID and also grouped by timepoint. 

The program `ellipse_fit_analysis.py` evaluates the goodness of fit between contours and their corresponding fitted ellipse: 
- Comparitive plots of Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) are created in both reference and deformed configurations. 
- A scatter plot of Mean Absolute Error (MAE) against Dice area overlap coefficent is created. 
- A scatter plot of the area bound between two contours $\Delta A_c$ and area bound between two ellipses $\Delta A_e$ with line of best fit is created. 


##### Methods Summary

###### `calculate_overlap_metrics(contour_x, contour_y, ellipse_x, ellipse_y)`

Calculates area-based overlap metrics between contour points and fitted ellipse.

* **Input**: Coordinate arrays for contour points and ellipse points
* **Output**: Dictionary with areas, Dice coefficient, IoU, and area difference percentage
* **Method**: Uses mask-based approach for Dice calculation; computes polygon-based metrics when possible

###### `calculate_dice_from_masks(contour_x, contour_y, ellipse_x, ellipse_y)`

Specialized function for Dice coefficient calculation.

* **Input**: Coordinate arrays for contour points and ellipse points
* **Output**: Dice coefficient (float, 0-1 range)
* **Method**:
   1. Creates high-resolution grid (500 points)
   2. Sorts points by x-coordinate
   3. Uses linear interpolation between points (to match cubic voxel source)
   4. Generates binary masks by filling areas under curves
   5. Computes: 2 × intersection / (sum of areas)

For semi-elliptical shapes, this approach effectively captures area overlap even when one input is a sparse point cloud rather than a continuous shape.


# Polynomial fitting

# Area analysis
Area analysis is completed independently of any ellipse or polynomial fits. The script `area_main.py` imports `ellipse_data.csv` and uses reoriented, translated (but not rotated or centered) horizontal and vertical contours to find area underneath each set of points. Area is found using trapezium method of integration. Both points are plotted on a graph with the difference between deformed and baseline written as text. 

In `longitudinal_main.py`, the difference between reference and deformed area is plotted, grouped by patient and timepoint. 

Area change relative to first scan is also plotted (saved as `area_change_longitudinal.png`) as smooth splines for patients with more than 2 datapoints, and as a straight line for patients with 2 data points.  Original data first area is taken as datum 0, magnitude of other points is adjusted to reflect the new datum. A cubic spline is created from this offset data and plotted. 


## Statistical Analysis
Initial pilot data was analysed, subjects N = 12, total scans = 46. Initially, to account for between patient variability, a mixed effects model was trialled. However, due to significant sparsity in the pilot data, appropriate paired tests were conducted to discover relationships between categorical timepoints. 
Analysis is conducted in `stats_main.py`. 

### Note on Imputation
Missing data was not imputed due to the very high level of sparsity and the uncertainty surrounding missing data types. For example, a missing scan in between an early and a late san is likely missing completely at random (MCAR) data. However, missing scans at timepoints after any early - mid timepoint scans could be MCAR but could also be missing at random (MAR) or missing not at random (MNAR), e.g. if the patient dies. If their death was due to herniation (area of deformation measurement), then that would be MAR, if their death was due to any other or a combination of any other factors excluding area of deformation, that would be MNAR. 

MCAR timepoints were not imputed due to the proportion of data that the imputed values would represent. This proportion would be so great that the analysis results would be overly dependent on the type of imputation, devaluing the analysis results. 

For MAR and MNAR timepoints, these were not imputed due to the impossibility of distinguishing between the data types, the high likelihood of the data being MNAR and the difficulty in addressing MNAR in this particular dataset with high levels of sparsity. Imputation difficulties with MNAR data containing random effects include model overfitting and introducing bias or accidental data manipulation. The latter would stem from a lack of understanding of the dataset and any contextual variables that may or may not have been recorded or provided as covariates in the analysis. 

### Loading and Cleaning Data
In `stats_main.py`, data is loaded from `area_data.csv` and `patient_id`,`timepoint` and `area_diff` columns are retained for analysis. Since the `acute` timepoint was found to be of most interest as per the `area_change_longitudinal.png` visualisation, data was filtered such that all patients to be analysed had the `acute` timepoint. 

### Basic visualisation
A boxplot of herniation area over time was created to show distribution of data over time. for timepoints with less than 5 data points, only the median was plotted. 

### Pairwise Testing
For all of a patient's timepoints, paired t-tests (`ttest_rel`) were conducted using the `acute` timeline as a baseline for comparison. If there were not enough pairs for the paired t-test to work, an exception was introduced setting `stat_t` and `p_value_t` to be `np.nan`. 

Similarly, the Wilcoxon signed-rank test was performed. Due to sample size constraints, this is likely better suited to the dataset since it does not assume data is normally distributed. 

Results were stored in a DataFrame as `results_df`. The False Discovery Rate (FDR) correction was then applied using `multipletests` function from `statsmodels.stats.multitest` library with the `method` variable set to `fdr_bh`. 

This process was similarly applied for all other pairs not containing the `acute` timepoint, stored as `results_all_pairs_df`. 

#### Note about False Discovery Rate (FDR)
It is important to apply the False Discovery Rate (FDR) correction due to performing multiple statistical tests, which increases the likelihood of false positive errors. 
The FDR correction:
1. Sorts p-values in ascending order
2. Calculates adjusted p-values, $q_i$ using the formula $q_i = \frac{p_i \times m}{i}$ where $p_i$ is a sorted p-value, $m$ is the total number of tests and $i$ is the rank of the p-value from smallest to largest. 
3. Adjusted p-values $q_i$ are compared to a chosen significance level. In the `fdr_bh` method (Benjamin-Hochberg procedure) used by `statsmodels.stats.multitest.multipletests` python function, the default significance level, $\alpha$ is set to 0.05. 
    - If $q_i \leq \alpha$, null hypothesis $H_0$ is rejected for that test. 
    - $\alpha$ is the maximum proportion of false positives accepted in the analysis. 

FDR-BH procedure was selected over more conservative measures e.g. Family-Wise Error Rate (FWER) methods including Holm-Bonferroni. This is because it is a simple measure with lower risk of failing to detect true positives, whilst still controlling for false positives in a manner useful for this exploratory research. At pilot study stage, the more stringent FWER methods were found to be overly restrictive. 

#### Visualisations
Data visualisation conducted in `stats_main.py`. Number of pairwise comparisons available was shown in the `data_availability.png` matrix. Significance matrices for FDR corrected t-test (dubious given small sample size) (`significance_matrix.png`) and Wilcoxon signed-rank test both uncorrected `significance_matrix_wilcoxon_uncorrected.png` and corrected `significance_matrix_wilcoxon.png`. 

# DTI Processing #
Path: `/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/19978/ultra-fast/Hour_00034.8016-Date_20111024/U-ID22691/nipype/DATASINK/DTIspace/dwi_proc/`

The `main.sh` script located in `DTI_Processing_Scripts` folder extracts already processed dti images `DTI_corrected.nii.gz`, `dtifitWLS_FA.nii.gz`, `dtifitWLF_MD.nii.gz` from `patient/timepoint` folder location in the `DECOMPRESSION_Legacy` data.  This data was processed by Stefan Winzeck [1].

BET extraction is performed on `DTI_corrected.nii.gz` using `fslmaths` to multiply it with an ANTS mask, from same dataset. 

For each patient and timepoint, this BET extraction, FA and MD scans are registered to the corresponding T1 at the relevant timepoint. The T1 mask is then transformed to DTI space to avoid disturbing the DTI data. `$dtiregmatinv = ${save_dir}dtiregmatinv_${timepoint}.mat`.

## Region of Interest Extraction Pipeline

Region of interest analysis is completed using sequentially larger spheres around the end points of contour (manually picked coordinates).  These coordinates are recorded in `LEGACY_DTI_coords_fully_manual.csv`. 

### Overview
The pipeline consists of three main scripts:

1. `roi_main_new_coords.sh` - Main orchestration script that processes patients/timepoints and manages the overall workflow
2. `roi_create.sh` - Creates spherical ROIs around anatomical points of interest
3. `roi_extract.sh` - Extracts DTI metrics from the created ROIs
4. `roi_WM_segmentation.sh` - Registers white matter mask to DTI space and creates white matter specific ROIs
5. `roi_extract_wm.sh` - Extracts DTI metrics from the white matter specific ROIs

### Requirements

- FSL (FMRIB Software Library)
- ANTs (Advanced Normalization Tools)
- Bash shell environment
- Input DTI data including:
  - FA maps (dtifitWLS_FA.nii.gz)
  - MD maps (dtifitWLS_MD.nii.gz)
  - Brain masks (ANTS_T1_brain_mask.nii.gz)
  - WM masks in T1 space (tissueMap_WM.nii.gz)
  - ANTs transformation files (ants_rig_DTItoT1_InverseComposite.h5)
  - Coordinate CSV file with anatomical points of interest

### Usage

#### Main Script

```bash
./roi_main.sh --num_bins=<value> --bin_size=<value> --filter_fa_values=<true/false> --overwrite=<true/false>
```

### Input Data Structure
The scripts expect a specific directory structure:

For numeric patient IDs: `/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/<patient_id>/<timepoint>/`
For alphanumeric patient IDs: `/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI/<patient_id>/`

The coordinate CSV file should be located within the repo at `DTI_Processing_Scripts/LEGACY_DTI_coords_fully_manual.csv` and follow a specific format with anatomical and DTI space coordinates.

### Script Details 
#### `roi_main_newcoords.sh`
This script:

1. Parses command line arguments
2. Iterates through non-excluded patients in the coordinate CSV
3. Locates relevant DTI data (mask, FA, MD maps)
4. Calls roi_create.sh to create ROIs
5. Calls roi_extract.sh to extract metrics and output to individual and master CSV files

#### `roi_create.sh`

This script:

1. Creates output directories for ROIs
2. Retrieves anatomical coordinates from CSV
3. Creates point ROIs at specified coordinates
4. Creates concentric rings (spheres) around these points
5. Applies brain mask to constrain ROIs to brain tissue
6. Optimizes processing by reusing existing ROIs when possible


#### `roi_extract.sh`
This script:

1. Calculates mean FA and MD values for each ROI ring
2. Optionally filters FA values > 1 (which are physically implausible)
3. Outputs results to individual and master CSV files

#### `roi_WM_segmentation.sh`
This script:

1. Loads anatomical white matter masks from T1 space
2. Registers WM masks to DTI space using existing ANTs transformation (located at `T1space/ants_rig_DTItoT1_InverseComposite.h5` for LEGACY or `dwi/proc_set1_nobzero/nipype/DATASINK/T1space/ants_rig_DTItoT1_InverseComposite.h5` for CENTER subjects)
3. Binarizes the WM mask (all non-zero values become 1)
4. Creates WM-specific versions of existing ROIs by multiplying with the WM mask
5. Processes both FA and MD ROIs for all anatomical points and ring sizes
6. Stores results in a dedicated WM_mask_DTI_space subdirectory


#### `roi_extract_wm.sh`
This script:

1. Calculates mean FA and MD values for each white matter-specific ROI
2. Maintains the same filtering functionality as the original extraction script
3. Handles missing WM ROIs by inserting "NA" values
4. Outputs results to WM-specific individual and master CSV files
5. By default filters FA values between 0.05 and 1 (physically plausible range)



### Special Features

- `NEW` suffix: When using `num_bins=5` or `num_bins=10` with `bin_size=4`, the scripts use a `"_NEW"` suffix for output directories and files
- Filtering: Option to filter out implausible FA values (> 1)
- Overwrite control: Option to skip ROI creation if directories already exist
- Optimization: Intelligent reuse of existing ROIs when creating larger bins
- Concurrency safety: File locking when writing to master CSV
- White matter specificity: Option to isolate and analyse only white matter voxels within ROIs

### Output Files

- Individual ROI files: `/home/cmb247/rds/hpc-work/April2025_DWI/<patient_id>/<timepoint>/roi_files_<num_bins>x<bin_size>vox/`
- White matter ROI files: `/home/cmb247/rds/hpc-work/April2025_DWI/<patient_id>/<timepoint>/roi_files_<num_bins>x<bin_size>vox/WM_mask_DTI_space/`
- Individual CSV results: `DTI_Processing_Scripts/results/<patient_id>_<timepoint>_metrics_<num_bins>x<bin_size>vox.csv`
- White matter CSV results: `DTI_Processing_Scripts/results/<patient_id>_<timepoint>_metrics_<num_bins>x<bin_size>vox_wm.csv`
- Master CSV results: `DTI_Processing_Scripts/results/all_metrics_<num_bins>x<bin_size>vox.csv`
- White matter master CSV: `DTI_Processing_Scripts/results/all_metrics_<num_bins>x<bin_size>vox_wm.csv`

Note: When using special parameters or filtering, appropriate suffixes are added to file and directory names. Most current is with suffix `NEW_filtered`. 




[comment]: # The script `create_roi_native_space.sh` takes these files and gets FA and MD data within that space, saves it as a `.pkl` file (`{patient_id}_{timepoint}_dti_values.pkl`).  


## Harmonisation

After extracting DTI metrics from ROIs, it's often necessary to harmonize data across different scanners to account for scanner-related variability. This pipeline uses the neuroCombat package for batch effect correction.

### Overview

The harmonisation script:
1. Merges DTI metrics with scanner information
2. Applies the neuroCombat algorithm to correct for scanner-related batch effects
3. Outputs harmonized metrics for further analysis

### Requirements

- Python 3.6+
- Required packages:
 - numpy
 - pandas
 - matplotlib
 - neuroCombat

### Scanner Information

The script expects a CSV file containing scanner information with the following columns:
- `patient_id`: Unique identifier for each patient
- `timepoint`: Timepoint of the scan
- `Cohort`: Study cohort information
- `Site`: Scanner site location
- `Model`: Scanner model

### Array Averaging Function

The script now provides functionality to extract and average values from specific rings:

```python
average_rings(input_filename, output_filename, rings_to_average=[5, 6, 7])
```

This function averages DTI metrics across specified rings (e.g., rings 5, 6, and 7) for each metric-location combination, creating new columns with the format `{METRIC}_{LOCATION}_ring_{RINGS}_avg`.

### Harmonization Options

The `process_metrics_file()` function accepts a `mean_only` parameter:

```python
process_metrics_file(input_filename, harmonized_output_filename, mean_only=True)
```

- When `mean_only=True`: Standard processing for single-value metrics (default behavior).
- When `mean_only=False`: Extracts numeric values from array strings, calculates their means, and then applies harmonization to these means.

### Data Cleaning

- FA values below 0.0001 are replaced with NaN.
- MD values that are negative (physically impossible) are replaced with NaN.


### Usage

Run `python DTI_Processing_Scripts/harmonisation.py` to harmonise data. Data is output in `DTI_Processing_Scripts/` directory as CSV. 

Example workflow for array-based metrics:
```python
# Step 1: Average values from rings 5, 6, and 7
average_rings(
    input_filename='DTI_Processing_Scripts/results/all_metrics_10x4vox_all_values.csv',
    output_filename='DTI_Processing_Scripts/results/all_metrics_10x4vox_rings_5_6_7_mean.csv',
    rings_to_average=[5, 6, 7]
)

# Step 2: Harmonize the averaged data
process_metrics_file(
    input_filename='DTI_Processing_Scripts/results/all_metrics_10x4vox_rings_5_6_7_mean.csv',
    harmonized_output_filename='DTI_Processing_Scripts/merged_data_10x4vox_567_harmonised.csv'
)
```

Mean based metrics can use the `process_metrics_file` function directly. 


## DTI Plotting and Statistical Analysis

This module provides comprehensive plotting and statistical analysis functions for DTI (Diffusion Tensor Imaging) metrics, including visualization of fractional anisotropy (FA) and mean diffusivity (MD) across different brain regions and timepoints.

### Overview

The DTI plotting pipeline:
1. Processes timepoint data and standardizes categories
2. Calculates differences between control and craniectomy sides
3. Performs linear mixed effects (LME) modeling
4. Creates publication-ready visualizations
5. Conducts statistical tests including Jonckheere-Terpstra trend analysis

### Requirements

- Python 3.6+
- Required packages:
  - numpy, pandas, matplotlib, seaborn
  - scipy, statsmodels
  - rpy2 (for R integration)
  - R packages: lme4, emmeans, pbkrtest, lmerTest, PMCMRplus

### Key Functions

#### Data Processing
```python
process_timepoint_data(input_file_location)
```
Standardizes timepoint categories and sorts data by patient ID and timepoint order.

#### Difference Calculations
```python
parameter_differences(df)
```
Calculates differences between baseline (control) and actual (craniectomy) values for FA and MD parameters in anterior and posterior regions.

#### Visualization Functions

**Boxplot with LME Results:**
```python
create_timepoint_boxplot_LME_dti(df, parameter, result, timepoints, 
                                 fixed_effects_result=None, fixed_only=False)
```
Creates boxplots showing DTI differences across timepoints with overlaid LME model predictions and confidence intervals.

**Ring Analysis Plots:**
```python
plot_all_rings_combined(df, parameter, num_bins=5, save_path=None)
```
Visualizes all ring data over time with patient-specific colors and region-specific markers.

**Correlation Analysis:**
```python
create_area_predicts_fa_plot(df, result4, result5, show_combined=True)
create_area_predicts_md_plot(df, result4, result5, show_combined=True)
```
Creates scatter plots showing relationships between herniation area and DTI changes.

#### Statistical Testing

**Jonckheere-Terpstra Test:**
```python
jt_test(df, parameter='fa', regions=(2,10), alternative='increasing', 
        combine_regions=False)
```
Performs trend analysis across ring regions to detect systematic changes from craniectomy site.

**Data Availability Matrix:**
```python
data_availability_matrix(data, timepoints, diff_column='fa_anterior_diff')
```
Creates heatmap showing patient overlap across different timepoints.

### Timepoint Categories

The script automatically recategorizes timepoints based on days since injury:
- `ultra-fast`: 0-2 days
- `fast`: 2-8 days  
- `acute`: 8-42 days
- `3-6mo`: 42-278 days (combines 3mo and 6mo)
- `12-24mo`: 278+ days (combines 12mo and 24mo)

### Statistical Models

**Linear Mixed Effects Model:**

$Y_{ijk} = \beta_0 + \sum_{t=1}^{T-1} \beta_{1t} \cdot \text{Timepoint}_{jt} + \beta_2 \cdot \text{Region}_k + u_i + \varepsilon_{ijk}$

Where:
- $Y_{ijk}$: DTI difference (control - craniectomy) for subject $i$, timepoint $j$, region $k$
- $\beta_0$: Intercept (mean DTI difference at reference timepoint and region)
- $\beta_{1t}$: Fixed effect coefficient for timepoint $t$ (relative to reference)
- $\text{Timepoint}_{jt}$: Indicator variable (1 if timepoint $t$, 0 otherwise)
- $\beta_2$: Fixed effect coefficient for brain region
- $\text{Region}_k$: Indicator variable (0 = anterior, 1 = posterior)
- $u_i$: Random intercept for subject $i$, where $u_i \sim \mathcal{N}(0, \sigma_u^2)$
- $\varepsilon_{ijk}$: Residual error, where $\varepsilon_{ijk} \sim \mathcal{N}(0, \sigma^2)$

**Area Prediction Models:**

*Model 1 (Simple):*
$Y_{ij} = \alpha_0 + \alpha_1 \cdot \text{Area}_{ij} + v_i + \eta_{ij}$

*Model 2 (With timepoint control):*
$Y_{ij} = \alpha_0 + \alpha_1 \cdot \text{Area}_{ij} + \sum_{t=1}^{T-1} \alpha_{2t} \cdot \text{Timepoint}_{jt} + v_i + \eta_{ij}$

Where:
- $Y_{ij}$: DTI difference for subject $i$ at timepoint $j$
- $\alpha_0$: Intercept
- $\alpha_1$: Effect of herniation area on DTI changes (primary parameter of interest)
- $\text{Area}_{ij}$: Herniation area measurement
- $\alpha_{2t}$: Timepoint effects (Model 2 only)
- $v_i$: Random intercept for subject $i$, where $v_i \sim \mathcal{N}(0, \sigma_v^2)$
- $\eta_{ij}$: Residual error, where $\eta_{ij} \sim \mathcal{N}(0, \sigma^2)$

### Usage

```python
# Process harmonized data
data = process_timepoint_data('DTI_Processing_Scripts/merged_data_harmonised.csv')

# Calculate differences
data = parameter_differences(data)

# Create visualizations
create_timepoint_boxplot_LME_dti(data, 'fa', lme_result, 
                                timepoints=['ultra-fast', 'fast', 'acute', '3-6mo', '12-24mo'])

# Perform trend analysis
jt_results = jt_test(data, parameter='fa', regions=(2,10), alternative='increasing')
```

### Output

All plots are saved in:
- `DTI_Processing_Scripts/dti_plots/` (PNG format)
- `../Thesis/phd-thesis-template-2.4/Chapter6/Figs/` (high-resolution for publication)

The module generates publication-ready figures with proper statistical annotations, confidence intervals, and standardized formatting using the `set_publication_style()` function.




Freesurfer location: `/home/cmb247/Desktop/Project_3/Freesurfer/`


## References
[1] Winzeck, S. (2020). Methods for Data Management in Multi-Centre MRI Studies and Applications to Traumatic Brain Injury [Apollo - University of Cambridge Repository]. https://doi.org/10.17863/CAM.71122











