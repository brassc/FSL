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
- recorded in `tracking_doc.csv`


# Contour points extraction #
Takes .csv file of marked skull end points in voxel coordinates and extracts the contour on the BET brain mask between these points using openCV python library. 

Patients selected according to patient selection criteria, outlined below.

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
    - The centered horizontal contours are stored in `h_def_rot` and `h_ref_rot` and the data slice is returned.
5. Fit ellipse using least squares method in `fit_ellipse` function
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
        - Data slice is updated using `update_dataframe` function, adding the filtered data to `ellipse_h_def`, `ellipse_v_def`, `ellipse_h_ref`, `ellipse_v_ref`.
    - The data is then returned to main program as `ellipse_data`.
6. `ellipse_data` is appended to `transformed_df` DataFrame.

## Ellipse analysis ##
Analysis of ellipse parameters is completed in script `longitudinal_data.py`. This imports the `ellipse_data.csv` data, converts from wide to long format, creates global plots of `h_optimal` for both deformed and reference configurations grouped by patient ID and also grouped by timepoint. 



# DTI Processing #
Path: `/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/19978/ultra-fast/Hour_00034.8016-Date_20111024/U-ID22691/nipype/DATASINK/DTIspace/dwi_proc/`

Freesurfer location: `/home/cmb247/Desktop/Project_3/Freesurfer/`









