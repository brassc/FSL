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

4. **Check bet output. If not satisfactory, run (optional) manual modification script.**
- Create `segtocut.nii.gz` file - this is a binary mask of region to be removed. Do in ITKSNAP or similar.
- output modified mask.nii.gz and modified bet.nii.gz with original basename & `maskmodified` or `modified` appended.
- bash script: `manualeditbet.sh` 
- implementation: `./manualeditbet.sh -p <patient id> -t <timepoint> -f <segmented area to be removed .nii.gz file>`
- recorded in `tracking_doc.csv`


# Contour points extraction #
Takes .csv file of marked skull end points in voxel coordinates and extracts the contour on the BET brain mask between these points using openCV python library. 

Patients selected according to patient selection criteria, outlined below.

### Patient selection criteria: ###
- 'Free' bulging i.e. no fluid accumulation, brain allowed to expand
- Not too much other pathology e.g. 13198 fast vs. 3 mo scan
- Easily identifiable skull 'end' points

**Patient ID, timepoint, skull end point coordinates for all eligible patients documented in** `included_patient_info.csv`

### Patient slice selection process ###
1. Go to midline
2. Select high point on inferior side of corpus callosum arch
3. Select skull end points if visible. If not visible, exclude. 










