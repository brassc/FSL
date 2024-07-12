# Image Processing Scripts #
This folder contains bet script for processing original images. This processing pipeline includes bias field correction using FSL fast, then registering each patient scan to first time point available using FLIRT. 

Note: absolute paths specified so runs from repo location.


Pipeline:
1. FSL fast with -B option to bias correct raw images
- output bias corrected images to directory `OG_Scans_bias_corr`. 
- log file (`bias_correction_log.txt`) located at `BET_Extractions` directory level
- checks log file to see whether item has already been processed. i.e. once running do not delete
- bash script: `step1biascorr.sh`

2. FSL flirt to register bias corrected images to first time point available for each patient. First time point selected dynamically. 
- takes `*restore.nii.gz` files from `OG_Scans_bias_corr` directory. 
- output registered images to `T1_time1_bias_corr_registered_scans` directory.
- checks output directory to see if `*$timepoint*registered.nii.gz` file is already present before proceeding with flirt
- log file (`bias_reg_log.txt`) located at `BET_Extractions` directory level
- bash script: `step2reg.sh`

3. FSL bet. Extract brain, create mask. Manual, case by case basis. 
- Takes output from `T1_time1_bias_corr_registered_scans` directory.
- log file for bet params (`bias_bet_reg_log.txt`) located at `BET_Extractions` directory level
- bash script: `step3bet.sh`
- implementation: `./step3bet.sh -p <patient_id> -t <timepoint>  -b '<bet parameters e.g. -f 0.55'' -c <crop voxel dimn>`

4. Check bet output. If not satisfactory, run manual modification script.
- Create `segtocut.nii.gz` file - this is a binary mask of region to be removed. Do in ITKSNAP or similar.
- output modified mask.nii.gz and modified bet.nii.gz with original basename & `maskmodified` or `modified` appended.
- bash script: `manualeditbet.sh` 
- implementation: `./manualeditbet.sh -p <patient id> -t <timepoint> -f <segmented area to be removed .nii.gz file>`


