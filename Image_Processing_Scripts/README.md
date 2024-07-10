# Image Processing Scripts #
This folder contains bet script for processing original images. This processing pipeline includes bias field correction using FSL fast, then registering each patient scan to first time point available using FLIRT. 

Note: absolute paths specified so runs from repo location.


Pipeline:
1. FSL fast with -B option to bias correct raw images
- output bias corrected images to directory `OG_Scans_bias_corr`. 
- log file (`bias_correction_log.txt`) located at `BET_Extractions` directory level
- bash script: functrial1.sh

2. FSL flirt to register bias corrected images to first time point available for each patient. First time point selected dynamically. 
- takes `*restore.nii.gz` files from `OG_Scans_bias_corr` directory. 
- output registered images to `T1_time1_bias_corr_registered_scans` directory.
- log file (`bias_reg_log.txt`) located at `BET_Extractions` directory level
- bash script: t1mulregbiascorr.sh

3. FSL bet. Extract brain, create mask. Manual, case by case basis. 
- Takes output from `T1_time1_bias_corr_registered_scans` directory.
- log file for bet params (`bias_bet_reg_log.txt`) located at `BET_Extractions` directory level
- bash script: betregbiascorr.sh
