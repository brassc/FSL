This BET_Scripts folder contains scripts to manipulate BET. This includes removing lower portion of a scan, dilating a BET mask by region etc. Uses fslmaths and bet algorithms by FMRIB. 

**Conventional MRI work on cluster:**

1. ./t1mulreg_script.sh ‘all subsequent T1 scans were registered to the first T1 scan for each patient’. 
    - #RUN FROM "BET_Extractions" DIRECTORY LEVEL
    - #THIS SCRIPT DYNAMICALLY SELECTS THE EARLIEST T1 SCAN (TIME 1) IN THE 'OG_Scans' DIRECTORY AND REGISTERS ALL SUBSEQUENT T1 TO THAT 'EARLIEST' SCAN.

2. brain extraction on image by image basis using **betnoneck.sh** as bet_{patient_timepoint}_{bet_params} and bet_mask_{patient_timepoint}_{bet_params}
    - crop neck, perform bet, create binary mask, record this to log file ~repos/FSL/bet_log.txt
    - optional: regional dilation of mask using ‘dilate_region_script.sh’
    - Note: directories given as absolute path so can be run directly from repo
