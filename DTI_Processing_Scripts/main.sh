#!/bin/bash

module load fsl

# Define the priority of keywords in an array
keywords=("ultra-fast" "fast" "acute" "3mo" "6mo" "12mo" "24mo")

# Define patient list
subdirectories=("19978" "12519")
# "13198" "13782" "13990" "14324" "16754" "19344" "19575" "19978" "19981" "20174" "20651" "20942" "21221" "22725" "22785" "23348")

# get location of processed DWI images
# define the base directory
base_dir="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi"

for sub in "${subdirectories[@]}"; do
    echo "Processing subject $sub..."

    for timepoint in "${keywords[@]}"; do

        # find the DTIspace dir
        nipype_dir=$(find "$base_dir/$sub/$timepoint/" -type d -name "nipype" 2>/dev/null | head -n 1)

        # Check it exists
        if [[ -d "$nipype_dir" ]]; then
           #echo "nipype dir for subject $sub: $nipype_dir"
           DTIspace_dir="$nipype_dir/DATASINK/DTIspace/"
           #echo "DTIspace_dir: $DTIspace_dir"
           
           DTI_corr_scan="$DTIspace_dir/dwi_proc/DTI_corrected.nii.gz"
           DTI_bval="$DTIspace_dir/dwi_proc/DTI_corrected.bval"
           DTI_bvec="$DTIspace_dir/dwi_proc/DTI_corrected.bvec"
           DTI_mask="$DTIspace_dir/masks/ANTS_T1_brain_mask.nii.gz"

           dtifitWLS_FA="$DTIspace_dir/dti/dtifitWLS_FA.nii.gz"
           dtifitWLS_MD="$DTIspace_dir/dti/dtifitWLS_MD.nii.gz"
           dtifitWLS_L1="$DTIspace_dir/dti/dtifitWLS_L1.nii.gz"
           dtifitWLS_L2="$DTIspace_dir/dti/dtifitWLS_L2.nii.gz"
           dtifitWLS_L3="$DTIspace_dir/dti/dtifitWLS_L3.nii.gz"
           dtifitWLS_MO="$DTIspace_dir/dti/dtifitWLS_MO.nii.gz"
           dtifitWLS_S0="$DTIspace_dir/dti/dtifitWLS_S0.nii.gz"
           dtifitWLS_V1="$DTIspace_dir/dti/dtifitWLS_V1.nii.gz"
           dtifitWLS_V2="$DTIspace_dir/dti/dtifitWLS_V2.nii.gz"
           dtifitWLS_V3="$DTIspace_dir/dti/dtifitWLS_V3.nii.gz"
           powermap="$DTIspace_dir/dti/powermap.nii.gz"
           traceMap_b1000="$DTIspace_dir/dti/traceMap_b1000.nii.gz"

           
           save_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/$sub/dti_reg"
           if [ ! -d $save_dir ]; then
               mkdir "${save_dir}"
           fi
           dtibet="$save_dir/betdti.nii.gz"

           
           #use this mask to do brain extraction on DTI_corr_scan. Save to new place. 
           fslmaths $DTI_corr_scan -mul $DTI_mask $dtibet






           # registration (after dtifit)

           t1_scan_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/$sub/T1w_time1_bias_corr_registered_scans/BET_Output/"
           #search scan directory for scan name containing timepoint
           t1_scan=$(find $t1_scan_dir -type f -name "*$timepoint*.nii.gz")

           flirt -in $dtibet -ref $t1_scan -out "$save_dir/dtibet_reg.nii.gz" -omat "$save_dir/dtibet_reg.mat"
           flirt -in $dtifitWLS_FA -ref $t1_scan -out "$save_dir/dtifitWLS_FA_reg.nii.gz"
           flirt -in $dtifitWLS_MD -ref $t1_scan -out "$save_dir/dtifitWLS_MD_reg.nii.gz"
           # further reg not required for now
           #flirt -in $dtifitWLS_L1 -ref $t1_scan -out "$save_dir/dtifitWLS_L1_reg.nii.gz"
        fi
    done
done
