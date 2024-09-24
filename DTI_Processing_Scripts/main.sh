#!/bin/bash

module load fsl

# Define the priority of keywords in an array
keywords=("ultra-fast" "fast" "acute" "3mo" "6mo" "12mo" "24mo")

# Define patient list
subdirectories=("12519" "13198" "13782" "13990" "14324" "16754" "19344" "19575" "19978" "19981" "20174" "20651" "20942" "21221" "22725" "22785" "23348")

# get location of processed DWI images
# define the base directory
base_dir="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi"

for sub in "${subdirectories[@]}"; do
    echo "Processing subject $sub..."

    for timepoint in "${keywords[@]}"; do

        # find the DTIspace dir
        nipype_dir=$(find "$base_dir/$sub/$timepoint/" -type d -name "nipype" 2>/dev/null | head -n 1)

        # Check if the nipype directory was found
        if [ -z "$nipype_dir" ]; then
            echo "Warning: 'nipype' directory not found for subject $sub at timepoint $timepoint"
            continue  # skip this timepoint and continue with the next
        fi

        # Check it exists
        if [[ -d "$nipype_dir" ]]; then
           
           DTIspace_dir="$nipype_dir/DATASINK/DTIspace/"
           #echo "DTIspace_dir: $DTIspace_dir"
           
           
           DTI_corr_scan="${DTIspace_dir}dwi_proc/DTI_corrected.nii.gz"
           DTI_bval="${DTIspace_dir}dwi_proc/DTI_corrected.bval"
           DTI_bvec="${DTIspace_dir}dwi_proc/DTI_corrected.bvec"
           DTI_mask="${DTIspace_dir}masks/ANTS_T1_brain_mask.nii.gz"
           #echo "DTI_corr_scan location: $DTI_corr_scan"
	   
           dtifitWLS_FA="${DTIspace_dir}dti/dtifitWLS_FA.nii.gz"
           dtifitWLS_MD="${DTIspace_dir}dti/dtifitWLS_MD.nii.gz"
           dtifitWLS_L1="${DTIspace_dir}dti/dtifitWLS_L1.nii.gz"
           dtifitWLS_L2="${DTIspace_dir}dti/dtifitWLS_L2.nii.gz"
           dtifitWLS_L3="${DTIspace_dir}dti/dtifitWLS_L3.nii.gz"
           dtifitWLS_MO="${DTIspace_dir}dti/dtifitWLS_MO.nii.gz"
           dtifitWLS_S0="${DTIspace_dir}dti/dtifitWLS_S0.nii.gz"
           dtifitWLS_V1="${DTIspace_dir}dti/dtifitWLS_V1.nii.gz"
           dtifitWLS_V2="${DTIspace_dir}dti/dtifitWLS_V2.nii.gz"
           dtifitWLS_V3="${DTIspace_dir}dti/dtifitWLS_V3.nii.gz"
           powermap="${DTIspace_dir}dti/powermap.nii.gz"
           traceMap_b1000="${DTIspace_dir}dti/traceMap_b1000.nii.gz"

           #echo "dtifitWLS_FA location: $dtifitWLS_FA"
           
           save_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/$sub/dti_reg/"
           if [ ! -d $save_dir ]; then
               mkdir "${save_dir}"
           fi
           dtibet="${save_dir}betdti_${timepoint}_notreg.nii.gz"
           dtibet3d="${save_dir}dtibet3d_$timepoint.nii.gz"
           

           echo "doing bet extraction for $sub $timepoint on corrected dti scan..."
           #use this mask to do brain extraction on DTI_corr_scan. Save to new place. 
           fslmaths $DTI_corr_scan -mul $DTI_mask $dtibet
           echo "bet complete."



           

           # registration (after dtifit)
           echo "registering to t1 scan..."
           t1_scan_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/$sub/T1w_time1_bias_corr_registered_scans/BET_Output/"
           #search scan directory for scan name containing timepoint
           echo "finding t1 scan..."
           # search preferentially for bet that has been manually modified
           # check fast first: i.e. protect against ultra-fast 
           if [ $timepoint == "fast" ]; then
               t1_scan=$(find $t1_scan_dir -type f -name "*$timepoint*modified*.nii.gz" ! -name "*mask*" ! -name "*segto*" ! -name "ultra")
               if [ -z "$t1_scan" ]; then
                   t1_scan=$(find $t1_scan_dir -type f -name "*$timepoint*.nii.gz" ! -name "*mask*.nii.gz" ! -name "*segto*" ! -name "ultra")
               fi
           else
               t1_scan=$(find $t1_scan_dir -type f -name "*$timepoint*modified*.nii.gz" ! -name "*mask*.nii.gz" ! -name "*segto*")
               # If no file is found containing "modified", search again with just the timepoint
               if [ -z "$t1_scan" ]; then
                   t1_scan=$(find $t1_scan_dir -type f -name "*$timepoint*.nii.gz" ! -name "*mask*.nii.gz" ! -name "*segto*")
               fi
           fi
           

           # Check if the scan was found
           if [[ -n "$t1_scan" ]]; then
               echo "Scan found: $t1_scan"
           else
               echo "No T1 scan file found for timepoint $timepoint."
           fi




           ## MASK SEARCH

           # search preferentially for bet mask that has been manually modified
           # check fast first: i.e. protect against ultra-fast 
           if [ $timepoint == "fast" ]; then
               t1_mask=$(find $t1_scan_dir -type f -name "*$timepoint*modified*mask*.nii.gz" ! -name "*segto*" ! -name "ultra")
               if [ -z "$t1_mask" ]; then
                   t1_mask=$(find $t1_scan_dir -type f -name "*$timepoint*mask*.nii.gz" ! -name "*segto*" ! -name "ultra")
               fi
           else
               t1_mask=$(find $t1_scan_dir -type f -name "*$timepoint*modified*mask*.nii.gz" ! -name "*segto*")
               # If no file is found containing "modified", search again with just the timepoint
               if [ -z "$t1_mask" ]; then
                   t1_mask=$(find $t1_scan_dir -type f -name "*$timepoint*mask*.nii.gz" ! -name "*segto*")
               fi
           fi

           
           # Check if the mask was found
           if [[ -n "$t1_mask" ]]; then
               echo "Mask found: $t1_mask"
           else
               echo "No T1 mask file found for timepoint $timepoint."
           fi

           dti_corr_reg="${save_dir}dtireg_nobet_$timepoint.nii.gz"
           
           echo "registering bet for $sub $timepoint..."
           dtibet_reg="${save_dir}dtibet_reg_$timepoint.nii.gz"
           dtiregmat="${save_dir}dtibet_reg_$timepoint.mat"
           flirt -in "$dtibet" -ref "$t1_scan" -out $dtibet_reg  -omat "$dtiregmat"
           echo "Complete."
           #echo "Applying this registration matrix to dti_corrected prebet image..."
           #flirt -in "$DTI_corr_scan" -ref "$t1_scan" -applyxfm -init "$dtiregmat" -out "${save_dir}DTI_corr_scan_reg_$timepoint.nii.gz" 
           #echo "Complete."


           # inverse transform t1_mask to dti space using the dtibet -> t1bet omat
           echo "transforming t1 mask to corrected dti native space..."
           t1maskdtispace="${save_dir}t1_mask_in_dti_space_$timepoint.nii.gz"
           # get inverse of transform
           dtiregmatinv="${save_dir}dtiregmatinv_${timepoint}.mat"
           convert_xfm -omat $dtiregmatinv -inverse $dtiregmat
           flirt -in "$t1_mask" -ref "$DTI_corr_scan" -applyxfm -init "$dtiregmatinv" -out "$t1maskdtispace"
           fslmaths "$t1maskdtispace" -bin "$t1maskdtispace"
           echo "Complete."
           
           echo "Performing dtifit..."
           dtifitdir="${save_dir}/dtifitdir"
           if [[ ! -d $dtifitdir ]]; then
               mkdir -p $dtifitdir
           fi
           dtifit -k "$DTI_corr_scan" -o "$dtifitdir/dtifit_$timepoint" -m "$t1maskdtispace" -r "$DTI_bvec" -b "$DTI_bval" --save_tensor --wls
           echo "dtifit for $sub $timepoint completed. "
           
           echo "registering dtifit FA and MD to T1..."
           flirt -in "$dtifitdir/dtifit_${timepoint}_FA.nii.gz" -ref "$t1_scan" -out "$dtifitdir/dtifit_${timepoint}_reg_FA.nii.gz"
           flirt -in "$dtifitdir/dtifit_${timepoint}_MD.nii.gz" -ref "$t1_scan" -out "$dtifitdir/dtifit_${timepoint}_reg_MD.nii.gz"
        fi

    done
echo "Subject $sub dti extraction complete."
done
echo "End of program."
