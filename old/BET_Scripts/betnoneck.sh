#!/bin/bash
#Load modules
module load fsl

# Define input params
patient_id="19978"
directory="/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/"
timepoint="fast"
input_basename="T1w_time1.T1w_verio_P00030_19978_fast_20111027_U-ID22723_registered.nii.gz"

input_image="${directory}${input_basename}"
bet_params="-f 0.70 -R"
bet_params_filename=$(echo "$bet_params" | tr ' ' '_') # remove spaces so bet_params can be used in filename

neck_cut='56'

# Define output params
neckcut_basename="neckcut_${timepoint}_${input_basename}"
neckcut_image="${directory}${neckcut_basename}"

biascorrect_basename="biascorrect_${patient_id}_${timepoint}"
biascorrect_image="${directory}$biascorrect_basename"

output_basename="bet_${timepoint}_${bet_params_filename}"
mask_output_basename="bet_mask_$timepoint_$bet_params_filename"
output_image="${directory}${output_basename}"
output_mask="${directory}${mask_output_basename}"

# Define BET log file
log_file_basename="bet_log_file.txt"
log_dir="/home/cmb247/repos/FSL/"
log_file="${log_dir}${log_file_basename}"

# BET without neck:

## EITHER THIS:
# 1. Cut neck
#echo "crop neck fslroi..."
#fslroi $input_image $neckcut_image 0 -1 0 -1 $neck_cut -1
#echo "fslroi neck crop complete"
# 2. Perform bet
#bet $neckcut_image $output_image $bet_params
# 3. Create mask
#fslmaths $output_image -bin $output_mask
# 4. Delete neckcut image
#rm $neckcut_image
# 5. Check output
#fsleyes $output_mask $output_image $input_image
# 6. Write to log file bet params
#echo "$patient_id, $timepoint, $bet_params" >> $log_file

## OR THIS:
# 1. get bias corrected image as ${biascorrect_image}/T1_biascorr.nii.gz
echo "bias correction: $biascorrect_image"
if [ -d "$biascorrect_image.anat" ]; then
    echo "Directory exists: $biascorrect_image"
    echo "bias correction already complete."
else
    echo "performing fsl_anat pipeline bias correction..."
    fsl_anat -i $input_image -o $biascorrect_image --noreorient --noreg --nononlinreg --noseg --nosubcortseg
    echo "fsl_anat pipeline bias correction complete."
fi
# 2. BET
echo "performing bet..." 
bet "${biascorrect_image}.anat/T1_biascorr.nii.gz" $output_image $bet_params
echo "BET complete"
# 3. Create mask
fslmaths $output_image -bin $output_mask
# 3. Check output
fsleyes $input_image $output_mask $output_image
# 4. write to log file
echo "$patient_id, $timepoint, $bet_params, biascorr=YES" >> $log_file






chmod +x betnoneck.sh
