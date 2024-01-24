#!/bin/bash

# Define input parameters
patientID="16577"
run_descriptor="A"
input_image="$patientID.$run_descriptor.DILATEDPOSTCROPacute_bet.nii.gz"
fslmaths_area_dim="-roi 0 -1 202 -1 0 -1 0 1" #frontal portion of brain image 0 176 120 120 0 256
frontal_part_mask="$run_descriptor.frontal_part_mask"
frontal_brain_mask="$run_descriptor.frontal_brain_mask"
dilated_frontal_brain_mask="$run_descriptor.frontal_dilated_brain_mask"
rest_part_mask="$run_descriptor.rest_part_mask"
rest_brain_mask="$run_descriptor.rest_brain_mask"
dil2_frontal_brain_mask="$run_descriptor.dil2_frontal_brain_mask"
dil2_frontal_brain_mask_binary="$run_descriptor.dil2_fbm_binary"
rest_brain_mask_binary="$run_descriptor.rest_bm_binary"
combined_mask="$run_descriptor.front_comb_mask"
final_brain_mask="$run_descriptor.front_final_brain_mask"
upper_brain_mask="upper_brain_mask" #original cropped T1w image
output_image="$patientID.$run_descriptor.SUPERDILATEDPOSTCROPacute_bet.nii.gz"


log_location="$run_descriptor.super_frontal_dilate_log.txt"

# Log the command
echo "Dilating frontal portion of brain using fslmaths: $fslmaths_area_dim" > $log_location

# Execute the command and append output to the log
fslmaths $input_image $fslmaths_area_dim $frontal_part_mask >> $log_location
fslmaths $input_image -mul $frontal_part_mask $frontal_brain_mask >> $log_location
fslmaths $frontal_brain_mask -dilF $dilated_frontal_brain_mask >> $log_location
# Step 4: Recombine with the rest of the original brain mask
# First, create an inverse of the frontal_part_mask to mask out the frontal part
fslmaths $frontal_part_mask -binv $rest_part_mask

# Apply the rest_part_mask to the original_mask to get the non-frontal part
fslmaths $input_image -mul $rest_part_mask $rest_brain_mask
#dilate frontal portion again
fslmaths $dilated_frontal_brain_mask -dilF $dil2_frontal_brain_mask >> $log_location
# Combine the dilated frontal part with the rest part
fslmaths $dil2_frontal_brain_mask -bin $dil2_frontal_brain_mask_binary
fslmaths $rest_brain_mask -bin $rest_brain_mask_binary
fslmaths $dil2_frontal_brain_mask_binary -add $rest_brain_mask_binary $combined_mask
fslmaths $combined_mask -thr 0.5 -bin $final_brain_mask
fslmaths $final_brain_mask -mul $upper_brain_mask $output_image


chmod +x dilate_front_script.sh



