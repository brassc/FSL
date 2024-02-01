#!/bin/bash

# Define input parameters
patientID="16577"
input_image="T1w_trio_P00273_16577_acute_20090126_U-ID16035.nii.gz"
#output_image="$patientID.acute_bet.nii.gz"
#bet_params="-f 0.0 -g -0.5 -R"
#bet_mesh_params="-e"
fslmaths_crop_dim="-roi 0 -1 0 -1 0 85 0 1" #lower portion of brain image
lower_part_mask="lower_part_mask.nii.gz"
lower_brain_mask="lower_brain_mask.nii.gz"
lower_ero="eroded_lower_brain_mask.nii.gz"
upper_part_mask="upper_part_mask.nii.gz"
upper_brain_mask="upper_brain_mask.nii.gz"
#input_image=$upper_brain_mask # for cropped input to bet


# Log the command
echo "Cropping the head using fslmaths: $fslmaths_crop_dim" > crop_log.txt
#echo "Running bet with parameters: $bet_params" > crop_log.txt
#echo "bet $input_image $output_image $bet_params" >> crop_log.txt

# Execute the command and append output to the log
fslmaths $input_image $fslmaths_crop_dim $lower_part_mask >> crop_log.txt
fslmaths $lower_part_mask -binv $upper_part_mask >> crop_bet_log.txt
#fslmaths $input_image -mul $lower_part_mask $lower_brain_mask >> crop_log.txt
fslmaths $input_image -mul $upper_part_mask $upper_brain_mask >> crop_log.txt

#$bet $input_image $output_image $bet_params >> crop_bet_log.txt 2>&1
#$bet $output_image $output_image $bet_mesh_params >> crop_bet_log.txt 2>&1

chmod +x crop_script.sh


