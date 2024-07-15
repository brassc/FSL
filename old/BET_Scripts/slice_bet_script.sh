#!/bin/bash
module load fsl

# THIS SCRIPT PERFORMS BET EXTRACTION SUCH THAT SLICE Z 145 LOOKS GOOD FOR PATIENT 19978 TIME ACUTE
# 1) NECK CROPPING 2) BET EXTRACTION 3) RESTORE ORIGINAL DIMENSIONS 4) EXTRACT BINARY MASK. 

# Set paths
input_image="/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_acute_20111102_U-ID22791_registered.nii.gz"
#input_image="/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_fast_20111027_U-ID22723_registered.nii.gz"
directory='/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/'
output_image="${directory}betslice_T1w_time1.T1w_verio_P00030_19978_acute_20111102_U-ID22791_registered.nii.gz"
input_cut="${directory}cut_T1w_time1.T1w_verio_P00030_19978_acute_20111102_U-ID22791_registered.nii.gz"
patient_id='19978'
patient_timepoint='acute'

# BET before slicing
# cut neck off
cut_above='56'
bet_params='-f 0.5 -R'
bet_p=$(echo "$bet_params" | tr -d ' ')
fslroi $input_image $input_cut 0 -1 0 -1 $cut_above -1
bet ${input_cut} ${directory}${patient_timepoint}_bet${bet_p} $bet_params


# add neck blank box back in
# Determine dimensions of the original and cut images to calculate the removed slices
original_dim_z=$(fslval $input_image dim3)
bet_dim_z=$(fslval ${directory}${patient_timepoint}_bet${bet_p} dim3)
# Assume ${directory}/bet${bet_p} is your BET processed image
x_dim=$(fslval ${directory}${patient_timepoint}_bet${bet_p} dim1)
y_dim=$(fslval ${directory}${patient_timepoint}_bet${bet_p} dim2)
x_pixdim=$(fslval ${directory}${patient_timepoint}_bet${bet_p} pixdim1)
y_pixdim=$(fslval ${directory}${patient_timepoint}_bet${bet_p} pixdim2)
z_pixdim=$(fslval ${directory}${patient_timepoint}_bet${bet_p} pixdim3)


# Calculate the number of slices removed during the cut
removed_slices=$((original_dim_z - bet_dim_z + 1)) #add 1, was one too few before
echo $removed_slices

# Create an empty image with the same dimensions as the cut-off part
# Create the empty image
fslcreatehd $x_dim $y_dim $removed_slices 1 $x_pixdim $y_pixdim $z_pixdim 1 0 0 0 16 ${directory}empty.nii.gz


# Concatenate the empty image back to the BET output along the Z-axis
fslmerge -z ${directory}${patient_timepoint}_restored_bet${bet_p} ${directory}empty ${directory}${patient_timepoint}_bet${bet_p}


# Create binary mask
fslmaths ${directory}${patient_timepoint}_restored_bet${bet_p} -bin ${directory}${patient_timepoint}_restored_bet_mask${bet_p}

echo "${directory}${patient_timepoint}_restored_bet_mask${bet_p}"


# Extract one slice
#slice_to_extract='89' #'145-$cut_above
#fslroi ${directory}bet$bet_p ${directory}slice${bet_p}.nii.gz 0 -1 0 -1 $slice_to_extract 1

fsleyes "${directory}${patient_timepoint}_restored_bet_mask${bet_p}" "${directory}${patient_timepoint}_bet${bet_p}" "${input_image}"


chmod +x slice_bet_script.sh

