#!/bin/bash
#Load modules
module load fsl

# Define input parameters
patient_id="19978"
input_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/19978/OG_Scans/" # make sure to put / at end of directory
timepoint="fast"
input_basename="T1w_verio_P00030_19978_fast_20111027_U-ID22723.nii.gz"
input_basename_without_extension="${input_basename%.*}"
input_image="${input_directory}${input_basename}"

# Define output parameters
output_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/19978/OG_Scans_bias_corr/" # make sure to put / at end of directory
output_basename="${input_basename_without_extension}_bias_corr.nii.gz"
output_image="${output_directory}${output_basename}"



# Verify input image
if [ ! -f "$input_image" ]; then
    echo "Error: input_image ${input_image} does not exist."
    exit 1
fi
echo "starting bias correction for input_image: $input_image..."

# create bias correction output directory OG_Scans_bias_corr
if [ -d $output_directory ]; then
    echo "Output directory ${output_directory} exists."
else
    mkdir "${output_directory}"
    echo "Output directory ${output_directory} created."
fi

# Do bias correction using FSL fast
echo "Performing bias correction now..."
fast -B -o $output_image $input_image

# Check if successful
if [ $? -eq 0 ]; then
    echo "Bias field correction completed successfully."
    echo "Output file basename: $output_basename"
else
    echo "Bias field correction failed."
    exit 1
fi





chmod +x trial1.sh
