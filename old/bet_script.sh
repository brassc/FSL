#!/bin/bash

module load fsl

# Define input parameters
patientID="12519"
folder_loc="${patientID}/MNI152_registered_scans"
input_image="${folder_loc}/mni152.T1w_trio_P00030_12519_acute_20070427_U-ID11556_registered.nii.gz"
file_name=$(basename "$input_image")
output_image="${folder_loc}/BET/T1BET_$file_name.nii.gz"
bet_params="-f 0.4 -g -0.15 -R"
bet_params_without_spaces=${bet_params// /}
# output_image="${output_image}_${bet_params_without_spaces}"
bet_mesh_params="-e"


# Define the priority of keywords in an array
keywords=("ultra-fast" "fast" "acute" "3mo" "6mo" "12mo" "24mo")

# Iterate through the list of keywords
for keyword in "${keywords[@]}"; do
  if [[ $file_name == *"$keyword"* ]]; then
    extracted_keyword="${filename%%$keyword*}$keyword"
    echo "Extracted keyword: $extracted_keyword"
  fi
done

log_file="${folder_loc}/BET/bet_log_$extracted_keyword.txt"

# Log the command
echo "Running bet with parameters: $bet_params" >> $log_file
echo "bet $input_image $output_image $bet_params" >> $log_file

# Execute the command and append output to the log
bet $input_image $output_image $bet_params >> $log_file 2>&1
# bet $output_image $output_image $bet_mesh_params >> $log_file 2>&1

chmod +x bet_script.sh


