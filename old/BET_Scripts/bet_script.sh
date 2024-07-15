#!/bin/bash

# Define input parameters
patientID="14324"
input_image="T1w_trio_P00030_14324_acute_20080801_U-ID15035.nii.gz"
output_image="$patientID.acute_bet.nii.gz"
bet_params="-f 0.0 -B"
bet_mesh_params="-e"

# Log the command
echo "Running bet with parameters: $bet_params" > bet_log.txt
echo "bet $input_image $output_image $bet_params" >> bet_log.txt

# Execute the command and append output to the log
bet $input_image $output_image $bet_params >> bet_log.txt 2>&1
bet $output_image $output_image $bet_mesh_params >> bet_log.txt 2>&1

chmod +x bet_script.sh


