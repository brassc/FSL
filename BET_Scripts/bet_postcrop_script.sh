#!/bin/bash

# Define input parameters
patientID="16577"
input_image="upper_brain_mask.nii.gz" # for cropped input to bet
output_image="$patientID.POSTCROPacute_bet.nii.gz"
#bet_params="-f 0.0 -g -0.5 -B"
bet_params="-f 0.0 -g -0.5 -R"
bet_mesh_params="-e"

# Log the command
echo "Running bet with parameters: $bet_params" > bet_postcrop_log.txt
echo "bet $input_image $output_image $bet_params" >> bet_postcrop_log.txt

# Execute the command and append output to the log

bet $input_image $output_image $bet_params >> bet_postcrop_log.txt 2>&1
bet $output_image $output_image $bet_mesh_params >> bet_postcrop_log.txt 2>&1

chmod +x bet_postcrop_script.sh







