#!/bin/bash
#Load modules
module load fsl

## EXAMPLE USAGE: ./step3bet.sh -p 19978 -t fast -b "-f 0.70 -R"

# Function to display usage
usage() {
    echo "Usage: $0 -p PATIENT_ID -t TIMEPOINT -b BET_PARAMS"
    echo "  -p PATIENT_ID    : Patient ID"
    echo "  -t TIMEPOINT     : Timepoint (e.g., fast)"
    echo "  -b BET_PARAMS    : BET parameters (e.g., '-f 0.70 -R')"
    exit 1
}

# Parse input arguments
while getopts ":p:t:b:" opt; do
    case $opt in
        p) patient_id="$OPTARG" ;;
        t) timepoint="$OPTARG" ;;
        b) bet_params="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check mandatory parameters
if [ -z "$patient_id" ] || [ -z "$timepoint" ] || [ -z "$bet_params" ]; then
    usage
fi










## Define patient info input parameters
#patient_id="19978"
#timepoint="fast"

# Define BET parameters
#bet_params="-f 0.70 -R"
bet_params_filename=$(echo "$bet_params" | tr ' ' '_') # remove spaces so bet_params can be used in filename
neck_cut='56'

## Define remaining input parameters
input_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1w_time1_bias_corr_registered_scans/"
input_basename="initialise"
log_file="/home/cmb247/Desktop/Project_3/BET_Extractions/bias_bet_reg_log.txt"

# Select input_basename based on timepoint

# Function to find input scan basename based on timepoint
find_input_basename() {
    local input_basename
    local found=0
    
    
    while IFS= read -r file; do
        basename=$(basename "$file")
        if [[ "$basename" == "T1"* && "$basename" == *"$timepoint"* && "$basename" == *"restore_registered.nii.gz" ]]; then
            if [[ "$timepoint" == "fast" && "$basename" != *"ultra-fast"* ]]; then
                input_basename=$basename
                found=1
                echo "$input_basename"
                break
            elif [[ "$timepoint" != "fast" || "$basename" != *"ultra-fast"* ]]; then
                input_basename=$basename
                found=1
                echo "$input_basename"
                break
            fi
        fi
    done < <(find "$input_directory" -type f)

    if [ $found -eq 0 ]; then
        echo "Error: No matching file found for patient_id $patient_id and timepoint $timepoint."
        exit 1
    fi
}


# Function to perform BET with neck
perform_bet_with_neck() {
    echo "Performing BET..." 
    bet $input_image $output_image $bet_params
    echo "BET complete"
    fslmaths $output_image -bin $output_mask
    echo "$patient_id, $timepoint, $bet_params, biascorr=YES" >> $log_file
}





## MAIN SCRIPT EXECUTION
input_basename=$(find_input_basename)

echo "Final input basename is $input_basename"
input_basename_without_extension="${input_basename%.nii.gz}"
input_image="${input_directory}${input_basename}"
echo "Input image: $input_image"

# Define output parameters
output_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1w_time1_bias_corr_registered_scans/" # make sure to put / at end of directory
output_basename="${input_basename_without_extension}_bet_rbc_${bet_params_filename}.nii.gz"
mask_output_basename="${input_basename_without_extension}_bet_mask_rbc_${bet_params_filename}.nii.gz"
output_image="${output_directory}${output_basename}"
output_mask="${output_directory}${mask_output_basename}"


# Verify input image
if [ ! -f "$input_image" ]; then
    echo "Error: input_image ${input_image} does not exist."
    exit 1
fi

echo "Checking if output file exists..."

## CHECK IF OUTPUT FILE ALREADY EXISTS
if [ -f "$output_image" ]; then
    echo "Output file ${output_image} already exists. Skipping bet."
    exit 0
else
    "Output file $output_image does not exist. Proceeding with BET"
fi

# List files in the output directory for verification
echo "Listing files in output directory:"
ls -l "$output_directory"

## EITHER THIS: (BET without neck)
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


## OR THIS: (BET with neck)
# 2. BET
perform_bet_with_neck




chmod +x step3bet.sh
