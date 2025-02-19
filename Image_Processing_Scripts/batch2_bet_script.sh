#!/bin/bash
#Load modules
module load fsl
moule load freesurfer

# Function to display usage
usage() {
    echo "Usage: $0 -g GUPI -h HOUR_NUMBER -b BET_PARAMS [-c NECK_CUT]"
    echo "  -g GUPI    : Patient ID"
    echo "  -h HOUR_NUMBER     : Hour number (e.g., 00376)"
    echo "  -b BET_PARAMS    : BET parameters (e.g., '-f 0.70 -R')"
    echo "  -c NECK_CUT      : (Optional) z voxel dimension to crop out (e.g. 56)"
    exit 1
}

# Parse input arguments
while getopts ":g:h:b:c:" opt; do
    case $opt in
        g) patient_id="$OPTARG" ;;
        h) timepoint="$OPTARG" ;;
        b) bet_params="$OPTARG" ;;
        c) neck_cut="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check mandatory parameters
if [ -z "$patient_id" ] || [ -z "$timepoint" ] || [ -z "$bet_params" ]; then
    usage
fi


# Set default neck cut if not provided
if [ -z "$neck_cut" ]; then
    neck_cut=0
fi


# remove spaces so bet_params can be used in filename
bet_params_filename=$(echo "$bet_params" | tr ' ' '_') 

## Define remaining input parameters
input_directory="/rds-d5/user/cmb247/hpc-work/Feb2025_working/${patient_id}/bias_corr/"
input_basename="initialise"
log_file="/rds-d5/user/cmb247/hpc-work/Feb2025_working/bias_bet_reg_log.txt"



# Function to perform BET with neck
perform_bet_with_neck() {
    echo "Performing BET..." 
    bet $input_image $output_image $bet_params
    echo "BET complete"
    fslmaths $output_image -bin $output_mask
    echo "$patient_id, $timepoint, $bet_params, biascorr=YES" >> $log_file
}


# Function to perform BET with neck crop (default value is 0, neck not cropped without -c argument in function call). 
perform_bet_and_crop_neck() {
    # 1. Cut neck
    echo "crop neck using fslmaths -roi..."
    # select lower portion of brain image i.e. neck
    fslmaths_crop_dim="-roi 0 -1 0 -1 0 $neck_cut 0 1" 
    fslmaths $input_image $fslmaths_crop_dim $lower_part_mask || exit 1
    # invert to get upper part of image
    fslmaths $lower_part_mask -binv $upper_part_mask || exit 1
    # multiply original image with upper mask to get upper brain
    fslmaths $input_image -mul $upper_part_mask $upper_brain || exit 1
    echo "fslroi neck crop complete, neck cut: $neck_cut"
    echo "Performing BET on cropped image..." 
    bet $upper_brain $output_image $bet_params || exit 1
    echo "BET complete"
    fslmaths $output_image -bin $output_mask || exit 1
    # 4. Delete neckcut image
    echo "Deleting temp files..."
    rm $lower_part_mask $upper_part_mask $upper_brain
}

# Function to write or update log
write_log() {
    log_entry="$patient_id    $timepoint    $bet_params"
    # Append neck_cut only if it is not equal to 0
    [ "$neck_cut" -ne 0 ] && log_entry+="    crop $neck_cut"

    if grep -q "^$patient_id    $timepoint" "$log_file"; then
        # Entry exists, update it
        sed -i "/^$patient_id[[:space:]]\+$timepoint/c\\$log_entry" "$log_file"
    else
        # Entry does not exist, append it
        echo "$log_entry" >> "$log_file"
    fi
}


## MAIN SCRIPT EXECUTION

# Select input_basename based on timepoint
input_basename=$(basename $(ls ${input_directory}T1_bias-Hour-${timepoint}*.nii.gz))


echo "Final input basename is $input_basename"
input_basename_without_extension="${input_basename%.nii.gz}"
input_image="${input_directory}${input_basename}"
echo "Input image: $input_image"

#fsleyes $input_image

# Define output parameters
GUPI_dir=$(dirname "${input_directory}")
output_directory="$GUPI_dir/BET_Output/"
echo "output dir: $output_directory"




if [ $neck_cut -eq 0 ]; then
    output_basename="${input_basename_without_extension}_bet_rbc_${bet_params_filename}.nii.gz"
    mask_output_basename="${input_basename_without_extension}_bet_mask_rbc_${bet_params_filename}.nii.gz"
else
    output_basename="${input_basename_without_extension}_bet_rbc_${bet_params_filename}_cropped_$neck_cut.nii.gz"
    mask_output_basename="${input_basename_without_extension}_bet_mask_rbc_${bet_params_filename}_cropped_$neck_cut.nii.gz"
fi
    
output_image="${output_directory}${output_basename}"
output_mask="${output_directory}${mask_output_basename}"



# cropping variables
lower_part_mask="${output_directory}lower_part_mask.nii.gz"
upper_part_mask="${output_directory}upper_part_mask.nii.gz"
upper_brain="${output_directory}upper_brain.nii.gz"


# Verify input image
if [ ! -f "$input_image" ]; then
    echo "Error: input_image ${input_image} does not exist."
    exit 1
fi



echo "Checking if output file exists..."



# Check output directory exists, if not make it
mkdir -p "$output_directory" 

## Check if output file already exists
if [ -f "$output_image" ]; then
    echo "Output file ${output_image} already exists. Skipping BET."
    echo "Writing to log..."
    write_log #user defined function
    echo "Opening in fsleyes..."
    fsleyes $input_image $output_image $output_mask
    exit 0
else
    "Output file $output_image does not exist. Proceeding with BET"
fi


# BET
perform_bet_and_crop_neck

echo "Writing to log..."
write_log

# View with fsleyes
echo "Opening in fsleyes..."
fsleyes $input_image $output_image $output_mask






