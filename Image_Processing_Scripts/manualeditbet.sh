#!/bin/bash
module load fsl

## EXAMPLE USAGE: ./manualeditbet.sh -p 12519 -t 6mo -f segtocut.nii.gz

# Function to display usage
usage() {
    echo "Usage: $0 -p PATIENT_ID -t TIMEPOINT -f SEGTOCUTFILE"
    echo "  -p PATIENT_ID    : Patient ID"
    echo "  -t TIMEPOINT     : Timepoint (e.g., fast)"
    echo "  -f SEGTOCUT    : Segmented area .nii to cut off (e.g. segtocut.nii.gz)"
    exit 1
}

# Parse input arguments
while getopts ":p:t:f:" opt; do
    case $opt in
        p) patient_id="$OPTARG" ;;
        t) timepoint="$OPTARG" ;;
        f) segtocutfile="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check mandatory parameters
if [ -z "$patient_id" ] || [ -z "$timepoint" ] || [ -z "$segtocutfile" ]; then
    usage
fi

## Define remaining input parameters
input_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1w_time1_bias_corr_registered_scans/BET_Output/" #remember / at the end!!!
input_basename="initialise"

# Select input_basename based on timepoint

# Function to find input mask basename based on timepoint
find_mask_input_basename() {
    local input_basename
    local found=0
    
    
    while IFS= read -r file; do
        basename=$(basename "$file")
        if [[ "$basename" == "T1"* && "$basename" == *"$timepoint"* && "$basename" == *"restore_registered_bet_mask"*".nii.gz" ]]; then
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


# Function to find input image basename based on timepoint
find_input_basename() {
    local input_basename
    local found=0
    
    
    while IFS= read -r file; do
        basename=$(basename "$file")
        if [[ "$basename" == "T1"* && "$basename" == *"$timepoint"* && "$basename" == *"restore_registered_bet"*".nii.gz" && "$basename" != *"mask"* ]]; then
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

## MAIN SCRIPT EXECUTION
mask_input_basename=$(find_mask_input_basename)
input_basename=$(find_input_basename)

echo "Final input basename is $input_basename"
echo "Final mask input basename is $mask_input_basename"

input_basename_without_extension="${input_basename%.nii.gz}"
mask_input_basename_without_extension="${mask_input_basename%.nii.gz}"
input_mask="${input_directory}${mask_input_basename}"
input_image="${input_directory}${input_basename}"
echo "Input mask: $input_mask"

segmentation_to_cut=${input_directory}$6

# Define output file names
inverted_mask="${input_directory}segtocut_inverted.nii.gz"
output_mask="${input_directory}${mask_input_basename_without_extension}modifiedmask.nii.gz"
output_image="${input_directory}${input_basename_without_extension}modified.nii.gz"

# Step 1: Create an inverted mask from the segmentation
echo "Creating inverted mask..."
fslmaths $segmentation_to_cut -binv $inverted_mask
echo "Completed."


# Step 2: Apply the inverted mask to the brain image
echo "Applying inverted mask to remove specified region..."
fslmaths $input_mask -mas $inverted_mask $output_mask

echo "Mask region removal completed. Output saved as $output_mask"

echo "Modifying bet image..."
fslmaths $input_image -mul $output_mask $output_image
echo "Region removal completed. Output saved as $output_mask"

echo "Opening in fsleyes..."
fsleyes $output_image $output_mask

echo "Script complete."




chmod +x manualeditbet.sh
