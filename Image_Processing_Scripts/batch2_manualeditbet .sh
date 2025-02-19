#!/bin/bash
module load fsl

## EXAMPLE USAGE: ./manualeditbet.sh -p 12519 -t 6mo -f segtocut.nii.gz
## segtocut.nii.gz is mask of area to cut away created using itksnap
## EXAMPLE USAGE: ./manualeditbet.sh -p 12519 -t 6mo -a SEGTOADD.nii.gz area to add created using itksnap

# Function to display usage
usage() {
    echo "Usage: $0 -g GUPI -h HOUR_NUMBER -f SEGTOCUTFILE"
    echo "  -g GUPI    : GUPI"
    echo "  -h HOUR_NUMBER     : Hour number (e.g., 00376)"
    echo "  -f SEGTOCUT    : Segmented area .nii to cut off (e.g. segtocut.nii.gz)"
    echo "  -a SEGTOADD    : Segmented area to add (e.g. seg to add.nii.gz)"
    echo "Either -f <segtocutfile> or -a <segtoaddfile> must be provided."
    exit 1
}

# Parse input arguments
while getopts ":g:h:f:a:" opt; do
    case $opt in
        g) patient_id="$OPTARG" ;;
        h) timepoint="$OPTARG" ;;
        f) segtocutfile="$OPTARG" ;;
        a) segtoaddfile="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check mandatory parameters
if [ -z "$patient_id" ] || [ -z "$timepoint" ]; then
    usage
fi

# Check that either -f or -a is provided
if [ -z "$segtocutfile" ] && [ -z "$segtoaddfile" ]; then
    echo "Error: Either -f <segtocutfile> or -a <segtoaddfile> must be provided."
    usage
fi


## Define remaining input parameters
input_directory="/rds-d5/user/cmb247/hpc-work/Feb2025_working/${patient_id}/BET_Output/"
input_basename="initialise"

input_prebet_image_dir="/rds-d5/user/cmb247/hpc-work/Feb2025_working/${patient_id}/bias_corr/"

prebet_basename="init"


# Select input_basename based on timepoint



## MAIN SCRIPT EXECUTION
mask_input_basename=$(basename $(ls ${input_directory}T1_bias-Hour-${timepoint}*bet_mask*.nii.gz))
input_basename=$(basename $(ls ${input_directory}T1_bias-Hour-${timepoint}*bet_rbc*.nii.gz))

echo "Final input basename is $input_basename"
echo "Final mask input basename is $mask_input_basename"

input_basename_without_extension="${input_basename%.nii.gz}"
mask_input_basename_without_extension="${mask_input_basename%.nii.gz}"
input_mask="${input_directory}${mask_input_basename}"
input_image="${input_directory}${input_basename}"
echo "Input mask: $input_mask"



# Define output file names
inverted_mask="${input_directory}segtocut_inverted.nii.gz"
output_mask="${input_directory}${mask_input_basename_without_extension}modifiedmask.nii.gz"
output_image="${input_directory}${input_basename_without_extension}modified.nii.gz"



# Conditional logic for -f option
if [ -n "$segtocutfile" ]; then

    segmentation_to_cut=${input_directory}$6

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
fi

if [ -n "$segtoaddfile" ]; then
    segmentation_to_add=${input_directory}$6

    # Adding masks together
    echo "Adding segmentation area to mask..."
    fslmaths $input_mask -add $segmentation_to_add $output_mask
    echo "Area addition complete. Thresholding now..."

    # Thresholding to ensure binary values
    fslmaths $output_mask -thr 0.5 -bin $output_mask
    echo "Mask region addition and thresholding completed. Output saved as $output_mask"

    echo "Multiplying original image by new mask to obtain modified BET..."
    GUPI_dir=$(dirname "${input_directory}")
    input_prebet_image_dir="$GUPI_dir/bias_corr/"
    prebet_basename=$(basename $(ls ${input_prebet_image_dir}T1_bias-Hour-${timepoint}*.nii.gz))
    input_prebet_image="${input_prebet_image_dir}${prebet_basename}" 
    
    fslmaths $input_prebet_image -mul $output_mask $output_image
    echo "Region addition to BET complete. Output saved as $output_image"

    echo "Opening in fsleyes..."
    fsleyes $output_mask $output_image
    echo "Script complete."
fi

    



chmod +x manualeditbet.sh
