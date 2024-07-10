#!/bin/bash
#Load modules
module load fsl

## PATIENT PARAMETERS
patient_id="19978"
timepoint="fast"

##SET REMAINING VARIABLES
## Define remaining input parameters
input_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/OG_Scans/" # make sure to put / at end of directory
input_basename="initialise"
#find input scan basename based on timepoint input
while IFS= read -r file; do
    # Extract basename of file
    basename=$(basename "$file")
    
    # Check if the file name contains "T1" and "$timepoint"
    if [[ "$basename" == "T1"* && "$basename" == *"$timepoint"* && "$basename" == *".nii.gz" ]]; then 
        # Exclude files that start with "ultra-" before search term (i.e. "ultra-fast" is fine)
        if [[ "$timepoint" == "fast" && "$basename" != *"ultra-fast"* ]]; then
            input_basename=$basename
            echo "Input basename set as $input_basename."
        elif [[ "$timepoint" == "fast" && "$basename" == *"ultra-fast"* ]]; then
            echo "skipping ultra-fast basename allocation for $timepoint timepoint."
        else
            input_basename=$basename
            echo "Input basename set as $input_basename."
        fi

    fi
done < <(find "$input_directory" -type f)

echo "Final input basename is $input_basename"
input_basename_without_extension="${input_basename%.nii.gz}"
input_image="${input_directory}${input_basename}"
echo "$input_image"

# Define output parameters
output_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/OG_Scans_bias_corr/" # make sure to put / at end of directory
output_basename="${input_basename_without_extension}_bias_corr.nii.gz"
output_image="${output_directory}${output_basename}"



## VERIFY INPUT IMAGE
# Verify input image
if [ ! -f "$input_image" ]; then
    echo "Error: input_image ${input_image} does not exist."
    exit 1
fi
echo "starting bias correction for input_image: $input_image..."


## STARTING BIAS CORRECTION
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
