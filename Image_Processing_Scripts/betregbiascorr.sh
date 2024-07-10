#!/bin/bash
#Load modules
module load fsl

# Define patient info input parameters
patient_id="19978"
timepoint="fast"

# Define BET parameters
bet_params="-f 0.70 -R"
bet_params_filename=$(echo "$bet_params" | tr ' ' '_') # remove spaces so bet_params can be used in filename
neck_cut='56'

## Define remaining input parameters
input_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1_time1_bias_corr_registered_scans/"
input_basename="initialise"
log_file="/home/cmb247/Desktop/Project_3/BET_Extractions/bias_bet_reg_log.txt"

# Select input_basename based on timepoint
#find input scan basename based on timepoint input
while IFS= read -r file; do
    # Extract basename of file
    basename=$(basename "$file")
        
    # Check if the file name contains "T1" and "$timepoint"
    if [[ "$basename" == "T1"* && "$basename" == *"$timepoint"* && "$basename" == *".nii.gz" ]]; then 
        # Exclude files that start with "ultra-" before search term (i.e. "ultra-fast" is fine)
        if [[ "$timepoint" == "fast" && "$basename" != *"ultra-fast"* ]]; then
            input_basename=$basename
            found=1
            echo "Input basename set as $input_basename."
        elif [[ "$timepoint" == "fast" && "$basename" == *"ultra-fast"* ]]; then
            echo "skipping ultra-fast basename allocation for $timepoint timepoint."
        else
            input_basename=$basename
            found=1
            echo "Input basename set as $input_basename."
        fi

    fi
done < <(find "$input_directory" -type f)

# Check if a matching file was found
if [ $found -eq 0 ]; then
    echo "Error: No matching file found for patient_id $patient_id and timepoint $timepoint."
    return 1
fi

echo "Final input basename is $input_basename"
input_basename_without_extension="${input_basename%.nii.gz}"
input_image="${input_directory}${input_basename}"
echo "Input image: $input_image"

# Define output parameters
output_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1_time1_bias_corr_registered_scans/" # make sure to put / at end of directory
output_basename="${input_basename_without_extension}_bet_rbc_${bet_params_filename}.nii.gz"
mask_output_basename="${input_basename_without_extension}_bet_mask_rbc_${bet_params_filename}.nii.gz"
output_image="${output_directory}${output_basename}"
output_mask="${output_directory}${mask_output_basename}"



## VERIFY INPUT IMAGE
# Verify input image
if [ ! -f "$input_image" ]; then
    echo "Error: input_image ${input_image} does not exist."
    exit 1
fi

## CHECK IF OUTPUT FILE ALREADY EXISTS
if [ -f "$output_image" ]; then
    echo "Output file ${output_image} already exists. Skipping bet."
    return 0
fi

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
echo "performing bet..." 
bet $input_image $output_image $bet_params
echo "BET complete"
# 3. Create mask
fslmaths $output_image -bin $output_mask
# 3. Check output
fsleyes $input_image $output_mask $output_image
# 4. write to log file
echo "$patient_id, $timepoint, $bet_params, biascorr=YES" >> $log_file






chmod +x betregbiascorr.sh
