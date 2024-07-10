#!/bin/bash



#Function to perform bias correction for a given patient ID and timepoint

perform_bias_correction() {
    local patient_id=$1
    local timepoint=$2

    #Load modules
    module load fsl

    ##SET REMAINING VARIABLES
    ## Define remaining input parameters
    input_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/OG_Scans/" # make sure to put / at end of directory
    input_basename="initialise"
    log_file="/home/cmb247/Desktop/Project_3/BET_Extractions/bias_correction_log.txt"
    found=0
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
    
    ## CHECK IF OUTPUT FILE ALREADY EXISTS
    if [ -f "$output_image" ]; then
        echo "Output file ${output_image} already exists. Skipping bias correction."
        return 0
    fi

    ## STARTING BIAS CORRECTION
    echo "starting bias correction for input_image: $input_image..."
    # create bias correction output directory OG_Scans_bias_corr
    if [ -d $output_directory ]; then
        echo "Output directory ${output_directory} exists."
    else
        mkdir "${output_directory}"
        echo "Output directory ${output_directory} created."
    fi

    # Do bias correction using FSL fast
    echo "Performing bias correction for $patient_id $timepoint now..."
    fast -B -o $output_image $input_image

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "Bias field correction completed successfully."
        echo "Bias field correction completed for $patient_id $timepoint successfully." >> $log_file
        echo "Output file basename: $output_basename"
    else
        echo "Bias field correction failed."
        echo "Bias field correction failed for $patient_id $timepoint." >> $log_file

        exit 1
    fi

}

chmod +x functrial1.sh


# Call the function with patient ID and timepoint
# Perform_bias_correction "19978" "fast"

# Arrays of patient IDs and timepoints
patient_ids=("19978" "12345" "67890")  # Add more patient IDs as needed
timepoints=("ultra-fast" "fast" "acute" "3mo" "6mo" "12mo" "24mo")

# Iterate over each patient ID and timepoint
for patient_id in "${patient_ids[@]}"; do
    for timepoint in "${timepoints[@]}"; do
        perform_bias_correction "$patient_id" "$timepoint"
    done
done
