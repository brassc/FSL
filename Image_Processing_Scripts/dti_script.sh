#!/bin/bash

# Script to flatten cortex using freesurfer

# Load modules
module load freesurfer

export SUBJECTS_DIR=/home/cmb247/Desktop/Project_3/Freesurfer/
echo "SUBJECTS_DIR is: $SUBJECTS_DIR"

patient_id=19978
timepoint=ultra-fast

input_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1w_time1_bias_corr_registered_scans/"

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


input_basename=$(find_input_basename)
destination_dir="/home/cmb247/Desktop/Project_3/Freesurfer/${patient_id}_$timepoint/"

# Check if the destination directory exists, if not, create it
if [ ! -d "$destination_dir" ]; then
    mkdir -p "$destination_dir"
fi

# Copy the file to the destination directory
cp "$input_directory$input_basename" "$destination_dir"

echo "File copied to $destination_dir"

# Starting recon-all
recon-all -i $destination_dir$input_basename -s "${patient_id}_${timepoint}" -all 









