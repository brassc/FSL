#!/bin/bash

# Apply T1 Mask to DTI Script
# This script applies a T1-derived brain mask to DTI_corrected.nii.gz files

# Usage: 
#   ./apply_mask_to_dti.sh                    # Process all patients and timepoints
#   ./apply_mask_to_dti.sh patient_id         # Process all timepoints for a specific patient
#   ./apply_mask_to_dti.sh patient_id timepoint # Process a specific patient and timepoint

# Set variables
basepath="/home/cmb247/rds/hpc-work/April2025_DWI"

# Check arguments and determine processing mode
if [ $# -eq 0 ]; then
    # Process all patients and timepoints
    echo "No arguments provided. Processing all patients and timepoints."
    echo "This will process all DTI datasets in ${basepath}"
    echo "Press any key to confirm, or Ctrl+C to cancel..."
    read -n 1 -s
    echo "Proceeding with batch processing..."
    process_all=true
    patient_id=""
    timepoint=""
elif [ $# -eq 1 ]; then
    # Process all timepoints for a specific patient
    echo "Processing all timepoints for patient: $1"
    process_all=false
    patient_id=$1
    timepoint=""
elif [ $# -eq 2 ]; then
    # Process a specific patient and timepoint
    echo "Processing patient: $1, timepoint: $2"
    process_all=false
    patient_id=$1
    timepoint=$2
else
    echo "Usage: $0 [patient_id] [timepoint]"
    exit 1
fi

# Load required modules
echo "Loading required modules..."
module load fsl

# Function to apply T1 mask to a DTI file
apply_mask_to_dti() {
    local patient=$1
    local tp=$2
    
    echo "======================================================"
    echo "Processing patient: $patient, timepoint: $tp"
    echo "======================================================"
    
    # Set up directories based on your structure
    dti_dir="${basepath}/${patient}/${tp}/Stefan_preprocessed_DWI_space"
    preproc_dir="${basepath}/${patient}/${tp}/preprocessed_DWI_space"
    
    # Check if DTI file exists
    dti_file="${dti_dir}/DTI_corrected.nii.gz"
    
    if [ ! -f "$dti_file" ]; then
        echo "Error: DTI file not found: $dti_file"
        return 1
    fi
    
    echo "Found DTI file: $dti_file"

    # Check for output file - skip if it already exists
    output_file="${dti_dir}/DTI_corrected_bet.nii.gz"
    if [ -f "$output_file" ]; then
        echo "Output file already exists: $output_file"
        echo "Skipping this dataset."
        return 0
    fi
    
    # Look for the T1 mask in the T1_space_bet directory
    t1_mask_dir="${basepath}/${patient}/${tp}/T1_space_bet"
    
    # Find the mask file (file containing "mask" in the filename)
    mask_file=$(find "${t1_mask_dir}" -type f -name "*mask*" | head -n 1)
    
    # Check if the mask exists
    if [ ! -f "$mask_file" ]; then
        echo "Error: Brain mask not found: $mask_file"
        echo "Checking for any brain mask file..."
        mask_file=$(find "${preproc_dir}" -name "*mask*.nii.gz" | head -n 1)
        
        if [ ! -f "$mask_file" ]; then
            echo "Error: No brain mask found in ${preproc_dir}"
            return 1
        fi
    fi
    
    echo "Found brain mask: $mask_file"

    # Look for T1 image (same directory but without "mask" in the name)
    
    t1_image=$(find "${t1_mask_dir}" -type f -name "*.nii.gz" -not -name "*mask*" | head -n 1)
    
    if [ -z "$t1_image" ]; then
        echo "Warning: T1 image not found in ${t1_mask_dir}"
        return 1
    fi

    # echo "Found T1 image: $t1_image"

    # # Registering T1 image to DTI space
    # echo "Registering T1 image to DTI space..."

    # Check if T1 image exists
    if [ -f "$t1_image" ]; then
        echo "Found T1 image: $t1_image"
        
        # Register T1 to DTI space
        echo "Registering T1 to DTI space..."
        if [ -f "${dti_dir}/T1_to_DTI.mat" ]; then
            echo "Transformation matrix already exists. Skipping registration."
        else
            echo "Performing FLIRT registration..."
        
            flirt -in "$t1_image" \
                -ref "$dti_file" \
                -omat "${dti_dir}/T1_to_DTI.mat" \
                -dof 6
        fi
        
        # Apply transformation to the T1 brain mask
        # echo "Transforming T1 mask to DTI space..."
        # check if T1_mask_in_DTI_space.nii.gz already exists
        if [ -f "${dti_dir}/T1_mask_in_DTI_space.nii.gz" ]; then
            echo "T1 mask in DTI space already exists. Skipping transformation."
        else
            echo "Transforming T1 mask to DTI space..."
        
            flirt -in "$mask_file" \
                -ref "$dti_file" \
                -applyxfm -init "${dti_dir}/T1_to_DTI.mat" \
                -out "${dti_dir}/T1_mask_in_DTI_space.nii.gz" \
                -interp nearestneighbour
            
            # Ensure binary mask
            fslmaths "${dti_dir}/T1_mask_in_DTI_space.nii.gz" -bin "${dti_dir}/T1_mask_in_DTI_space.nii.gz"
            echo "T1 mask successfully registered to DTI space"
            
            
        fi
    else
        echo "Error: Could not find T1 image. Exiting."
        return 1
    fi
    # Check if the registered mask exists
    if [ ! -f "${dti_dir}/T1_mask_in_DTI_space.nii.gz" ]; then
        echo "Error: Registered mask not found: ${dti_dir}/T1_mask_in_DTI_space.nii.gz"
        return 1
    fi
    # Use the registered mask
    registered_mask="${dti_dir}/T1_mask_in_DTI_space.nii.gz"
    
    # Check if output file exists
    if [ -f "$output_file" ]; then
        echo "Output file already exists: $output_file"
        echo "Skipping this dataset."
        return 0
    fi
    

    # Apply the registered mask to the DTI file
    echo "Applying brain mask to DTI..."
    #echo "DTI file: $dti_file"
    #echo "Mask file: $registered_mask"
    fslmaths "$dti_file" -mas "$registered_mask" "$output_file"
    
    

    # Check if the operation was successful
    if [ -f "$output_file" ]; then
        echo "Successfully created masked DTI: $output_file"

                
        # Copy the bval and bvec files if they exist
        bval_file="${dti_dir}/DTI_corrected.bval"
        bvec_file="${dti_dir}/DTI_corrected.bvec"
        
        if [ -f "$bval_file" ]; then
            cp "$bval_file" "${dti_dir}/DTI_corrected_bet.bval"
            echo "Copied bval file"
        else
            echo "Warning: bval file not found: $bval_file"
        fi
        
        if [ -f "$bvec_file" ]; then
            cp "$bvec_file" "${dti_dir}/DTI_corrected_bet.bvec"
            echo "Copied bvec file"
        else
            echo "Warning: bvec file not found: $bvec_file"
        fi
    else
        echo "Error: Failed to create masked DTI"
        return 1
    fi
    
    return 0
}

# Main processing loop
if [ "$process_all" = true ]; then
    # Process all patients and timepoints
    for patient_dir in "${basepath}/"*/; do
        patient=$(basename "$patient_dir")
        echo "Found patient: $patient"
        
        for tp_dir in "${patient_dir}"*/; do
            tp=$(basename "$tp_dir")
            echo "Found timepoint: $tp"
            
            apply_mask_to_dti "$patient" "$tp"
        done
    done
else
    if [ -z "$timepoint" ]; then
        # Process all timepoints for a specific patient
        for tp_dir in "${basepath}/${patient_id}/"*/; do
            tp=$(basename "$tp_dir")
            echo "Found timepoint: $tp"
            
            apply_mask_to_dti "$patient_id" "$tp"
        done
    else
        # Process a specific patient and timepoint
        apply_mask_to_dti "$patient_id" "$timepoint"
    fi
fi

echo "Processing complete!"
