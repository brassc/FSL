#!/bin/bash

# Define base paths
SOURCE_BASE="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi"
DEST_BASE="/rds-d5/user/cmb247/hpc-work/April2025_DWI"

# This script COPIES data from the source to destination

# Create destination directory if it doesn't exist
mkdir -p "$DEST_BASE"

# Find all patient directories
for patient_dir in "$SOURCE_BASE"/*; do
    # Check if it's a directory
    if [ -d "$patient_dir" ]; then
        patient_id=$(basename "$patient_dir")
        echo "Processing patient: $patient_id"
        
        # Find all timepoint directories for this patient
        for timepoint_dir in "$patient_dir"/*; do
            if [ -d "$timepoint_dir" ]; then
                timepoint=$(basename "$timepoint_dir")
                echo "  Processing timepoint: $timepoint"
                
                # Find DWI_space directories (could be nested at different levels)
                # Using find to locate all DWI_space directories under this timepoint
                dwi_dirs=$(find "$timepoint_dir" -type d -name "DWI_space")
                
                # Process each DWI_space directory found
                for dwi_dir in $dwi_dirs; do
                    echo "    Found DWI_space at: $dwi_dir"
                    
                    # Create destination directory
                    dest_dir="$DEST_BASE/$patient_id/$timepoint/OG_DWI_space"
                    mkdir -p "$dest_dir"
                    
                    # Copy the contents of DWI_space to destination
                    echo "    Copying contents to: $dest_dir"
                    cp -r "$dwi_dir"/* "$dest_dir"/
                    
                    # Check if copy was successful
                    if [ $? -eq 0 ]; then
                        echo "    Copy successful!"
                    else
                        echo "    ERROR: Copy failed!"
                    fi
                done
            fi
        done
    fi
done

echo "All DWI_space data has been COPIED to $DEST_BASE. The original data remains intact."