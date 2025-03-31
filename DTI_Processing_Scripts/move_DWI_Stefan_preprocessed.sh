#!/bin/bash

# Define base paths
SOURCE_BASE="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi"
DEST_BASE="/rds-d5/user/cmb247/hpc-work/April2025_DWI"

# Files to be copied
FILES_TO_COPY=("DTI_corrected.nii.gz" "DTI_corrected.bvec" "DTI_corrected.bval")

# Create destination base directory if it doesn't exist
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
                echo " Processing timepoint: $timepoint"
                
                # Find directories containing DTIspace/dwi_proc with the specified files
                for dwi_proc_dir in $(find "$timepoint_dir" -type d -path "*/nipype/DATASINK/DTIspace/dwi_proc"); do
                    echo " Found dwi_proc directory at: $dwi_proc_dir"
                    
                    # Check if all required files exist in this directory
                    all_files_exist=true
                    for file in "${FILES_TO_COPY[@]}"; do
                        if [ ! -f "$dwi_proc_dir/$file" ]; then
                            echo " Missing required file: $file"
                            all_files_exist=false
                            break
                        fi
                    done
                    
                    # If all files exist, copy them to destination
                    if [ "$all_files_exist" = true ]; then
                        # Create destination directory
                        dest_dir="$DEST_BASE/$patient_id/$timepoint/Stefan_preprocessed_DWI_space"
                        mkdir -p "$dest_dir"
                        
                        # Copy the specified files to destination
                        echo " Copying files to: $dest_dir"
                        for file in "${FILES_TO_COPY[@]}"; do
                            cp "$dwi_proc_dir/$file" "$dest_dir/"
                            
                            # Check if copy was successful
                            if [ $? -eq 0 ]; then
                                echo " Successfully copied: $file"
                            else
                                echo " ERROR: Failed to copy: $file"
                            fi
                        done
                        
                        echo " Copy operation completed for this directory"
                    else
                        echo " Skipping directory due to missing files"
                    fi
                done
            fi
        done
    fi
done

echo "All DWI processing completed. Files have been copied to $DEST_BASE in Stefan_preprocessed_DWI_space folders."