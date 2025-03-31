#!/bin/bash

# Define base paths
SOURCE_BASE="/home/cmb247/Desktop/Project_3/BET_Extractions"
DEST_BASE="/home/cmb247/rds/hpc-work/April2025_DWI"

# This script COPIES T1 BET scans and corresponding masks to destination directories
# It prioritizes "modified" files when multiple files exist for the same patient/timepoint

# For each patient directory in the source path
for patient_dir in "$SOURCE_BASE"/*; do
    # Check if it's a directory
    if [ -d "$patient_dir" ]; then
        patient_id=$(basename "$patient_dir")
        echo "Processing patient: $patient_id"
        
        # Find T1w bias corrected registered BET files
        t1_dir="$patient_dir/T1w_time1_bias_corr_registered_scans/BET_Output"
        if [ ! -d "$t1_dir" ]; then
            echo " WARNING: T1 directory not found for patient $patient_id, skipping."
            continue
        fi
        
        # Create a temporary array to store timepoints we've already processed
        declare -A processed_timepoints
        
        # First pass: find all "modified" BET files (exclude mask files)
        echo " First pass: Looking for modified files..."
        modified_bet_files=$(find "$t1_dir" -type f -name "*modified*" -name "*_bet_*" | grep -v "mask")
        
        for bet_file in $modified_bet_files; do
            bet_filename=$(basename "$bet_file")
            echo " Processing modified BET file: $bet_filename"
            
            # Extract the timepoint
            if [[ $bet_filename =~ ${patient_id}_([^_]+) ]]; then
                timepoint="${BASH_REMATCH[1]}"
                echo " Detected timepoint: $timepoint"
                
                # Create destination directory
                dest_dir="$DEST_BASE/$patient_id/$timepoint/T1_space_bet"
                mkdir -p "$dest_dir"
                
                # Copy BET file to destination
                echo " Copying modified BET file to: $dest_dir"
                cp "$bet_file" "$dest_dir"
                
                # Find corresponding mask file
                # First try pattern with _bet_mask_
                mask_file=$(find "$t1_dir" -type f -name "${bet_filename/_bet_/_bet_mask_}")
                
                # If not found, try other common mask naming patterns
                if [ ! -f "$mask_file" ]; then
                    mask_file=$(find "$t1_dir" -type f -name "${bet_filename/_bet_/_mask_}")
                fi
                
                if [ ! -f "$mask_file" ]; then
                    # One more attempt - look for any mask file with similar base name
                    mask_base=$(echo "$bet_filename" | sed 's/_bet_.*//')
                    mask_file=$(find "$t1_dir" -type f -name "${mask_base}*mask*" -name "*modified*")
                fi
                
                if [ -f "$mask_file" ]; then
                    echo " Copying modified mask file to: $dest_dir"
                    cp "$mask_file" "$dest_dir"
                    echo " Mask copy successful!"
                else
                    echo " WARNING: Could not find corresponding modified mask file for $bet_filename"
                fi
                
                # Check if BET file copy was successful
                if [ -f "$dest_dir/$(basename "$bet_file")" ]; then
                    echo " Modified BET file copy successful!"
                    # Mark this timepoint as processed
                    processed_timepoints["$timepoint"]=1
                else
                    echo " ERROR: Modified BET file copy failed!"
                fi
            else
                echo " ERROR: Could not extract timepoint from modified filename: $bet_filename"
            fi
        done
        
        # Second pass: process regular (non-modified) files for timepoints not already processed
        echo " Second pass: Processing non-modified files for remaining timepoints..."
        regular_bet_files=$(find "$t1_dir" -type f -name "*_bet_*" | grep -v "mask" | grep -v "modified")
        
        for bet_file in $regular_bet_files; do
            bet_filename=$(basename "$bet_file")
            
            # Extract the timepoint
            if [[ $bet_filename =~ ${patient_id}_([^_]+) ]]; then
                timepoint="${BASH_REMATCH[1]}"
                
                # Skip if we already processed a modified file for this timepoint
                if [[ ${processed_timepoints["$timepoint"]} -eq 1 ]]; then
                    echo " Skipping non-modified file for timepoint $timepoint (modified version already processed)"
                    continue
                fi
                
                echo " Processing non-modified BET file for timepoint: $timepoint"
                
                # Create destination directory
                dest_dir="$DEST_BASE/$patient_id/$timepoint/T1_space_bet"
                mkdir -p "$dest_dir"
                
                # Copy BET file to destination
                echo " Copying non-modified BET file to: $dest_dir"
                cp "$bet_file" "$dest_dir"
                
                # Find corresponding mask file with similar logic as above
                mask_file=$(find "$t1_dir" -type f -name "${bet_filename/_bet_/_bet_mask_}")
                
                if [ ! -f "$mask_file" ]; then
                    mask_file=$(find "$t1_dir" -type f -name "${bet_filename/_bet_/_mask_}")
                fi
                
                if [ ! -f "$mask_file" ]; then
                    mask_base=$(echo "$bet_filename" | sed 's/_bet_.*//')
                    mask_file=$(find "$t1_dir" -type f -name "${mask_base}*mask*" | grep -v "modified")
                fi
                
                if [ -f "$mask_file" ]; then
                    echo " Copying non-modified mask file to: $dest_dir"
                    cp "$mask_file" "$dest_dir"
                    echo " Mask copy successful!"
                else
                    echo " WARNING: Could not find corresponding non-modified mask file for $bet_filename"
                fi
                
                # Check if BET file copy was successful
                if [ -f "$dest_dir/$(basename "$bet_file")" ]; then
                    echo " Non-modified BET file copy successful!"
                else
                    echo " ERROR: Non-modified BET file copy failed!"
                fi
            else
                echo " ERROR: Could not extract timepoint from non-modified filename: $bet_filename"
            fi
        done
        
        # Clean up
        unset processed_timepoints
    fi
done

echo "All T1 BET data has been COPIED to $DEST_BASE. The original data remains intact."
echo "Modified files were prioritized over non-modified files where available."