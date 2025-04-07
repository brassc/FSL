# This script takees DTI_corrected_bet (created by applyT1masktoStefanpreprocessed.sh) and performs FSL dtifit

#!/bin/bash

# Base directory containing all patient data
BASE_DIR="/home/cmb247/rds/hpc-work/April2025_DWI"

# Find all patient directories
for patient_dir in ${BASE_DIR}/*; do
    if [ -d "$patient_dir" ]; then
        patient_id=$(basename "$patient_dir")
        echo "Processing patient: $patient_id"
        
        # Find all timepoint directories for this patient
        for timepoint_dir in ${patient_dir}/*; do
            if [ -d "$timepoint_dir" ]; then
                timepoint=$(basename "$timepoint_dir")
                echo "  Processing timepoint: $timepoint"
                
                # Path to the preprocessed data
                PREPROC_DIR="${timepoint_dir}/Stefan_preprocessed_DWI_space"
                
                if [ -d "$PREPROC_DIR" ]; then
                    # Input files
                    DATA_FILE="${PREPROC_DIR}/DTI_corrected_bet.nii.gz"
                    MASK_FILE="${PREPROC_DIR}/DTI_corrected_bet_mask.nii.gz"
                    BVEC_FILE="${PREPROC_DIR}/DTI_corrected_bet.bvec"
                    BVAL_FILE="${PREPROC_DIR}/DTI_corrected_bet.bval"
                    
                    # Output basename
                    OUT_BASE="${PREPROC_DIR}/dti"
                    
                    # Check if all required files exist
                    if [ -f "$DATA_FILE" ] && [ -f "$BVEC_FILE" ] && [ -f "$BVAL_FILE" ]; then
                        echo "    Running dtifit on ${PREPROC_DIR}"
                        
                        # Check if mask file exists
                        if [ ! -f "$MASK_FILE" ]; then
                            echo "    Warning: Mask file not found. Using data file for automatic mask generation."
                            MASK_FILE="$DATA_FILE"
                        fi
                        
                        # Run dtifit (single-threaded)
                        dtifit \
                            -k "$DATA_FILE" \
                            -m "$MASK_FILE" \
                            -r "$BVEC_FILE" \
                            -b "$BVAL_FILE" \
                            -o "$OUT_BASE" \
                            --save_tensor
                        
                        echo "    Completed dtifit for ${patient_id}/${timepoint}"
                    else
                        echo "    Error: Required files not found in ${PREPROC_DIR}"
                        echo "    Skipping this directory"
                    fi
                else
                    echo "    Error: Preprocessed directory not found: ${PREPROC_DIR}"
                    echo "    Skipping this timepoint"
                fi
            fi
        done
    fi
done

echo "All processing complete!"