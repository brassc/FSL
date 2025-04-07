#!/bin/bash

# DWI Preprocessing Script
# Performs MRtrix denoising, Gibbs ringing correction, and FSL eddy correction

# Usage: 
#   ./preprocess_dwi.sh                    # Process all patients and timepoints
#   ./preprocess_dwi.sh patient_id         # Process all timepoints for a specific patient
#   ./preprocess_dwi.sh patient_id timepoint # Process a specific patient and timepoint

# Set variables
basepath="/home/cmb247/rds/hpc-work/April2025_DWI"

# Check arguments and determine processing mode
if [ $# -eq 0 ]; then
    # Process all patients and timepoints
    echo "No arguments provided. Processing all patients and timepoints."
    echo "This will process all DWI datasets in ${basepath}"
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
basepath="/home/cmb247/rds/hpc-work/April2025_DWI"
# Load required modules
echo "Loading required modules..."
module load mrtrix
module load fsl

# Function to process a single DWI dataset
process_dwi() {
    local patient=$1
    local tp=$2
    
    echo "======================================================"
    echo "Processing patient: $patient, timepoint: $tp"
    echo "======================================================"
    
    input_dir="${basepath}/${patient}/${tp}/OG_DWI_space"
    output_dir="${basepath}/${patient}/${tp}/preprocessed_DWI_space"
    
    # Create output directory if it doesn't exist
    mkdir -p ${output_dir}
    
    # Check if input data exists
    dwi_file=$(find ${input_dir} -name "*.nii.gz" -o -name "*.nii" | head -n 1)

    if [ -z "$dwi_file" ]; then
        echo "Error: No DWI file found in ${input_dir}"
        return 1
    fi

    echo "Found DWI file: ${dwi_file}"

    # Find bvec and bval files
    bvec_file=$(find ${input_dir} -name "*.bvec" | head -n 1)
    bval_file=$(find ${input_dir} -name "*.bval" | head -n 1)

    if [ -z "$bvec_file" ] || [ -z "$bval_file" ]; then
        echo "Error: bvec or bval file not found in ${input_dir}"
        return 1
    fi

    echo "Found bvec file: ${bvec_file}"
    echo "Found bval file: ${bval_file}"

    # Get the base filename without extension
    filename=$(basename ${dwi_file})
    filename_base="${filename%.*}"
    if [[ "$filename" == *.nii.gz ]]; then
        filename_base="${filename_base%.*}"
    fi

    echo "Processing ${filename_base}..."

    # Step 1: Denoise DWI data using MRtrix dwidenoise
    if [ -f "${output_dir}/${filename_base}_denoised.nii.gz" ]; then
        echo "Step 1: Denoised file already exists. Skipping denoising step."
    else
        echo "Step 1: MRtrix denoising..."
        dwidenoise ${dwi_file} ${output_dir}/${filename_base}_denoised.nii.gz
    fi

    # Step 2: Gibbs ringing correction using MRtrix mrdegibbs
    if [ -f "${output_dir}/${filename_base}_denoised_degibbs.nii.gz" ]; then
        echo "Step 2: Degibbs file already exists. Skipping Gibbs ringing correction step."
    else
        echo "Step 2: Gibbs ringing correction..."
        mrdegibbs ${output_dir}/${filename_base}_denoised.nii.gz ${output_dir}/${filename_base}_denoised_degibbs.nii.gz
    fi
    

    # Step 3: Generate b0 image for topup (if available)
    if [ -f "${output_dir}/${filename_base}_b0.nii.gz" ]; then
        echo "Step 3: b0 file already exists. Skipping b0 extraction step."
    else
        echo "Step 3: Extracting b0 image..."
        # Use the -fslgrad option to explicitly provide bvec and bval files
        dwiextract -fslgrad ${bvec_file} ${bval_file} ${output_dir}/${filename_base}_denoised_degibbs.nii.gz -bzero ${output_dir}/${filename_base}_b0.nii.gz
    fi
    

    # Step 4: Prepare for eddy correction
    echo "Step 4: Preparing for eddy correction..."

    # Create index file for eddy
    nvols=$(fslnvols ${output_dir}/${filename_base}_denoised_degibbs.nii.gz)
    indx=""
    for ((i=1; i<=$nvols; i++)); do
        indx="$indx 1"
    done
    echo $indx > ${output_dir}/index.txt


    # Create acquisition parameters file for eddy
    # Format: [PhaseEncodingDirection x y z] [TotalReadoutTime]
    # Most common configuration: anterior-posterior phase encoding
    echo "0 -1 0 0.05" > ${output_dir}/acqp.txt
    echo "# This is an assumed configuration for anterior-posterior phase encoding" >> ${output_dir}/acqp.txt
    echo "# If eddy correction fails, try one of these alternatives:" >> ${output_dir}/acqp.txt
    echo "# 0 1 0 0.05 (posterior-anterior)" >> ${output_dir}/acqp.txt
    echo "# -1 0 0 0.05 (left-right)" >> ${output_dir}/acqp.txt
    echo "# 1 0 0 0.05 (right-left)" >> ${output_dir}/acqp.txt

    # Find which volumes are b=0
    bvals=$(cat ${bval_file})
    b0_masks=""
    count=0
    for bval in $bvals; do
        if [ $bval -lt 50 ]; then
            b0_masks="$b0_masks 1"
        else
            b0_masks="$b0_masks 0"
        fi
        count=$((count+1))
    done
    echo $b0_masks > ${output_dir}/b0_masks.txt

    # Step 5: Run FSL's eddy correction
    echo "Step 5: Running eddy correction..."

     # Register T1 mask to DWI space for eddy
    echo "Step 5: Registering T1 brain mask to DWI space..."
    t1_mask_dir="${basepath}/${patient}/${tp}/T1_space_bet"
    
    # Find the mask file (file containing "mask" in the filename)
    t1_mask_file=$(find "${t1_mask_dir}" -type f -name "*mask*" | head -n 1)
    
    if [ -f "$t1_mask_file" ]; then
        echo "Found T1 brain mask: $t1_mask_file"
        
        # Find T1 image (assuming it's in the same directory but without "mask" in the name)
        t1_image=$(find "${t1_mask_dir}" -type f -name "*.nii.gz" -not -name "*mask*" | head -n 1)
        
        if [ -z "$t1_image" ]; then
            echo "Warning: T1 image not found in ${t1_mask_dir}"
            echo "Looking in parent directory..."
            t1_image=$(find "${basepath}/${patient}/${tp}" -maxdepth 1 -type f -name "*.nii.gz" -not -name "*mask*" | head -n 1)
        fi
        
        if [ -f "$t1_image" ]; then
            echo "Found T1 image: $t1_image"
            
            # Check if transformation matrix already exists
            if [ -f "${output_dir}/T1_to_DWI.mat" ] && [ -f "${output_dir}/${filename_base}_brain_mask.nii.gz" ]; then
                echo "T1-to-DWI transformation matrix and brain mask already exist. Skipping registration."
            else
                # Register T1 to DWI using the b0 image as target
                echo "Registering T1 to DWI space..."
                flirt -in "$t1_image" \
                      -ref "${output_dir}/${filename_base}_b0.nii.gz" \
                      -omat "${output_dir}/T1_to_DWI.mat" \
                      -dof 6
                
                # Apply transformation to the T1 brain mask
                echo "Transforming T1 mask to DWI space..."
                flirt -in "$t1_mask_file" \
                      -ref "${output_dir}/${filename_base}_b0.nii.gz" \
                      -applyxfm -init "${output_dir}/T1_to_DWI.mat" \
                      -out "${output_dir}/${filename_base}_brain_mask.nii.gz" \
                      -interp nearestneighbour
                
                # Ensure binary mask
                fslmaths "${output_dir}/${filename_base}_brain_mask.nii.gz" -bin "${output_dir}/${filename_base}_brain_mask.nii.gz"
                
                echo "T1 mask successfully registered to DWI space"
            fi
        else
            echo "Error: Could not find T1 image for registration. Using fallback method."
            fslmaths ${output_dir}/${filename_base}_b0.nii.gz -bin ${output_dir}/${filename_base}_brain_mask.nii.gz
        fi
    else
        echo "Warning: T1 brain mask not found at ${t1_mask_dir}"
        echo "Creating a basic mask from b0 image (not optimal)..."
        fslmaths ${output_dir}/${filename_base}_b0.nii.gz -bin ${output_dir}/${filename_base}_brain_mask.nii.gz
    fi
    
    # Run eddy_openmp (standard eddy)

    # Check if eddy correction has already been completed
    if [ -f "${output_dir}/${filename_base}_eddy_corrected.nii.gz" ]; then
        echo "Eddy correction already completed. Skipping this step."
    else
        echo "Running eddy_openmp..."
        eddy --verbose \
            --imain=${output_dir}/${filename_base}_denoised_degibbs.nii.gz \
            --mask=${output_dir}/${filename_base}_brain_mask.nii.gz \
            --acqp=${output_dir}/acqp.txt \
            --index=${output_dir}/index.txt \
            --bvecs=${bvec_file} \
            --bvals=${bval_file} \
            --out=${output_dir}/${filename_base}_eddy_corrected \
            --nthr=2

        # Copy the bvec and bval files to the output directory
        cp ${bvec_file} ${output_dir}/${filename_base}_eddy_corrected.bvec
        cp ${bval_file} ${output_dir}/${filename_base}_eddy_corrected.bval
    fi
    echo "Preprocessing complete for ${patient}, ${tp}!"
    echo "Final preprocessed DWI: ${output_dir}/${filename_base}_eddy_corrected.nii.gz"
    echo "Final bvecs: ${output_dir}/${filename_base}_eddy_corrected.bvec"
    echo "Final bvals: ${output_dir}/${filename_base}_eddy_corrected.bval"
    echo "Note: Brain extraction (BET) has not been performed."

    return 0
}

# Main execution block
if [ "$process_all" = true ]; then
    # Process all patients and timepoints
    echo "Processing all datasets in ${basepath}..."
    
    # Find all patient directories
    for patient_dir in $(find ${basepath} -maxdepth 1 -type d -not -path "${basepath}"); do
        current_patient=$(basename ${patient_dir})

        # Skip if this isn't a valid patient directory
        if [ ! -d "${patient_dir}" ]; then
            echo "Skipping invalid patient directory: ${patient_dir}"
            continue
        fi

        # Skip if T1 directory doesn't exist for this patient
        if [ -z "$(find "${patient_dir}" -maxdepth 2 -type d -name "T1_space_bet" | head -n 1)" ]; then
        echo "patient_dir: ${patient_dir}"
            echo "T1 directory not found for patient ${current_patient}"
            echo "Skipping this patient."
            continue
        fi


        
        # Find all timepoint directories for current patient
        for timepoint_dir in $(find ${patient_dir} -maxdepth 1 -type d); do
            current_timepoint=$(basename ${timepoint_dir})
            
            # Skip if not a proper timepoint directory (check if OG_DWI_space exists)
            if [ ! -d "${patient_dir}/${current_timepoint}/OG_DWI_space" ]; then
                continue
            fi
            
            # Process this patient/timepoint
            process_dwi ${current_patient} ${current_timepoint}
        done
    done
    
elif [ -n "$patient_id" ] && [ -z "$timepoint" ]; then
    # Process all timepoints for a specific patient
    echo "Processing all timepoints for patient: ${patient_id}..."
    
    # Find all timepoint directories for current patient
    patient_dir="${basepath}/${patient_id}"
    if [ ! -d "${patient_dir}" ]; then
        echo "Error: Patient directory not found: ${patient_dir}"
        exit 1
    fi
    
    for timepoint_dir in $(find ${patient_dir} -maxdepth 1 -type d); do
        current_timepoint=$(basename ${timepoint_dir})
        
        # Skip if not a proper timepoint directory (check if OG_DWI_space exists)
        if [ ! -d "${patient_dir}/${current_timepoint}/OG_DWI_space" ]; then
            continue
        fi
        
        # Process this patient/timepoint
        process_dwi ${patient_id} ${current_timepoint}
    done
    
else
    # Process a specific patient and timepoint
    if [ ! -d "${basepath}/${patient_id}" ]; then
        echo "Error: Patient directory not found: ${basepath}/${patient_id}"
        exit 1
    fi
    
    if [ ! -d "${basepath}/${patient_id}/${timepoint}" ]; then
        echo "Error: Timepoint directory not found: ${basepath}/${patient_id}/${timepoint}"
        exit 1
    fi
    
    if [ ! -d "${basepath}/${patient_id}/${timepoint}/OG_DWI_space" ]; then
        echo "Error: OG_DWI_space directory not found: ${basepath}/${patient_id}/${timepoint}/OG_DWI_space"
        exit 1
    fi
    
    process_dwi ${patient_id} ${timepoint}
fi

echo "All processing complete!"