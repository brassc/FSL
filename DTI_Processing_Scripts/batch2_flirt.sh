#!/bin/bash

# Set base directory
basepath="/home/cmb247/rds/hpc-work/April2025_DWI"

# Load FSL module
module load fsl
echo "FSL module loaded"

# Loop through all directories in the base path
for patient_dir in ${basepath}/*; do
    # Get patient ID from directory name
    patient=$(basename "${patient_dir}")
    
    # Check if directory name contains both letters and numbers
    if [[ -d "${patient_dir}" && "${patient}" =~ [a-zA-Z] && "${patient}" =~ [0-9] ]]; then
        echo "Processing patient: ${patient}"
        
        
        # Find all timepoint directories for this patient
        for tp_dir in ${patient_dir}/*; do
            if [[ -d "${tp_dir}" ]]; then
                tp=$(basename "${tp_dir}")
                echo "Processing timepoint: ${tp}"
                
                
                # Define directories
                t1_dir="${tp_dir}/T1_space_bet"
                dwi_dir="${tp_dir}/proc_set1_nobzero/nipype/DATASINK/DTIspace/dwi_proc"
                output_dir="${tp_dir}/T1_space_bet/coordinates_files"

                echo "  T1 Directory: ${t1_dir}"
                echo "  DWI Directory: ${dwi_dir}"
                
                # Check if required directories exist
                if [[ ! -d "${t1_dir}" || ! -d "${dwi_dir}" ]]; then
                    echo "  Required directories not found, skipping"
                    echo " ************************************"
                    echo " ****************************************"
                    echo " ************************************"
                    continue
                fi
                
                # Create output directory
                mkdir -p "${output_dir}"
                
                # Find required files
                t1_image=$(find "${t1_dir}" -type f -name "*.nii.gz" -not -name "*mask*" | head -n 1)
                t1_mask=$(find "${t1_dir}" -type f -name "*mask*.nii.gz" | head -n 1)
                dwi_file=$(find "${dwi_dir}" -name "DTI_corrected.nii.gz" -o -name "*.nii" | head -n 1)

                echo "  Found files:"
                echo "    T1 Image: ${t1_image}"
                echo "    T1 Mask: ${t1_mask}"
                echo "    DWI File: ${dwi_file}"
                
                # Check if files exist
                if [[ ! -f "${t1_image}" || ! -f "${t1_mask}" || ! -f "${dwi_file}" ]]; then
                    echo "  Required files not found, terminating script"
                    echo "FATAL ERROR: Cannot proceed without required files" >&2
                    # Using just exit 1 here
                    exit 1
                fi
                
                
                
                # Get filename base for output
                filename=$(basename ${dwi_file})
                filename_base="${filename%.*}"
                if [[ "$filename" == *.nii.gz ]]; then
                    filename_base="${filename_base%.*}"
                fi
                
                echo "  Registering T1 to DWI space..."
                echo "    T1 Image: ${t1_image}"
                echo "    T1 Mask: ${t1_mask}"
                echo "    DWI File: ${dwi_file}"
                echo "    Output Directory: ${output_dir}"
                # echo "    Filename Base: ${filename_base}"

                # Check if FLIRT has already been run
                if [[ -f "${output_dir}/T1_to_DWI.mat" && -f "${output_dir}/T1_mask_in_DTI_space.nii.gz" ]]; then
                    echo "  FLIRT registration already completed, skipping"
                    continue
                fi
                

                # Register T1 to DWI
                flirt -in "${t1_image}" \
                      -ref "${dwi_file}" \
                      -omat "${output_dir}/T1_to_DWI.mat" \
                      -dof 6
                
                # Apply transformation to the mask
                flirt -in "${t1_mask}" \
                      -ref "${dwi_file}" \
                      -applyxfm -init "${output_dir}/T1_to_DWI.mat" \
                      -out "${output_dir}/T1_mask_in_DTI_space.nii.gz" \
                      -interp nearestneighbour
                
                # Ensure binary mask
                fslmaths "${output_dir}/T1_mask_in_DTI_space.nii.gz" -bin "${output_dir}/T1_mask_in_DTI_space.nii.gz"
                
                echo "  Registration complete for ${patient} - ${tp}"

                # copy DTI_corrected.nii.gz to output directory
                cp "${dwi_file}" "${output_dir}/"
                echo "  Copied DTI_corrected.nii.gz to output directory"
                # # copy T1 mask to output directory
                # cp "${t1_mask}" "${output_dir}/"
                # echo "  Copied T1 mask to output directory"

            fi
        done
    fi
done

echo "All processing complete!"
