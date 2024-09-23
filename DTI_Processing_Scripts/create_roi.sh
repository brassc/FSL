#!/bin/bash

# Define input CSV file and output directory
csv_file="/home/cmb247/repos/FSL/Image_Processing_Scripts/included_patient_info.csv"
output_dir="/home/cmb247/repos/FSL/DTI_Processing_Scripts"
dti_data="/home/cmb247/Desktop/Project_3/BET_Extractions/"  # Path to dtifit FA image
t1_data="/path/to/T1_data"          # Path to T1 image
radius=5                            # Radius for ROI sphere

# Read the CSV file line by line, skipping the header
tail -n +2 "$csv_file" | while IFS=, read -r patient_id timepoint z anterior_x anterior_y posterior_x posterior_y baseline_anterior_x baseline_posterior_x side excluded
do
    # Skip excluded patients
    if [ "$excluded" -eq 0 ]; then
        
        # Define filenames for ROIs
        anterior_roi_file="${output_dir}/roi_${patient_id}_${timepoint}_anterior.nii.gz"
        posterior_roi_file="${output_dir}/roi_${patient_id}_${timepoint}_posterior.nii.gz"
        baseline_anterior_roi_file="${output_dir}/roi_${patient_id}_${timepoint}_baseline_anterior.nii.gz"
        baseline_posterior_roi_file="${output_dir}/roi_${patient_id}_${timepoint}_baseline_posterior.nii.gz"
        
        # Find the DTI data for the patient and timepoint
        dti_data_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/${patient_id}/dti_reg/dtifitdir/"

        dti_data="${dti_data_dir}/dti_${timepoint}_FA.nii.gz"

        # Create anterior ROI
        fslmaths "$dti_data" -mul 0 -add 1 -roi $anterior_x 1 $anterior_y 1 $z 1 0 1 -kernel sphere $radius -fmean "$anterior_roi_file"
        
        # Create posterior ROI
        fslmaths "$dti_data" -mul 0 -add 1 -roi $posterior_x 1 $posterior_y 1 $z 1 0 1 -kernel sphere $radius -fmean "$posterior_roi_file"

        # Create baseline anterior ROI
        fslmaths "$dti_data" -mul 0 -add 1 -roi $baseline_anterior_x 1 $anterior_y 1 $z 1 0 1 -kernel sphere $radius -fmean "$baseline_anterior_roi_file"
        
        # Create baseline posterior ROI
        fslmaths "$dti_data" -mul 0 -add 1 -roi $baseline_posterior_x 1 $posterior_y 1 $z 1 0 1 -kernel sphere $radius -fmean "$baseline_posterior_roi_file"
        
        # Optional: Apply these ROIs to extract FA values, use the `fslstats` command
        echo "Processing patient: $patient_id, timepoint: $timepoint"
        # Optional: Apply these ROIs to extract FA values and output in one line
        anterior_FA=$(fslstats "$dti_data" -k "$anterior_roi_file" -M) # -k flag applies the mask, -M flag outputs the mean
        posterior_FA=$(fslstats "$dti_data" -k "$posterior_roi_file" -M)
        baseline_anterior_FA=$(fslstats "$dti_data" -k "$baseline_anterior_roi_file" -M)
        baseline_posterior_FA=$(fslstats "$dti_data" -k "$baseline_posterior_roi_file" -M)

        echo "${patient_id}, ${timepoint}, ${anterior_FA}, ${posterior_FA}, ${baseline_anterior_FA}, ${baseline_posterior_FA}" >> mean_fa_values.txt

    fi
done

