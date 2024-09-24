#!/bin/bash
module load fsl
# Define input CSV file and output directory
#csv_file="/home/cmb247/repos/FSL/Image_Processing_Scripts/included_patient_info.csv"
#output_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/"
#dti_data="/home/cmb247/Desktop/Project_3/BET_Extractions/"  # Path to dtifit FA image
#t1_data="/path/to/T1_data"          # Path to T1 image
#radius=9                            # Radius for ROI sphere in multiple of 3 (voxel size)


# Ensure CSV file is passed as argument
#if [[ -z "$1" ]]; then
#    printf "Usage: %s <csv_file>\n" "$0" >&2
#    exit 1
#fi

CSV_FILE="/home/cmb247/repos/FSL/Image_Processing_Scripts/included_patient_info.csv"
RADIUS=9 # Example radius for ROI, adjust as needed

# Main function
main() {
    tail -n +2 "$CSV_FILE" | -head -n 1 | while IFS=, read -r excluded patient_id timepoint z anterior_x anterior_y posterior_x posterior_y side baseline_anterior_x baseline_posterior_x comments; do

        # Trim spaces from all variables
        excluded=$(printf "%s" "$excluded" | xargs)
        patient_id=$(printf "%s" "$patient_id" | xargs)
        timepoint=$(printf "%s" "$timepoint" | xargs)
        z=$(printf "%s" "$z" | xargs)
        anterior_x=$(printf "%s" "$anterior_x" | xargs)
        anterior_y=$(printf "%s" "$anterior_y" | xargs)
        posterior_x=$(printf "%s" "$posterior_x" | xargs)
        posterior_y=$(printf "%s" "$posterior_y" | xargs)
        baseline_anterior_x=$(printf "%s" "$baseline_anterior_x" | xargs)
        baseline_posterior_x=$(printf "%s" "$baseline_posterior_x" | xargs)
        side=$(printf "%s" "$side" | xargs)
        
        printf "Excluded: %s, Patient ID: %s, Timepoint: %s, Z: %s, Anterior X: %s, Anterior Y: %s, Posterior X: %s, Posterior Y: %s, Baseline Anterior X: %s, Baseline Posterior X: %s, Side: %s\n" \
            "$excluded" "$patient_id" "$timepoint" "$z" "$anterior_x" "$anterior_y" "$posterior_x" "$posterior_y" "$baseline_anterior_x" "$baseline_posterior_x" "$side"
    
        # Skip excluded patients
        if [[ "$excluded" -eq 0 ]]; then
            #printf "Excluded: %s, Patient ID: %s, Timepoint: %s, Z: %s, Anterior X: %s, Anterior Y: %s, Posterior X: %s, Posterior Y: %s, Baseline Anterior X: %s, Baseline Posterior X: %s, Side: %s\n" \
            #    "$excluded" "$patient_id" "$timepoint" "$z" "$anterior_x" "$anterior_y" "$posterior_x" "$posterior_y" "$baseline_anterior_x" "$baseline_posterior_x" "$side"

            process_patient "$patient_id" "$timepoint" "$z" "$anterior_x" "$anterior_y" "$posterior_x" "$posterior_y" "$baseline_anterior_x" "$baseline_posterior_x"
        fi

    done
exit
}

# Function to process each patient
process_patient() {
    local patient_id="$1"
    local timepoint="$2"
    local z="$3"
    local anterior_x="$4"
    local anterior_y="$5"
    local posterior_x="$6"
    local posterior_y="$7"
    local baseline_anterior_x="$8"
    local baseline_posterior_x="$9"

    # Define directories and files
    local output_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/${patient_id}/dti_reg/rois/"
    mkdir -p "$output_dir"
    local dti_data_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/${patient_id}/dti_reg/dtifitdir/"
    local dti_data="${dti_data_dir}dti_${timepoint}_FA.nii.gz"

    # Define filenames for ROIs
    local anterior_roi_file="${output_dir}roi_${timepoint}_anterior.nii.gz"
    local posterior_roi_file="${output_dir}roi_${timepoint}_posterior.nii.gz"
    local baseline_anterior_roi_file="${output_dir}roi_${timepoint}_baseline_anterior.nii.gz"
    local baseline_posterior_roi_file="${output_dir}roi_${timepoint}_baseline_posterior.nii.gz"

    # Create anterior ROI
    fslmaths "$dti_data" -mul 0 -add 1 -roi "$anterior_x" 1 "$anterior_y" 1 "$z" 1 0 1 -kernel sphere "$RADIUS" -fmean "$anterior_roi_file"
    return
    # Create posterior ROI
    fslmaths "$dti_data" -mul 0 -add 1 -roi "$posterior_x" 1 "$posterior_y" 1 "$z" 1 0 1 -kernel sphere "$RADIUS" -fmean "$posterior_roi_file"

    # Create baseline anterior ROI
    fslmaths "$dti_data" -mul 0 -add 1 -roi "$baseline_anterior_x" 1 "$anterior_y" 1 "$z" 1 0 1 -kernel sphere "$RADIUS" -fmean "$baseline_anterior_roi_file"

    # Create baseline posterior ROI
    fslmaths "$dti_data" -mul 0 -add 1 -roi "$baseline_posterior_x" 1 "$posterior_y" 1 "$z" 1 0 1 -kernel sphere "$RADIUS" -fmean "$baseline_posterior_roi_file"

    # Extract and log FA values
    extract_and_log_fa "$dti_data" "$anterior_roi_file" "$posterior_roi_file" "$baseline_anterior_roi_file" "$baseline_posterior_roi_file" "$output_dir" "$patient_id" "$timepoint"
}

# Function to extract FA values and log them
extract_and_log_fa() {
    local dti_data="$1"
    local anterior_roi_file="$2"
    local posterior_roi_file="$3"
    local baseline_anterior_roi_file="$4"
    local baseline_posterior_roi_file="$5"
    local output_dir="$6"
    local patient_id="$7"
    local timepoint="$8"

    local log_file="${output_dir}mean_fa_values.txt"
    if [[ ! -f "$log_file" ]]; then
        printf "Patient ID, Timepoint, Anterior FA, Posterior FA, Baseline Anterior FA, Baseline Posterior FA\n" > "$log_file"
    fi

    # Extract FA values
    local anterior_FA; anterior_FA=$(fslstats "$dti_data" -k "$anterior_roi_file" -M)
    local posterior_FA; posterior_FA=$(fslstats "$dti_data" -k "$posterior_roi_file" -M)
    local baseline_anterior_FA; baseline_anterior_FA=$(fslstats "$dti_data" -k "$baseline_anterior_roi_file" -M)
    local baseline_posterior_FA; baseline_posterior_FA=$(fslstats "$dti_data" -k "$baseline_posterior_roi_file" -M)

    # Log results
    printf "%s, %s, %s, %s, %s, %s\n" "$patient_id" "$timepoint" "$anterior_FA" "$posterior_FA" "$baseline_anterior_FA" "$baseline_posterior_FA" >> "${log_file}"
}

# Start the script
main


exit
# Read the CSV file line by line, skipping the header
#excluded?, patient ID, timepoint, z coord (slice), anterior x coord, anterior y coord, posterior x coord, posterior y coord, side (L/R), baseline anterior x coord, baseline posterior x coord, COMMENTS

tail -n +2 "$csv_file" | while IFS=, read -r excluded patient_id timepoint z anterior_x anterior_y posterior_x posterior_y side baseline_anterior_x baseline_posterior_x comments
echo "Excluded: $excluded, Patient ID: $patient_id, Timepoint: $timepoint, Z: $z, Anterior X: $anterior_x, Anterior Y: $anterior_y, Posterior X: $posterior_x, Posterior Y: $posterior_y, Baseline Anterior X: $baseline_anterior_x, Baseline Posterior X: $baseline_posterior_x, Side: $side"

do
    # Skip excluded patients
    if [ "$excluded" -eq 0 ]; then
     
        # defome output directory
        output_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/"${patient_id}"/dti_reg/rois/"
        if [ ! -d "$output_dir" ]; then
            mkdir -p "$output_dir"
        fi
        
        # Define filenames for ROIs
        anterior_roi_file="${output_dir}roi_${timepoint}_anterior.nii.gz"
        posterior_roi_file="${output_dir}roi_${timepoint}_posterior.nii.gz"
        baseline_anterior_roi_file="${output_dir}roi_${timepoint}_baseline_anterior.nii.gz"
        baseline_posterior_roi_file="${output_dir}roi_${timepoint}_baseline_posterior.nii.gz"
        
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

        echo "${patient_id}, ${timepoint}, ${anterior_FA}, ${posterior_FA}, ${baseline_anterior_FA}, ${baseline_posterior_FA}" >> "${output_dir}mean_fa_values.txt"

    fi

done

