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
RADIUS=12 # Example radius for ROI, adjust as needed
num_dilations=$(( $RADIUS / 3 )) # Number of dilations for ROI sphere


# Function to transform coordinates into DTI space using Python script
transform_coordinates() {
    local t1_2_dwi_mat="$1"
    local anterior_x="$2"
    local anterior_y="$3"
    local posterior_x="$4"
    local posterior_y="$5"
    local baseline_anterior_x="$6"
    local baseline_posterior_x="$7"
    local z="$8"
    
    python3 - <<EOF
import numpy as np


# Load the transformation matrix
mat_file = '$t1_2_dwi_mat'
try: 
    transformation_matrix = np.loadtxt(mat_file)

    # Define the individual coordinates for transformation
    anterior_coords = np.array([$anterior_x, $anterior_y, $z, 1])
    posterior_coords = np.array([$posterior_x, $posterior_y, $z, 1])
    anterior_baseline_coords = np.array([$baseline_anterior_x, $anterior_y, $z, 1])
    posterior_baseline_coords = np.array([$baseline_posterior_x, $posterior_y, $z, 1])

    # Apply the transformation matrix to the anterior coordinates
    transformed_anterior = anterior_coords.dot(transformation_matrix.T)

    # Apply the transformation matrix to the posterior coordinates
    transformed_posterior = posterior_coords.dot(transformation_matrix.T)

    # Apply the transformation matrix to the baseline anterior coordinates
    transformed_baseline_anterior = anterior_baseline_coords.dot(transformation_matrix.T)

    # Apply the transformation matrix to the baseline posterior coordinates
    transformed_baseline_posterior = posterior_baseline_coords.dot(transformation_matrix.T)

    # Output the transformed anterior coordinates
    print(f"Anterior Transformed Coordinates: X={transformed_anterior[0]:.4f}, Y={transformed_anterior[1]:.4f}, Z={transformed_anterior[2]:.4f}")

    # Output the transformed posterior coordinates
    print(f"Posterior Transformed Coordinates: X={transformed_posterior[0]:.4f}, Y={transformed_posterior[1]:.4f}, Z={transformed_posterior[2]:.4f}")

    # Output the transformed baseline anterior coordinates
    print(f"Baseline Anterior Transformed Coordinates: X={transformed_baseline_anterior[0]:.4f}, Y={transformed_baseline_anterior[1]:.4f}, Z={transformed_baseline_anterior[2]:.4f}")

    # Output the transformed baseline posterior coordinates
    print(f"Baseline Posterior Transformed Coordinates: X={transformed_baseline_posterior[0]:.4f}, Y={transformed_baseline_posterior[1]:.4f}, Z={transformed_baseline_posterior[2]:.4f}")
except Exception as e:
    print(f"Error loading transformation matrix: {e}")
    exit(1)
EOF
}


# Main function
main() {
    tail -n +2 "$CSV_FILE" | head -n 2 | while IFS=, read -r excluded patient_id timepoint z anterior_x anterior_y posterior_x posterior_y side baseline_anterior_x baseline_posterior_x comments; do
        echo "CSV line: $excluded, $patient_id, $timepoint, $z, $anterior_x, $anterior_y, $posterior_x, $posterior_y, $baseline_anterior_x, $baseline_posterior_x, $side"
        

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
            # echo "transform coordinates into DTI space..."
            # get transformation matrix
            #T1_2_DWI_mat=/home/cmb247/Desktop/Project_3/BET_Extractions/19978/dti_reg/dtiregmatinv_${timepoint}.mat
            #echo "T1_2_DWI_mat: $T1_2_DWI_mat"
            
            # # transform coordinates
            # #echo "transforming anterior coordinates..."
            # anterior_x=$(echo "$anterior_x $anterior_y $z 1" | flirt -in - -applyxfm -init "$T1_2_DWI_mat" -out - | awk '{print $1}')
            # anterior_y=$(echo "$anterior_x $anterior_y $z 1" | flirt -in - -applyxfm -init "$T1_2_DWI_mat" -out - | awk '{print $2}')
            # z=$(echo "$anterior_x $anterior_y $z 1" | flirt -in - -applyxfm -init "$T1_2_DWI_mat" -out - | awk '{print $3}')
            # #echo "transforming posterior coordinates..."
            # posterior_x=$(echo "$posterior_x $posterior_y $z 1" | flirt -in - -applyxfm -init "$T1_2_DWI_mat" -out - | awk '{print $1}')
            # posterior_y=$(echo "$posterior_x $posterior_y $z 1" | flirt -in - -applyxfm -init "$T1_2_DWI_mat" -out - | awk '{print $2}')
            # #echo "transforming baseline anterior coordinates..."
            # baseline_anterior_x=$(echo "$baseline_anterior_x $anterior_y $z 1" | flirt -in - -applyxfm -init "$T1_2_DWI_mat" -out - | awk '{print $1}')
            # #echo "transforming baseline posterior coordinates..."
            # baseline_posterior_x=$(echo "$baseline_posterior_x $posterior_y $z 1" | flirt -in - -applyxfm -init "$T1_2_DWI_mat" -out - | awk '{print $1}')
            # echo "transformed coordinates: anterior_x: $anterior_x, anterior_y: $anterior_y, posterior_x: $posterior_x, posterior_y: $posterior_y, baseline_anterior_x: $baseline_anterior_x, baseline_posterior_x: $baseline_posterior_x"
            # exit
            #transform_coordinates "$T1_2_DWI_mat" "$anterior_x" "$anterior_y" "$posterior_x" "$posterior_y" "$baseline_anterior_x" "$baseline_posterior_x" "$z"
            #exit
            process_patient "$patient_id" "$timepoint" "$z" "$anterior_x" "$anterior_y" "$posterior_x" "$posterior_y" "$baseline_anterior_x" "$baseline_posterior_x"
        fi

    done

}

# finding t1 mask 
find_t1_mask() {
    local patient_id="$1"
    local timepoint="$2"
    # Echo to indicate the process is starting
    #echo "Registering to T1 scan..." 

    # Set the T1 scan directory
    t1_scan_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1w_time1_bias_corr_registered_scans/BET_Output/"

    ## MASK SEARCH

    # Search preferentially for BET mask that has been manually modified
    # Check for "fast" first to avoid ultra-fast files
    if [ "$timepoint" == "fast" ]; then
        t1_mask=$(find "$t1_scan_dir" -type f -name "*$timepoint*modified*mask*.nii.gz" ! -name "*segto*" ! -name "*ultra*")
        # If no modified mask is found, search for the non-modified mask
        if [ -z "$t1_mask" ]; then
            t1_mask=$(find "$t1_scan_dir" -type f -name "*$timepoint*mask*.nii.gz" ! -name "*segto*" ! -name "*ultra*")
        fi
    else
        # Search for modified mask first
        t1_mask=$(find "$t1_scan_dir" -type f -name "*$timepoint*modified*mask*.nii.gz" ! -name "*segto*")
        # If no modified mask is found, search for the non-modified mask
        if [ -z "$t1_mask" ]; then
            t1_mask=$(find "$t1_scan_dir" -type f -name "*$timepoint*mask*.nii.gz" ! -name "*segto*")
        fi
    fi

    # Return the mask location if found, or empty string if not
    if [[ -n "$t1_mask" ]]; then
        echo "$t1_mask"
    else
        echo ""  # Return an empty string if no mask is found
    fi
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
    local dti_data="${dti_data_dir}dtifit_${timepoint}_reg_FA.nii.gz"
    local t1_mask="/home/cmb247/Desktop/Project_3/BET_Extractions/${patient_id}/T1w_time1_bias_corr_registered_scans/BET_Output/"

    # Define filenames for ROIs
    local anterior_roi_file="${output_dir}roi_${timepoint}_anterior.nii.gz"
    local posterior_roi_file="${output_dir}roi_${timepoint}_posterior.nii.gz"
    local baseline_anterior_roi_file="${output_dir}roi_${timepoint}_baseline_anterior.nii.gz"
    local baseline_posterior_roi_file="${output_dir}roi_${timepoint}_baseline_posterior.nii.gz"
    
    t1_mask=$(find_t1_mask "$patient_id" "$timepoint")
    dti_mask="${dti_data_dir}dtifitWLS_FA_reg_${timepoint}_MASK.nii.gz"
    # binarise dti_data to create dti_mask
    fslmaths "$dti_data" -bin "$dti_mask"   
    # erode dti_mask using erode command by 2 voxel
    #fslmaths "$dti_mask" -ero "$dti_mask"

    create_spherical_roi() {
        # Arguments
        dti_mask="$1"
        x_coord="$2"
        y_coord="$3"
        z_coord="$4"
        roi_file="$5"
        radius=$RADIUS
        #t1_mask="$6"
        #echo "T1 mask: $t1_mask"
        

        # Create empty mask
        fslmaths "$dti_mask" -mul 0 "$roi_file"
        
        # Mark voxel location at (x, y, z)
        fslmaths "$roi_file" -add 1 -roi "$x_coord" 1 "$y_coord" 1 "$z_coord" 1 0 1 "$roi_file"
        # Dilate voxel to create sphere
        fslmaths "$roi_file" -kernel sphere "$radius" -fmean "${roi_file%.nii.gz}_sphere.nii.gz" -odt float
        
        # Threshold to keep only bright white areas
        fslmaths "${roi_file%.nii.gz}_sphere.nii.gz" -thr 0.0001 "$roi_file" -odt float
        #
        # Remove intermediate sphere file
        rm "${roi_file%.nii.gz}_sphere.nii.gz"

        # binarise roi mask
        fslmaths "$roi_file" -bin "$roi_file"
        
        echo "Removing portion of spherical ROI that lies outside brain..."
        # multiply roi by brain mask
        fslmaths "$roi_file" -mul "$dti_mask" "$roi_file"
        
        return
    }
    echo "Creating ROIs for $patient_id $timepoint..."
    echo "Creating anterior ROI..."
    create_spherical_roi "$dti_mask" $anterior_x $anterior_y $z $anterior_roi_file #"$dti_mask"
    echo "Completed."
    echo "Creating posterior ROI..."
    create_spherical_roi "$dti_mask" $posterior_x $posterior_y $z $posterior_roi_file #"$t1_mask"
    echo "Completed."
    echo "Creating baseline anterior ROI..."
    create_spherical_roi "$dti_mask" $baseline_anterior_x $anterior_y $z $baseline_anterior_roi_file #"$t1_mask"
    echo "Completed."
    echo "Creating baseline posterior ROI..."
    create_spherical_roi "$dti_mask" $baseline_posterior_x $posterior_y $z $baseline_posterior_roi_file #"$t1_mask"
    echo "Completed."
    echo "ROIs for $patient_id $timepoint created successfully."
    # Display ROIs in FSLeyes
    
    fsleyes "$dti_data" "$anterior_roi_file" "$posterior_roi_file" "$baseline_anterior_roi_file" "$baseline_posterior_roi_file"
    
    return
    
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
head -n 2 "$csv_file" | tail -n +2 "$csv_file" | while IFS=, read -r excluded patient_id timepoint z anterior_x anterior_y posterior_x posterior_y side baseline_anterior_x baseline_posterior_x comments
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

