#!/bin/bash
module load fsl

# Define variables
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

    # Apply the transformation matrix to the coordinates
    transformed_anterior = np.dot(transformation_matrix, anterior_coords)
    transformed_posterior = np.dot(transformation_matrix, posterior_coords)
    transformed_baseline_anterior = np.dot(transformation_matrix, anterior_baseline_coords)
    transformed_baseline_posterior = np.dot(transformation_matrix, posterior_baseline_coords)

    # Output the transformed coordinates
    print(f"{transformed_anterior[0]:.1f},{transformed_anterior[1]:.1f},{transformed_anterior[2]:.1f},{transformed_posterior[0]:.1f},{transformed_posterior[1]:.1f},{transformed_posterior[2]:.1f},{transformed_baseline_anterior[0]:.1f},{transformed_baseline_posterior[0]:.1f}")
except Exception as e:
    print(f"Error: {e}")
    exit(1)
EOF
}

# Main function
main() {
    # Read the CSV file line by line, skipping the header
    tail -n +2 "$CSV_FILE" | while IFS=, read -r excluded patient_id timepoint z anterior_x anterior_y posterior_x posterior_y side baseline_anterior_x baseline_posterior_x comments; do
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
        
        echo "Processing: Patient ID: $patient_id, Timepoint: $timepoint"
    
        # Skip excluded patients
        if [[ "$excluded" -eq 0 ]]; then
            process_patient "$patient_id" "$timepoint" "$z" "$anterior_x" "$anterior_y" "$posterior_x" "$posterior_y" "$baseline_anterior_x" "$baseline_posterior_x"
        fi
    done
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
    local base_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/${patient_id}"
    local output_dir="${base_dir}/dti_reg/rois_native/"
    mkdir -p "$output_dir"
    
    # Define paths for native DTI space data and transformation matrix
    local dti_data_dir="${base_dir}/dti_reg/dtifitdir/"
    local native_dti_data="${dti_data_dir}dtifit_${timepoint}_FA.nii.gz"  # Non-registered FA in native space
    local t1_2_dwi_mat="${base_dir}/dti_reg/dtiregmatinv_${timepoint}.mat"  # Inverse transformation matrix
    
    # Check if transformation matrix exists
    if [[ ! -f "$t1_2_dwi_mat" ]]; then
        echo "ERROR: Transformation matrix not found: $t1_2_dwi_mat"
        return 1
    fi
    
    # Check if native DTI data exists
    if [[ ! -f "$native_dti_data" ]]; then
        echo "ERROR: Native DTI data not found: $native_dti_data"
        return 1
    fi
    
    # Transform coordinates from T1 space to native DTI space
    echo "Transforming coordinates from T1 space to native DTI space..."
    transformed_coords=$(transform_coordinates "$t1_2_dwi_mat" "$anterior_x" "$anterior_y" "$posterior_x" "$posterior_y" "$baseline_anterior_x" "$baseline_posterior_x" "$z")
    
    # Check if transformation was successful
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to transform coordinates"
        return 1
    fi
    
    # Parse the transformed coordinates
    IFS=',' read -r ant_x ant_y ant_z post_x post_y post_z base_ant_x base_post_x <<< "$transformed_coords"
    
    echo "Transformed coordinates:"
    echo "  Anterior: ($ant_x, $ant_y, $ant_z)"
    echo "  Posterior: ($post_x, $post_y, $post_z)"
    echo "  Baseline Anterior X: $base_ant_x"
    echo "  Baseline Posterior X: $base_post_x"
    
    # Define filenames for ROIs in native space
    local anterior_roi_file="${output_dir}roi_${timepoint}_anterior_native.nii.gz"
    local posterior_roi_file="${output_dir}roi_${timepoint}_posterior_native.nii.gz"
    local baseline_anterior_roi_file="${output_dir}roi_${timepoint}_baseline_anterior_native.nii.gz"
    local baseline_posterior_roi_file="${output_dir}roi_${timepoint}_baseline_posterior_native.nii.gz"
    
    # Create a brain mask in native DTI space
    local dti_mask="${output_dir}dti_mask_${timepoint}_native.nii.gz"
    fslmaths "$native_dti_data" -bin "$dti_mask"
    
    # Function to create spherical ROI in native space
    create_spherical_roi() {
        local dti_mask="$1"
        local x_coord="$2"
        local y_coord="$3"
        local z_coord="$4"
        local roi_file="$5"
        local radius=$RADIUS
        
        echo "Creating ROI at ($x_coord, $y_coord, $z_coord) with radius $radius..."
        
        # Create empty mask
        fslmaths "$dti_mask" -mul 0 "$roi_file"
        
        # Mark voxel location at (x, y, z)
        fslmaths "$roi_file" -add 1 -roi "$x_coord" 1 "$y_coord" 1 "$z_coord" 1 0 1 "$roi_file"
        
        # Dilate voxel to create sphere
        fslmaths "$roi_file" -kernel sphere "$radius" -fmean "${roi_file%.nii.gz}_sphere.nii.gz" -odt float
        
        # Threshold to keep only bright white areas
        fslmaths "${roi_file%.nii.gz}_sphere.nii.gz" -thr 0.0001 "$roi_file" -odt float
        
        # Remove intermediate sphere file
        rm "${roi_file%.nii.gz}_sphere.nii.gz"
        
        # Binarize ROI mask
        fslmaths "$roi_file" -bin "$roi_file"
        
        # Restrict ROI to brain mask
        fslmaths "$roi_file" -mul "$dti_mask" "$roi_file"
        
        echo "ROI created: $roi_file"
    }
    
    echo "Creating ROIs for $patient_id $timepoint in native DTI space..."
    create_spherical_roi "$dti_mask" $ant_x $ant_y $ant_z $anterior_roi_file
    create_spherical_roi "$dti_mask" $post_x $post_y $post_z $posterior_roi_file
    create_spherical_roi "$dti_mask" $base_ant_x $ant_y $ant_z $baseline_anterior_roi_file
    create_spherical_roi "$dti_mask" $base_post_x $post_y $post_z $baseline_posterior_roi_file
    
    # Extract and log FA values
    extract_and_log_fa "$native_dti_data" "$anterior_roi_file" "$posterior_roi_file" "$baseline_anterior_roi_file" "$baseline_posterior_roi_file" "$output_dir" "$patient_id" "$timepoint"
    
    # Display ROIs in FSLeyes
    echo "Opening FSLeyes to display ROIs..."
    fsleyes "$native_dti_data" "$anterior_roi_file" -cm red -a 0.5 \
        "$posterior_roi_file" -cm blue -a 0.5 \
        "$baseline_anterior_roi_file" -cm green -a 0.5 \
        "$baseline_posterior_roi_file" -cm yellow -a 0.5
}

# Function to extract all FA values and save as pickle file
extract_and_log_fa() {
    local dti_data="$1"
    local anterior_roi_file="$2"
    local posterior_roi_file="$3"
    local baseline_anterior_roi_file="$4"
    local baseline_posterior_roi_file="$5"
    local output_dir="$6"
    local patient_id="$7"
    local timepoint="$8"

    local log_file="${output_dir}native_mean_fa_values.txt"
    if [[ ! -f "$log_file" ]]; then
        printf "Patient ID, Timepoint, Anterior FA, Posterior FA, Baseline Anterior FA, Baseline Posterior FA\n" > "$log_file"
    fi
    
    # Create Python script to extract all values
    local python_script="${output_dir}extract_all_values.py"
    
    cat > "$python_script" << 'EOL'
#!/usr/bin/env python3
import sys
import os
import numpy as np
import nibabel as nib
import pickle
from collections import defaultdict

def extract_all_values(fa_file, md_file, roi_files, labels, patient_id, timepoint, output_dir):
    # Load FA and MD data
    fa_img = nib.load(fa_file)
    fa_data = fa_img.get_fdata()
    
    md_img = nib.load(md_file)
    md_data = md_img.get_fdata()
    
    # Create dictionary to store results
    results = {
        'patient_id': patient_id,
        'timepoint': timepoint,
        'fa': {
            'roi_stats': defaultdict(dict),
            'all_values': defaultdict(list)
        },
        'md': {
            'roi_stats': defaultdict(dict),
            'all_values': defaultdict(list)
        }
    }
    
    # Process each ROI
    for roi_file, label in zip(roi_files, labels):
        # Load ROI mask
        roi_img = nib.load(roi_file)
        roi_data = roi_img.get_fdata()
        
        # Extract values within ROI (where mask is non-zero)
        mask = roi_data > 0
        
        # Process FA values
        fa_values = fa_data[mask]
        if len(fa_values) > 0:
            # Store FA statistics
            results['fa']['roi_stats'][label]['mean'] = float(np.mean(fa_values))
            results['fa']['roi_stats'][label]['median'] = float(np.median(fa_values))
            results['fa']['roi_stats'][label]['std'] = float(np.std(fa_values))
            results['fa']['roi_stats'][label]['min'] = float(np.min(fa_values))
            results['fa']['roi_stats'][label]['max'] = float(np.max(fa_values))
            results['fa']['roi_stats'][label]['n_voxels'] = int(len(fa_values))
            
            # Store all FA values
            results['fa']['all_values'][label] = fa_values.tolist()
        else:
            print(f"Warning: No voxels found in ROI {label}")
            results['fa']['roi_stats'][label]['mean'] = 0.0
            results['fa']['roi_stats'][label]['median'] = 0.0
            results['fa']['roi_stats'][label]['std'] = 0.0
            results['fa']['roi_stats'][label]['min'] = 0.0
            results['fa']['roi_stats'][label]['max'] = 0.0
            results['fa']['roi_stats'][label]['n_voxels'] = 0
            results['fa']['all_values'][label] = []
        
        # Process MD values
        md_values = md_data[mask]
        if len(md_values) > 0:
            # Store MD statistics
            results['md']['roi_stats'][label]['mean'] = float(np.mean(md_values))
            results['md']['roi_stats'][label]['median'] = float(np.median(md_values))
            results['md']['roi_stats'][label]['std'] = float(np.std(md_values))
            results['md']['roi_stats'][label]['min'] = float(np.min(md_values))
            results['md']['roi_stats'][label]['max'] = float(np.max(md_values))
            results['md']['roi_stats'][label]['n_voxels'] = int(len(md_values))
            
            # Store all MD values
            results['md']['all_values'][label] = md_values.tolist()
        else:
            print(f"Warning: No voxels found in ROI {label} for MD")
            results['md']['roi_stats'][label]['mean'] = 0.0
            results['md']['roi_stats'][label]['median'] = 0.0
            results['md']['roi_stats'][label]['std'] = 0.0
            results['md']['roi_stats'][label]['min'] = 0.0
            results['md']['roi_stats'][label]['max'] = 0.0
            results['md']['roi_stats'][label]['n_voxels'] = 0
            results['md']['all_values'][label] = []
    
    # Save results as pickle file
    pickle_file = os.path.join(output_dir, f"{patient_id}_{timepoint}_dti_values.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Output statistics to text files
    # FA stats
    fa_stats_file = os.path.join(output_dir, f"roi_stats_{patient_id}_{timepoint}_FA.txt")
    with open(fa_stats_file, 'w') as f:
        f.write("ROI,Mean,Median,StdDev,Min,Max,NumVoxels\n")
        for label in results['fa']['roi_stats']:
            stats = results['fa']['roi_stats'][label]
            f.write(f"{label},{stats['mean']:.6f},{stats['median']:.6f},{stats['std']:.6f},"
                   f"{stats['min']:.6f},{stats['max']:.6f},{stats['n_voxels']}\n")
    
    # MD stats
    md_stats_file = os.path.join(output_dir, f"roi_stats_{patient_id}_{timepoint}_MD.txt")
    with open(md_stats_file, 'w') as f:
        f.write("ROI,Mean,Median,StdDev,Min,Max,NumVoxels\n")
        for label in results['md']['roi_stats']:
            stats = results['md']['roi_stats'][label]
            f.write(f"{label},{stats['mean']:.6f},{stats['median']:.6f},{stats['std']:.6f},"
                   f"{stats['min']:.6f},{stats['max']:.6f},{stats['n_voxels']}\n")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: extract_all_values.py <fa_file> <md_file> <anterior_roi> <posterior_roi> <baseline_anterior_roi> <baseline_posterior_roi> <output_dir> <patient_id> <timepoint>")
        sys.exit(1)
    
    fa_file = sys.argv[1]
    md_file = sys.argv[2]
    anterior_roi = sys.argv[3]
    posterior_roi = sys.argv[4]
    baseline_anterior_roi = sys.argv[5]
    baseline_posterior_roi = sys.argv[6]
    output_dir = sys.argv[7]
    patient_id = sys.argv[8]
    timepoint = sys.argv[9]
    
    roi_files = [anterior_roi, posterior_roi, baseline_anterior_roi, baseline_posterior_roi]
    labels = ['anterior', 'posterior', 'baseline_anterior', 'baseline_posterior']
    
    results = extract_all_values(fa_file, md_file, roi_files, labels, patient_id, timepoint, output_dir)
    
    # Print FA and MD mean values for the bash script to capture (FA first, then MD)
    fa_means = f"{results['fa']['roi_stats']['anterior']['mean']},{results['fa']['roi_stats']['posterior']['mean']}," \
               f"{results['fa']['roi_stats']['baseline_anterior']['mean']},{results['fa']['roi_stats']['baseline_posterior']['mean']}"
    md_means = f"{results['md']['roi_stats']['anterior']['mean']},{results['md']['roi_stats']['posterior']['mean']}," \
               f"{results['md']['roi_stats']['baseline_anterior']['mean']},{results['md']['roi_stats']['baseline_posterior']['mean']}"
    print(f"{fa_means}|{md_means}")
EOL
    
    # Make the Python script executable
    chmod +x "$python_script"
    
    # Find MD file
    local native_md_data="${dti_data_dir}dtifit_${timepoint}_MD.nii.gz"  # Non-registered MD in native space
    
    # Check if MD file exists
    if [[ ! -f "$native_md_data" ]]; then
        echo "ERROR: Native MD data not found: $native_md_data"
        return 1
    fi
    
    # Run the Python script
    echo "Extracting all FA and MD values using Python..."
    local values=$(python3 "$python_script" "$dti_data" "$native_md_data" "$anterior_roi_file" "$posterior_roi_file" "$baseline_anterior_roi_file" "$baseline_posterior_roi_file" "$output_dir" "$patient_id" "$timepoint")
    
    # Parse the output to get mean values for the log file
    IFS='|' read -r fa_values md_values <<< "$values"
    
    # Extract FA values
    IFS=',' read -r anterior_FA posterior_FA baseline_anterior_FA baseline_posterior_FA <<< "$fa_values"
    
    # Extract MD values
    IFS=',' read -r anterior_MD posterior_MD baseline_anterior_MD baseline_posterior_MD <<< "$md_values"
    
    # Log FA mean values to text file
    printf "%s, %s, %s, %s, %s, %s\n" "$patient_id" "$timepoint" "$anterior_FA" "$posterior_FA" "$baseline_anterior_FA" "$baseline_posterior_FA" >> "${log_file}"
    
    # Create MD log file if it doesn't exist
    local md_log_file="${output_dir}native_mean_md_values.txt"
    if [[ ! -f "$md_log_file" ]]; then
        printf "Patient ID, Timepoint, Anterior MD, Posterior MD, Baseline Anterior MD, Baseline Posterior MD\n" > "$md_log_file"
    fi
    
    # Log MD mean values to text file
    printf "%s, %s, %s, %s, %s, %s\n" "$patient_id" "$timepoint" "$anterior_MD" "$posterior_MD" "$baseline_anterior_MD" "$baseline_posterior_MD" >> "${md_log_file}"
    
    echo "DTI values extracted and saved to pickle file: ${output_dir}${patient_id}_${timepoint}_dti_values.pkl"
    echo "FA statistics saved to: ${output_dir}roi_stats_${patient_id}_${timepoint}_FA.txt"
    echo "MD statistics saved to: ${output_dir}roi_stats_${patient_id}_${timepoint}_MD.txt"
}

# Start the script
main
echo "Script completed."