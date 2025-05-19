#!/bin/bash
module load fsl
# Input parameters
patient_id=$1
timepoint=$2
tp_base=$3  # Base directory for the timepoint
bin_size=$4 # Size of the bin for the rings
num_bins=$5 # Number of bins for the rings
master_csv=$6 # Path to the master CSV file
filter_fa_values=${7:-"true"} # Flag to enable/disable FA filtering, defaults to "true"
get_all_values=${8:-"false"} # Flag to enable/disable getting all values, defaults to "false"

# Set ROI dir
if [[ "$patient_id" =~ ^[0-9]+$ ]]; then
    # Patient ID contains only numbers
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox"
    # if $ num bins =5 and $ bin size =4 set roi_dir to the same as above but NEW
    if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox_NEW"
    elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox_NEW"
    fi
else
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox"
    # if $ num bins =5 and $ bin size =4 set roi_dir to the same as above but NEW
    if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox_NEW"
    elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox_NEW"
    fi
fi

# if filtered_fa_values is true, append to roi_dir
if [ "$filter_fa_values" == "true" ]; then
    roi_dir="${roi_dir}_filtered"
fi

# Set the WM mask ROI directory
wm_roi_dir="$roi_dir/WM_mask_DTI_space"

echo "ROI directory: $roi_dir"
echo "WM ROI directory: $wm_roi_dir"

if [ ! -d "$wm_roi_dir" ]; then
    echo "Error: WM ROI directory not found at $wm_roi_dir"
    echo "Please run roi_WM_segmentation.sh first"
    exit 1
fi

# Output CSV file for WM metrics
output_csv="DTI_Processing_Scripts/results/${patient_id}_${timepoint}_metrics_${num_bins}x${bin_size}vox_wm.csv"

if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
    output_csv="${output_csv%.csv}_NEW.csv"
elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
    output_csv="${output_csv%.csv}_NEW.csv"
fi

# Modify output_csv name if filtering is enabled
if [ "$filter_fa_values" = "true" ]; then
    output_csv="${output_csv%.csv}_filtered.csv"
fi

if [ "$get_all_values" = "true" ]; then
    output_csv="${output_csv%.csv}_all_values.csv"
fi

# Define master CSV for WM metrics
wm_master_csv="${master_csv%.csv}_wm.csv"
echo "WM Master CSV: $wm_master_csv"

# Check if master WM CSV exists, create if not
if [ ! -f "$wm_master_csv" ]; then
    mkdir -p $(dirname $wm_master_csv)
    
    # Initialize CSV header based on number of bins
    if [ $num_bins -eq 10 ]; then
        echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_anterior_ring_6,FA_anterior_ring_7,FA_anterior_ring_8,FA_anterior_ring_9,FA_anterior_ring_10,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_posterior_ring_6,FA_posterior_ring_7,FA_posterior_ring_8,FA_posterior_ring_9,FA_posterior_ring_10,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_anterior_ring_6,FA_baseline_anterior_ring_7,FA_baseline_anterior_ring_8,FA_baseline_anterior_ring_9,FA_baseline_anterior_ring_10,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,FA_baseline_posterior_ring_6,FA_baseline_posterior_ring_7,FA_baseline_posterior_ring_8,FA_baseline_posterior_ring_9,FA_baseline_posterior_ring_10,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_anterior_ring_6,MD_anterior_ring_7,MD_anterior_ring_8,MD_anterior_ring_9,MD_anterior_ring_10,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_posterior_ring_6,MD_posterior_ring_7,MD_posterior_ring_8,MD_posterior_ring_9,MD_posterior_ring_10,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_anterior_ring_6,MD_baseline_anterior_ring_7,MD_baseline_anterior_ring_8,MD_baseline_anterior_ring_9,MD_baseline_anterior_ring_10,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5,MD_baseline_posterior_ring_6,MD_baseline_posterior_ring_7,MD_baseline_posterior_ring_8,MD_baseline_posterior_ring_9,MD_baseline_posterior_ring_10" > $wm_master_csv
    elif [ $num_bins -eq 5 ]; then
        echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5" > $wm_master_csv
    else
        echo "Invalid number of bins specified. Please set num_bins to either 5 or 10."
        exit 1
    fi
fi

# if output_csv already exists, remove it
if [ -f "$output_csv" ]; then
    rm -f "$output_csv"
fi

# create new
mkdir -p $(dirname $output_csv)

echo "Output CSV: $output_csv"
# Initialize CSV header
if [ $num_bins -eq 10 ]; then
    echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_anterior_ring_6,FA_anterior_ring_7,FA_anterior_ring_8,FA_anterior_ring_9,FA_anterior_ring_10,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_posterior_ring_6,FA_posterior_ring_7,FA_posterior_ring_8,FA_posterior_ring_9,FA_posterior_ring_10,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_anterior_ring_6,FA_baseline_anterior_ring_7,FA_baseline_anterior_ring_8,FA_baseline_anterior_ring_9,FA_baseline_anterior_ring_10,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,FA_baseline_posterior_ring_6,FA_baseline_posterior_ring_7,FA_baseline_posterior_ring_8,FA_baseline_posterior_ring_9,FA_baseline_posterior_ring_10,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_anterior_ring_6,MD_anterior_ring_7,MD_anterior_ring_8,MD_anterior_ring_9,MD_anterior_ring_10,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_posterior_ring_6,MD_posterior_ring_7,MD_posterior_ring_8,MD_posterior_ring_9,MD_posterior_ring_10,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_anterior_ring_6,MD_baseline_anterior_ring_7,MD_baseline_anterior_ring_8,MD_baseline_anterior_ring_9,MD_baseline_anterior_ring_10,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5,MD_baseline_posterior_ring_6,MD_baseline_posterior_ring_7,MD_baseline_posterior_ring_8,MD_baseline_posterior_ring_9,MD_baseline_posterior_ring_10" > $output_csv
elif [ $num_bins -eq 5 ]; then
    echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5" > $output_csv
else
    echo "Invalid number of bins specified. Please set num_bins to either 5 or 10."
    exit 1
fi

# Initialize the data line with patient_id and timepoint
data_line="${patient_id},${timepoint}"

# Function to extract FA values with filtering (only used if flag is true)
extract_filtered_fa_value() {
    local fa_file=$1
    local mask_file=$2
    
    # Create a temporary mask that only includes voxels with values between 0.05 and 1
    local valid_range_mask="${fa_file%.nii.gz}_valid_range_mask_temp.nii.gz"
    
    # Create mask of valid values (between 0.05 and 1)
    fslmaths "$fa_file" -thr 0.05 -uthr 1 -bin "$valid_range_mask"
    
    # Combine with the input mask (logical AND)
    local combined_mask="${fa_file%.nii.gz}_combined_mask_temp.nii.gz"
    fslmaths "$valid_range_mask" -mul "$mask_file" "$combined_mask"
    
    # Calculate mean using the original data but with the combined mask
    local result=$(fslmeants -i "$fa_file" -m "$combined_mask")
    
    # Clean up temporary files
    rm -f "$valid_range_mask" "$combined_mask"
    
    echo "$result"
}

# Function to extract MD values with filtering based on FA values (used only if flag is true)
extract_filtered_md_value() {
    local fa_file=$1
    local md_file=$2
    local mask_file=$3
    
    # Create a temporary mask that only includes voxels with FA values between 0.05 and 1
    local valid_range_mask="${fa_file%.nii.gz}_valid_range_mask_temp.nii.gz"
    
    # Create mask of valid values (between 0.05 and 1)
    fslmaths "$fa_file" -thr 0.05 -uthr 1 -bin "$valid_range_mask"
    
    # Combine with the input mask (logical AND)
    local combined_mask="${fa_file%.nii.gz}_combined_mask_temp.nii.gz"
    fslmaths "$valid_range_mask" -mul "$mask_file" "$combined_mask"
    
    # Calculate mean using the MD data but with the combined mask
    local result=$(fslmeants -i "$md_file" -m "$combined_mask")
    
    # Clean up temporary files
    rm -f "$valid_range_mask" "$combined_mask"
    
    echo "$result"
}

# Iterate through parameters (FA, MD)
for parameter in FA MD; do
    echo "Extracting $parameter values from WM-masked ROIs..."
    
    # Iterate through all labels
    for label in ant post baseline_ant baseline_post; do
        # Process all bins for this parameter/label combination
        for ((i=1; i<=$num_bins; i++)); do
            # Check if the WM-masked files exist
            wm_roi="${wm_roi_dir}/$parameter/${label}_ring${i}_wm.nii.gz"
            wm_metric="${wm_roi_dir}/$parameter/${label}_ring${i}_${parameter}_wm.nii.gz"
            
            if [ ! -f "$wm_roi" ] || [ ! -f "$wm_metric" ]; then
                echo "Warning: Required WM files not found for $label ring $i"
                data_line="${data_line},NA"
                continue
            fi
            
            # Handle extraction based on user preferences
            if [ $get_all_values = 'false' ]; then
                if [ "$parameter" = "FA" ] && [ "$filter_fa_values" = "true" ]; then
                    # For FA with filtering
                    mean_value=$(extract_filtered_fa_value "$wm_metric" "$wm_roi")
                elif [ "$parameter" = "MD" ] && [ "$filter_fa_values" = "true" ]; then
                    # For MD with FA filtering
                    mean_value=$(extract_filtered_md_value "$wm_roi_dir/FA/${label}_ring${i}_FA_wm.nii.gz" "$wm_metric" "$wm_roi")
                else 
                    # Simple mean extraction
                    mean_value=$(fslmeants -i "$wm_metric" -m "$wm_roi")
                fi
                data_line="${data_line},${mean_value}"
            else
                # Get all values using showall option
                fslmeants -i "$wm_metric" -m "$wm_roi" -o DTI_Processing_Scripts/results/temp_wm.csv --showall
            
                # Extract values, removing empty lines
                values=$(cat DTI_Processing_Scripts/results/temp_wm.csv | grep -v "^$" | tail -1)
                
                # Format as array by wrapping in quotes and brackets
                value_array="\"[${values}]\""
                
                data_line="${data_line},${value_array}"
            fi
        done
    done
done

# Clean up temporary files
rm -f DTI_Processing_Scripts/results/temp_wm.csv

# Write data to CSV
echo "Writing data to CSV..."
echo "$data_line" >> $output_csv

# Write to master CSV
echo "Writing data to master WM CSV: $wm_master_csv"
echo "$data_line" >> "$wm_master_csv"

# After appending
echo "Verifying append operation..."
tail -n 1 "$wm_master_csv"

echo "WM ROI extraction complete for patient $patient_id at timepoint $timepoint"
echo "Results saved to $output_csv and $wm_master_csv"
exit 0