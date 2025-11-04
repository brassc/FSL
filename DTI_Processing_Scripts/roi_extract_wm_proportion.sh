#!/bin/bash
module load fsl

# Input parameters
patient_id=$1
timepoint=$2
tp_base=$3  # Base directory for the timepoint
bin_size=$4 # Size of the bin for the rings
num_bins=$5 # Number of bins for the rings
filter_fa_values=${6:-"true"} # Flag to enable/disable FA filtering, defaults to "true"

echo "=================================="
echo "WM PROPORTION EXTRACTION"
echo "Patient: $patient_id"
echo "Timepoint: $timepoint"
echo "Bins: ${num_bins}x${bin_size}vox"
echo "Filter FA: $filter_fa_values"
echo "=================================="

# Set ROI directories based on patient ID format
if [[ "$patient_id" =~ ^[0-9]+$ ]]; then
    # Patient ID contains only numbers (LEGACY)
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox"

    # Adjust directory name based on bin configuration
    if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox_NEW"
    elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox_NEW"
    fi
else
    # Patient ID contains letters and numbers (CENTER)
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox"

    # Adjust directory name based on bin configuration
    if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox_NEW"
    elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox_NEW"
    fi
fi

# Add filtered suffix if needed
if [ "$filter_fa_values" == "true" ]; then
    roi_dir="${roi_dir}_filtered"
fi

# Set the WM mask ROI directory
wm_roi_dir="$roi_dir/WM_mask_DTI_space"

echo "ROI directory: $roi_dir"
echo "WM ROI directory: $wm_roi_dir"

# Check if WM ROI directory exists
if [ ! -d "$wm_roi_dir" ]; then
    echo "Error: WM ROI directory not found at $wm_roi_dir"
    echo "Please run roi_WM_segmentation.sh first"
    exit 1
fi

# Define output CSV for WM proportion analysis
wm_prop_csv="DTI_Processing_Scripts/results/${patient_id}_${timepoint}_wm_proportion_analysis_${num_bins}x${bin_size}vox.csv"

if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
    wm_prop_csv="${wm_prop_csv%.csv}_NEW.csv"
elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
    wm_prop_csv="${wm_prop_csv%.csv}_NEW.csv"
fi

if [ "$filter_fa_values" = "true" ]; then
    wm_prop_csv="${wm_prop_csv%.csv}_filtered.csv"
fi

# Define master CSV for WM proportion analysis
wm_prop_master_csv="DTI_Processing_Scripts/results/all_wm_proportion_analysis_${num_bins}x${bin_size}vox.csv"

if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
    wm_prop_master_csv="${wm_prop_master_csv%.csv}_NEW.csv"
elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
    wm_prop_master_csv="${wm_prop_master_csv%.csv}_NEW.csv"
fi

if [ "$filter_fa_values" = "true" ]; then
    wm_prop_master_csv="${wm_prop_master_csv%.csv}_filtered.csv"
fi

echo "WM proportion CSV: $wm_prop_csv"
echo "WM proportion master CSV: $wm_prop_master_csv"

# Check if master CSV exists, create with header if not
if [ ! -f "$wm_prop_master_csv" ]; then
    mkdir -p $(dirname $wm_prop_master_csv)

    # Create header for rings 5, 6, 7 only
    header="patient_id,timepoint"
    for label in ant post baseline_ant baseline_post; do
        for ring in 5 6 7; do
            header="${header},WM_count_${label}_ring_${ring},total_count_${label}_ring_${ring},WM_prop_${label}_ring_${ring}"
        done
    done
    echo "$header" > "$wm_prop_master_csv"
    echo "Created WM proportion master CSV with header"
fi

# Remove individual CSV if it exists and create new one
if [ -f "$wm_prop_csv" ]; then
    rm -f "$wm_prop_csv"
fi
mkdir -p $(dirname $wm_prop_csv)

# Write header to individual CSV
header="patient_id,timepoint"
for label in ant post baseline_ant baseline_post; do
    for ring in 5 6 7; do
        header="${header},WM_count_${label}_ring_${ring},total_count_${label}_ring_${ring},WM_prop_${label}_ring_${ring}"
    done
done
echo "$header" > "$wm_prop_csv"

# Initialize data line
wm_prop_data_line="${patient_id},${timepoint}"

# Extract WM proportion data for rings 5, 6, 7
for label in ant post baseline_ant baseline_post; do
    for ring in 5 6 7; do
        echo "Processing WM proportion for ${label} ring ${ring}..."

        # Path to WM-masked ring (use FA directory as reference)
        wm_roi_ring="$wm_roi_dir/FA/${label}_ring${ring}_wm.nii.gz"

        # Path to original ring (from parent roi_dir)
        orig_roi_ring="$roi_dir/FA/${label}_ring${ring}.nii.gz"

        # Check if files exist
        if [ ! -f "$wm_roi_ring" ] || [ ! -f "$orig_roi_ring" ]; then
            echo "Warning: Required files not found for ${label} ring ${ring}"
            wm_prop_data_line="${wm_prop_data_line},NA,NA,NA"
            continue
        fi

        # Extract WM voxel count (non-zero voxels in WM-masked ring)
        wm_voxel_count=$(fslstats "$wm_roi_ring" -V | awk '{print $1}')

        # Extract total voxel count (non-zero voxels in original ring)
        total_voxel_count=$(fslstats "$orig_roi_ring" -V | awk '{print $1}')

        # Calculate WM proportion
        if [ "$total_voxel_count" != "0" ] && [ "$total_voxel_count" != "" ]; then
            wm_proportion=$(echo "scale=6; $wm_voxel_count / $total_voxel_count" | bc)
        else
            wm_proportion="NA"
        fi

        # Append to data line
        wm_prop_data_line="${wm_prop_data_line},${wm_voxel_count},${total_voxel_count},${wm_proportion}"

        echo "  WM voxels: ${wm_voxel_count}, Total voxels: ${total_voxel_count}, Proportion: ${wm_proportion}"
    done
done

# Write to individual CSV
echo "$wm_prop_data_line" >> "$wm_prop_csv"
echo "WM proportion data written to $wm_prop_csv"

# Write to master CSV
echo "$wm_prop_data_line" >> "$wm_prop_master_csv"
echo "WM proportion data appended to $wm_prop_master_csv"

echo ""
echo "WM proportion analysis complete for rings 5, 6, 7"
echo "Patient $patient_id at timepoint $timepoint processed successfully"

exit 0
