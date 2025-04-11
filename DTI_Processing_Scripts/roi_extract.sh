#!/bin/bash

# Input parameters
patient_id=$1
timepoint=$2
tp_base=$3  # Base directory for the timepoint
fa_map=$4  # Path to FA map
md_map=$5  # Path to MD map

# Set ROI dir
# Create spherical ROIs for each point
if [[ "$patient_id" =~ ^[0-9]+$ ]]; then
    # Patient ID contains only numbers
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files"
else
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files"
fi

exit 0
# Output CSV file
output_csv="DTI_Processing_Scripts/results/${patient_id}_${timepoint}_metrics.csv"
mkdir -p $(dirname $output_csv)

# Initialize CSV header
echo "patient_id,timepoint,FA_anterior_ring_1,FA_posterior_ring_1,FA_baseline_ant_ring_1,FA_baseline_post_ring_1,FA_anterior_ring_2,..." > $output_csv

# Extract mean values from each ROI
# For FA
fa_ant_r1=$(fslstats $fa_map -k $roi_dir/ant_ring1 -M)
fa_post_r1=$(fslstats $fa_map -k $roi_dir/post_ring1 -M)
fa_base_ant_r1=$(fslstats $fa_map -k $roi_dir/baseline_ant_ring1 -M)
fa_base_post_r1=$(fslstats $fa_map -k $roi_dir/baseline_post_ring1 -M)
# ... repeat for all rings

# For MD (same approach)
md_ant_r1=$(fslstats $md_map -k $roi_dir/ant_ring1 -M)
# ... and so on

# Write values to CSV
echo "$patient_id,$timepoint,$fa_ant_r1,$fa_post_r1,$fa_base_ant_r1,$fa_base_post_r1,..." >> $output_csv