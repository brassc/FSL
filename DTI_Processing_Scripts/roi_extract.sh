#!/bin/bash

# Input parameters
patient_id=$1
timepoint=$2
fa_map=$3  # Path to FA map
md_map=$4  # Path to MD map
roi_dir=$5 # Directory containing ROIs

# Output CSV file
output_csv="results/${patient_id}_${timepoint}_metrics.csv"
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