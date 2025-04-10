#!/bin/bash

# Path to your CSV with coordinates
coord_csv="DTI_Processing_Scripts/LEGACY_DTI_coords_transformed_manually_adjusted.csv"

# Base directories
# Base directory for mixed IDs (letters and numbers)
mixed_base="/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI"

dwi_base="path/to/dwi/images"
fa_base="path/to/fa/maps"
md_base="path/to/md/maps"
results_dir="results"

mkdir -p $results_dir

# Create master results CSV
master_csv="$results_dir/all_metrics.csv"
echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5" > $master_csv
# Process each non-excluded patient
grep -v "^1," $coord_csv | while IFS=, read excluded patient_id timepoint rest; do
    if [ "$excluded" == "0" ]; then
        
        
        # Determine path structure based on patient_id format
        if [[ "$patient_id" =~ ^[0-9]+$ ]]; then
            continue
            # # Patient ID contains only numbers
            # dwi_path="$dwi_base/sub-${patient_id}/${timepoint}/dwi.nii.gz"
            # fa_path="$fa_base/sub-${patient_id}/${timepoint}/dti_FA.nii.gz"
            # md_path="$md_base/sub-${patient_id}/${timepoint}/dti_MD.nii.gz"
        else
            echo "Processing patient $patient_id at timepoint $timepoint"
            echo "mixed base: $mixed_base"
            
            # Patient ID contains letters and numbers
            # tp_dir=$(find "$mixed_base/$patient_id" -type d -name "*Hour-${timepoint}_*" -path
            tp_dir=$(find "$mixed_base/$patient_id" -type d -name "Sub-*" -exec find {} -type d -name "Hour-${timepoint}_*" \; | head -n 1)
            echo "tp_dir: $tp_dir"
            tp_base=$(basename "$tp_dir")
            echo "tp_base: $tp_base"
            dti_dir="${tp_dir}/dwi/proc_set1_nobzero/nipype/DATASINK/DTIspace"
            #  "/dwi/proc_set1_nobzero/nipype/DATASINK/DTIspace" | head -n 1)
            if [ -z "$tp_dir" ]; then
                echo "ERROR: Could not find directory containing timepoint $timepoint for patient $patient_id"
                exit 1
            fi
            mask_path="$dti_dir/masks/ANTS_T1_brain_mask.nii.gz"
            fa_path="$dti_dir/dti/dtifitWLS_FA.nii.gz"
            md_path="$dti_dir/dti/dtifitWLS_MD.nii.gz"
        fi
        echo "Paths:"
        echo "ANTS mask path: $mask_path"
        echo "FA path: $fa_path"
        echo "MD path: $md_path"
        

        
        # Check if files exist
        if [ -f "$mask_path" ] && [ -f "$fa_path" ] && [ -f "$md_path" ]; then
            echo "we are here"
            
            # Step 1: Create spherical ROIs
            ./DTI_Processing_Scripts/roi_create.sh "$patient_id" "$timepoint" "$tp_base" "$mask_path" "$fa_path" "$md_path"
            exit 0
            # Step 2: Extract metrics
            # ./roi_extract.sh "$patient_id" "$timepoint" "$fa_path" "$md_path" "rois/${patient_id}/${timepoint}"
            
            # Append to master CSV
            cat "results/${patient_id}_${timepoint}_metrics.csv" | tail -n 1 >> $master_csv
        else
            echo "Missing files for patient $patient_id at timepoint $timepoint"
        fi
    fi
done

echo "All processing complete. Results in $master_csv"