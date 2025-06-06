#!/bin/bash

# Load required modules
module load fsl
module load parallel 2>/dev/null || true  # Try to load parallel module, continue even if it fails


# # Parse command line arguments
# if [ $# -lt 2 ]; then
#     echo "Usage: $0 <num_bins> <bin_size>"
#     echo "Example: $0 5 4"
#     exit 1
# fi

# num_bins=$1
# bin_size=$2

# Parse command line arguments
num_bins=""
bin_size=""
num_cpus=""

for arg in "$@"; do
    case $arg in
        --num_bins=*)
        num_bins="${arg#*=}"
        ;;
        --bin_size=*)
        bin_size="${arg#*=}"
        ;;
        --num_cpus=*)
        num_cpus="${arg#*=}"
        ;;
        *)
        # Unknown option
        ;;
    esac
done

# Check if required parameters are provided
if [ -z "$num_bins" ] || [ -z "$bin_size" ] || [ -z "$num_cpus" ] ; then
    echo "Usage: $0 --num_bins=<value> --bin_size=<value> --num_cpus=<value>"
    echo "Example: $0 --num_bins=5 --bin_size=4 --num_cpus=4"
    exit 1
fi

# Check if parallel is available
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel not found. Please load the parallel module or install it."
    exit 1
fi

# Path to your CSV with coordinates
coord_csv="DTI_Processing_Scripts/LEGACY_DTI_coords_transformed_manually_adjusted.csv"

# Base directories
# Base directory for mixed IDs (letters and numbers)
mixed_base="/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI"

dwi_base="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/"
results_dir="DTI_Processing_Scripts/results"


mkdir -p $results_dir

# Create master results CSV # SUPPLIED AT INPUT WHEN CALLING FUNCTION
#num_bins=10
#bin_size=2
#num_bins=5
#bin_size=4

master_csv="$results_dir/all_metrics_${num_bins}x${bin_size}vox.csv"
echo "num bins: $num_bins"

if [ $num_bins -eq 10 ]; then
    echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_anterior_ring_6,FA_anterior_ring_7,FA_anterior_ring_8,FA_anterior_ring_9,FA_anterior_ring_10,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_posterior_ring_6,FA_posterior_ring_7,FA_posterior_ring_8,FA_posterior_ring_9,FA_posterior_ring_10,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_anterior_ring_6,FA_baseline_anterior_ring_7,FA_baseline_anterior_ring_8,FA_baseline_anterior_ring_9,FA_baseline_anterior_ring_10,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,FA_baseline_posterior_ring_6,FA_baseline_posterior_ring_7,FA_baseline_posterior_ring_8,FA_baseline_posterior_ring_9,FA_baseline_posterior_ring_10,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_anterior_ring_6,MD_anterior_ring_7,MD_anterior_ring_8,MD_anterior_ring_9,MD_anterior_ring_10,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_posterior_ring_6,MD_posterior_ring_7,MD_posterior_ring_8,MD_posterior_ring_9,MD_posterior_ring_10,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_anterior_ring_6,MD_baseline_anterior_ring_7,MD_baseline_anterior_ring_8,MD_baseline_anterior_ring_9,MD_baseline_anterior_ring_10,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5,MD_baseline_posterior_ring_6,MD_baseline_posterior_ring_7,MD_baseline_posterior_ring_8,MD_baseline_posterior_ring_9,MD_baseline_posterior_ring_10" > $master_csv
elif [ $num_bins -eq 5 ]; then
    echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5" > $master_csv
else
    echo "Invalid number of bins specified. Please set num_bins to either 5 or 10."
    exit 1
fi


#echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5" > $master_csv
# Process each non-excluded patient
# Export variables for parallel
export dwi_base mixed_base bin_size num_bins results_dir master_csv PARALLEL_UNBUFFERED=1
parallel --citation 2>/dev/null || true
# Add this before the parallel call


# Then modify your parallel call to force unbuffered output
#grep -v "^1," $coord_csv | parallel --will-cite -d "\r\n" --env dwi_base,mixed_base,bin_size,num_bins,results_dir,master_csv -j $num_cpus --colsep ',' '


grep -v "^1," $coord_csv | parallel --will-cite -d "\r\n" \
    --env dwi_base,mixed_base,bin_size,num_bins,results_dir,master_csv -j $num_cpus --colsep ',' '


    excluded={1}
    patient_id={2}
    timepoint={3}

    # Force unbuffered output for all commands in this script
    exec > >(tee -a "DTI_Processing_Scripts/results/parallel_job_${patient_id}_${timepoint}.log")
    exec 2>&1



    if [ "$excluded" == "0" ]; then
        
        
        # Determine path structure based on patient_id format
        if [[ "$patient_id" =~ ^[0-9]+$ ]]; then
            # if patient id is just numbers
            echo "Processing patient $patient_id at timepoint $timepoint"
            tp_dir=$dwi_base/$patient_id/$timepoint
            # echo "tp_dir: $tp_dir"
            # Find the DTIspace directory
            dti_dir=$(find "$tp_dir" -type d -name "DTIspace" | head -n 1)
            #echo "DTI directory: $dti_dir"
            
            # if there is more than one dtispace directory (dti_dir), echo the first one as well as patient id and timepoint
            if [ $(echo "$dti_dir" | wc -l) -gt 1 ]; then
                echo "WARNING: More than one DTIspace directory found for patient $patient_id at timepoint $timepoint. Using the first one."
            fi
            # echo "DTI directory: $dti_dir"
            # break out of the loop and restart for new patient id and timepoint
            

            mask_path="$dti_dir/masks/ANTS_T1_brain_mask.nii.gz"
            fa_path="$dti_dir/dti/dtifitWLS_FA.nii.gz"
            md_path="$dti_dir/dti/dtifitWLS_MD.nii.gz"

            # echo "Paths:"
            # echo "ANTS mask path: $mask_path"
            # echo "FA path: $fa_path"
            # echo "MD path: $md_path"
            
        else
            echo "Processing patient $patient_id at timepoint $timepoint"
            #echo "mixed base: $mixed_base"
            #echo "patient id: $patient_id"
            #echo "timepoint: $timepoint"
            
            
            # Patient ID contains letters and numbers
            # tp_dir=$(find "$mixed_base/$patient_id" -type d -name "*Hour-${timepoint}_*" -path
            patient_dir="$mixed_base/$patient_id"
            #echo "patient_dir: $patient_dir"
            
            #tp_dir=$(find "$mixed_base/$patient_id" -type d -name "Sub-*" -exec find {} -type d -name "Hour-${timepoint}_*" \; | head -n 1)
            #tp_dir=$(find "$patient_dir" -type d -name "Sub-*" -exec find {} -type d -name "Hour-${timepoint}_*" \; 2>/dev/null | head -n 1)
            
            tp_dir=$(find "$patient_dir" -type d -path "*Hour-${timepoint}*" -o -path "*Hour_${timepoint}*" 2>/dev/null | head -n 1)
            #echo "tp_dir: $tp_dir"
            tp_base=$(basename "$tp_dir")
            # echo "tp_base: $tp_base"
            dti_dir="${tp_dir}/dwi/proc_set1_nobzero/nipype/DATASINK/DTIspace"
            #  "/dwi/proc_set1_nobzero/nipype/DATASINK/DTIspace" | head -n 1)
            #echo "dti_dir: $dti_dir"
            
            

            # if mre than one dti_dir, echo the first one as well as patient id and timepoint
            if [ $(echo "$dti_dir" | wc -l) -gt 1 ]; then
                echo "WARNING: More than one DTIspace directory found for patient $patient_id at timepoint $timepoint. Using the first one."
            fi
            
            if [ -z "$tp_dir" ]; then
                echo "ERROR: Could not find directory containing timepoint $timepoint for patient $patient_id"
                exit 1
            fi
            mask_path="$dti_dir/masks/ANTS_T1_brain_mask.nii.gz"
            fa_path="$dti_dir/dti/dtifitWLS_FA.nii.gz"
            md_path="$dti_dir/dti/dtifitWLS_MD.nii.gz"
        fi
        # echo "Paths:"
        # echo "ANTS mask path: $mask_path"
        # echo "FA path: $fa_path"
        # echo "MD path: $md_path"
        
        
        
        # Check if files exist
        if [ -f "$mask_path" ] && [ -f "$fa_path" ] && [ -f "$md_path" ]; then
            echo "All required files found for patient $patient_id at timepoint $timepoint"   
            # Step 1: Create spherical ROIs
            ./DTI_Processing_Scripts/roi_create.sh "$patient_id" "$timepoint" "$tp_base" "$mask_path" "$fa_path" "$md_path" "$bin_size" "$num_bins"
            
            # Step 2: Extract metrics
            ./DTI_Processing_Scripts/roi_extract.sh "$patient_id" "$timepoint" "$tp_base" "$bin_size" "$num_bins" "$fa_path" "$md_path" "$master_csv"
            
            echo "back in main script"
            # Append to master CSV with file locking
            exit 0
            #tail -n 1 "DTI_Processing_Scripts/results/${patient_id}_${timepoint}_metrics_${num_bins}x${bin_size}vox.csv" >> $master_csv
            echo "after tail"
            #cat "DTI_Processing_Scripts/results/${patient_id}_${timepoint}_metrics_${num_bins}x${bin_size}vox.csv" | tail -n 1 >> $master_csv
            # {
            #     flock -w 60 -x 200
            #     cat "DTI_Processing_Scripts/results/${patient_id}_${timepoint}_metrics_${num_bins}x${bin_size}vox.csv" | tail -n 1 >> $master_csv
            # } 200>$master_csv.lock
            
        else
            echo "Missing files for patient $patient_id at timepoint $timepoint"
        fi
    fi
'  # Close parallel with a single quote here instead of 'done'

echo "All processing complete. Results in $master_csv"