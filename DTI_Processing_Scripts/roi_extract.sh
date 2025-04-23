#!/bin/bash
module load fsl
# Input parameters
patient_id=$1
timepoint=$2
tp_base=$3  # Base directory for the timepoint
bin_size=$4 # Size of the bin for the rings
num_bins=$5 # Number of bins for the rings
fa_map=$6  # Path to FA map (not used in this script)
md_map=$7  # Path to MD map (not used in this script)

# Set ROI dir
# Create spherical ROIs for each point
if [[ "$patient_id" =~ ^[0-9]+$ ]]; then
    # Patient ID contains only numbers
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox"
else
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox"
fi
echo "ROI directory: $roi_dir"

if [ ! -d "$roi_dir" ]; then
    # default to roi_files 
    if [[ "$patient_id" =~ ^[0-9]+$ ]]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files"
    else
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files"
    fi
    echo "ROI directory not found, defaulting to: $roi_dir"
fi

# Output CSV file
output_csv="DTI_Processing_Scripts/results/${patient_id}_${timepoint}_metrics_${num_bins}x${bin_size}vox.csv"
# if output_csv already exists, remove it
if [ -f "$output_csv" ]; then
    rm -f "$output_csv"
fi
# create new
mkdir -p $(dirname $output_csv)


echo "Output CSV: $output_csv"

# Initialize CSV header
if [ $num_bins -eq 10 ]; then
    echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_anterior_ring_6,FA_anterior_ring_7,FA_anterior_ring_8,FA_anterior_ring_9,FA_anterior_ring_10,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_posterior_ring_6,FA_posterior_ring_7,FA_posterior_ring_8,FA_posterior_ring_9,FA_posterior_ring_10,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_anterior_ring_6,FA_baseline_anterior_ring_7,FA_baseline_anterior_ring_8,FA_baseline_anterior_ring_9,FA_baseline_anterior_ring_10,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,FA_baseline_posterior_ring_6,FA_baseline_posterior_ring_7,FA_baseline_posterior_ring_8,FA_baseline_posterior_ring_9,FA_baseline_posterior_ring_10" > $output_csv
elif [ $num_bins -eq 5 ]; then
    # echo "patient_id,timepoint" > $output_csv
    echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5" > $output_csv
else
    echo "Invalid number of bins specified. Please set num_bins to either 5 or 10."
    exit 1
fi
#echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5" > $output_csv

# Initialize the data line with patient_id and timepoint
data_line="${patient_id},${timepoint}"


echo "Extracting FA values..."

# Extract FA values for anterior rings
for ((i=1; i<=$num_bins; i++)); do
  # fslmeants -i "$roi_dir/FA/ant_ring${i}_FA.nii.gz" -m "$roi_dir/FA/ant_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
  # fa_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
  # # Format as array by wrapping in quotes and brackets
  # fa_array="\"[${fa_values}]\""
  # data_line="${data_line},${fa_array}"

  # Just gets the mean value without --showall
  fa_mean=$(fslmeants -i "$roi_dir/FA/ant_ring${i}_FA.nii.gz" -m "$roi_dir/FA/ant_ring${i}.nii.gz")
  data_line="${data_line},${fa_mean}"
done

# Extract FA values for posterior rings
for ((i=1; i<=$num_bins; i++)); do
  # fslmeants -i "$roi_dir/FA/post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/post_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
  # fa_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
  # fa_array="\"[${fa_values}]\""
  # data_line="${data_line},${fa_array}"

  # Just gets the mean value without --showall
  fa_mean=$(fslmeants -i "$roi_dir/FA/post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/post_ring${i}.nii.gz")
  data_line="${data_line},${fa_mean}"
done

# Extract FA values for baseline anterior rings
for ((i=1; i<=$num_bins; i++)); do
  # fslmeants -i "$roi_dir/FA/baseline_ant_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_ant_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
  # fa_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
  # fa_array="\"[${fa_values}]\""
  # data_line="${data_line},${fa_array}"

  # Just gets the mean value without --showall
  fa_mean=$(fslmeants -i "$roi_dir/FA/baseline_ant_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_ant_ring${i}.nii.gz")
  data_line="${data_line},${fa_mean}"
done
# Extract FA values for baseline posterior rings
for ((i=1; i<=$num_bins; i++)); do
  # fslmeants -i "$roi_dir/FA/baseline_post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_post_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
  # fa_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
  # fa_array="\"[${fa_values}]\""
  # data_line="${data_line},${fa_array}"

  # Just gets the mean value without --showall
  fa_mean=$(fslmeants -i "$roi_dir/FA/baseline_post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_post_ring${i}.nii.gz")
  data_line="${data_line},${fa_mean}"
done

# Extract MD values
echo "Extracting MD values..."
# Extract MD values for anterior rings
for ((i=1; i<=$num_bins; i++)); do
  # fslmeants -i "$roi_dir/MD/ant_ring${i}_MD.nii.gz" -m "$roi_dir/MD/ant_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
  # md_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
  # md_array="\"[${md_values}]\""
  # data_line="${data_line},${md_array}"

  # Just gets the mean value without --showall
  md_mean=$(fslmeants -i "$roi_dir/MD/ant_ring${i}_MD.nii.gz" -m "$roi_dir/MD/ant_ring${i}.nii.gz")
  data_line="${data_line},${md_mean}"
done

# Extract MD values for posterior rings
for ((i=1; i<=$num_bins; i++)); do
  # fslmeants -i "$roi_dir/MD/post_ring${i}_MD.nii.gz" -m "$roi_dir/MD/post_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
  # md_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
  # md_array="\"[${md_values}]\""
  # data_line="${data_line},${md_array}"

  # Just gets the mean value without --showall
  md_mean=$(fslmeants -i "$roi_dir/MD/post_ring${i}_MD.nii.gz" -m "$roi_dir/MD/post_ring${i}.nii.gz")
  data_line="${data_line},${md_mean}"
done

# Extract MD values for baseline anterior rings
for ((i=1; i<=$num_bins; i++)); do
  # fslmeants -i "$roi_dir/MD/baseline_ant_ring${i}_MD.nii.gz" -m "$roi_dir/MD/baseline_ant_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
  # md_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
  # md_array="\"[${md_values}]\""
  # data_line="${data_line},${md_array}"

  # Just gets the mean value without --showall
  md_mean=$(fslmeants -i "$roi_dir/MD/baseline_ant_ring${i}_MD.nii.gz" -m "$roi_dir/MD/baseline_ant_ring${i}.nii.gz")
  data_line="${data_line},${md_mean}"
done

# Extract MD values for baseline posterior rings
for ((i=1; i<=$num_bins; i++)); do
  # fslmeants -i "$roi_dir/MD/baseline_post_ring${i}_MD.nii.gz" -m "$roi_dir/MD/baseline_post_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
  # md_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
  # md_array="\"[${md_values}]\""
  # data_line="${data_line},${md_array}"

  # Just gets the mean value without --showall
  md_mean=$(fslmeants -i "$roi_dir/MD/baseline_post_ring${i}_MD.nii.gz" -m "$roi_dir/MD/baseline_post_ring${i}.nii.gz")
  data_line="${data_line},${md_mean}"
done

# Clean up temporary files
rm -f DTI_Processing_Scripts/results/temp.csv


# Write data to CSV
echo "Writing data to CSV..."
echo "$data_line" >> $output_csv
# echo "$patient_id,$timepoint,$fa_ant_r1,$fa_ant_r2,$fa_ant_r3,$fa_ant_r4,$fa_ant_r5,$fa_post_r1,$fa_post_r2,$fa_post_r3,$fa_post_r4,$fa_post_r5,$fa_base_ant_r1,$fa_base_ant_r2,$fa_base_ant_r3,$fa_base_ant_r4,$fa_base_ant_r5,$fa_base_post_r1,$fa_base_post_r2,$fa_base_post_r3,$fa_base_post_r4,$fa_base_post_r5,$md_ant_r1,$md_ant_r2,$md_ant_r3,$md_ant_r4,$md_ant_r5,$md_post_r1,$md_post_r2,$md_post_r3,$md_post_r4,$md_post_r5,$md_base_ant_r1,$md_base_ant_r2,$md_base_ant_r3,$md_base_ant_r4,$md_base_ant_r5,$md_base_post_r1,$md_base_post_r2,$md_base_post_r3,$md_base_post_r4,$md_base_post_r5" >> $output_csv
echo "Extraction complete for patient $patient_id at timepoint $timepoint"


# End of script