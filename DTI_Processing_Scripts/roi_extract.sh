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
master_csv=$8 # Path to the master CSV file
filter_fa_values=${9:-"false"} # New flag to enable/disable FA filtering, defaults to "false"
get_all_values=${10:-"false"} # New flag to enable/disable getting all values, defaults to "false"
overwrite_all=${11:-"false"} # New flag to enable/disable overwriting all values, defaults to "false"
md_extraction_overwrite=${12:-"false"} # New flag to enable/disable MD extraction overwrite, defaults to "false"

# Set ROI dir
# Create spherical ROIs for each point
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

if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
    #output_csv="DTI_Processing_Scripts/results/${patient_id}_${timepoint}_metrics_${num_bins}x${bin_size}vox_NEW.csv"
    output_csv="${output_csv%.csv}_NEW.csv"
elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
    #output_csv="DTI_Processing_Scripts/results/${patient_id}_${timepoint}_metrics_${num_bins}x${bin_size}vox_NEW.csv"
    output_csv="${output_csv%.csv}_NEW.csv"
fi

# Modify output_csv name if filtering is enabled
if [ "$filter_fa_values" = "true" ]; then
    # Append filtered to output_csv filename
    output_csv="${output_csv%.csv}_filtered.csv"

    # echo "Using filtered output files:"
    # echo "  - Individual CSV: $output_csv"
    # echo "  - Master CSV: $master_csv"
fi

if [ "$get_all_values" = "true" ]; then
    # Append all_values to output_csv filename
    output_csv="${output_csv%.csv}_all_values.csv"
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
#echo "patient_id,timepoint,FA_anterior_ring_1,FA_anterior_ring_2,FA_anterior_ring_3,FA_anterior_ring_4,FA_anterior_ring_5,FA_posterior_ring_1,FA_posterior_ring_2,FA_posterior_ring_3,FA_posterior_ring_4,FA_posterior_ring_5,FA_baseline_anterior_ring_1,FA_baseline_anterior_ring_2,FA_baseline_anterior_ring_3,FA_baseline_anterior_ring_4,FA_baseline_anterior_ring_5,FA_baseline_posterior_ring_1,FA_baseline_posterior_ring_2,FA_baseline_posterior_ring_3,FA_baseline_posterior_ring_4,FA_baseline_posterior_ring_5,MD_anterior_ring_1,MD_anterior_ring_2,MD_anterior_ring_3,MD_anterior_ring_4,MD_anterior_ring_5,MD_posterior_ring_1,MD_posterior_ring_2,MD_posterior_ring_3,MD_posterior_ring_4,MD_posterior_ring_5,MD_baseline_anterior_ring_1,MD_baseline_anterior_ring_2,MD_baseline_anterior_ring_3,MD_baseline_anterior_ring_4,MD_baseline_anterior_ring_5,MD_baseline_posterior_ring_1,MD_baseline_posterior_ring_2,MD_baseline_posterior_ring_3,MD_baseline_posterior_ring_4,MD_baseline_posterior_ring_5" > $output_csv

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


if [ "$md_extraction_overwrite" = "true" ]; then
    # Remove the temp.csv file if it exists
    echo "MD extraction overwrite enabled. Processing only MD values..."


    # first, check if output csv file exists
    if [ -f "$output_csv" ]; then

        # Force convert line endings first
        dos2unix "$output_csv" 2>/dev/null

        # Use several methods and take the first non-empty result
        second_line=""

        # Method 1: Use tail to skip header
        second_line=$(tail -n +2 "$output_csv" | head -n 1)





        #read the existing data line for this patient/timepoint

        # Use '^' to anchor to start of line, and escape commas
        # search_pattern="^${patient_id},${timepoint},*"
        # echo "Looking for pattern: '$search_pattern' in $output_csv"

        #existing_line=$(grep "$search_pattern" "$output_csv")
        # existing_line=$(sed -n '2p' "$output_csv")
        # Save entire file to a temporary file to ensure we have access

        echo "File exists check:"
        [ -f "$output_csv" ] && echo "File exists" || echo "File does not exist"

        echo "File size check:"
        wc -c "$output_csv"

        echo "Number of lines check:"
        wc -l "$output_csv"

        # Let's see what's actually in the file - full contents
        echo "Full file contents:"
        cat "$output_csv"

        # Display the extracted line
        echo "Second line from $output_csv: $second_line"

        echo "*******************************************"




        # First, dump the entire file content and pipe it to grep to get non-header lines
        echo "Step 1: Filtering out header line..."
        grep -v "^patient_id" "$output_csv" > /tmp/data_only.csv
        echo "Filter complete."

        # Check if we got anything
        echo "Step 2: Checking line count in filtered file..."
        line_count=$(wc -l < /tmp/data_only.csv)
        echo "Found $line_count lines in filtered file."

        # Now get the first line from this filtered file (which should be your data line)
        echo "Step 3: Extracting first line of filtered data..."
        data_line=$(head -n 1 /tmp/data_only.csv)
        echo "Data line: $data_line"

        # Store this in a variable for later use
        echo "Step 4: Storing in variable 'second_line'..."
        second_line=$data_line
        echo "Variable set. First 50 characters: ${second_line:0:50}..."



        cp "$output_csv" /tmp/temp_csv_file.csv

        line_2=$(awk 'NR==3 {print; exit}' "$output_csv")
        echo "Actual line 2 data: $line_2"

        dos2unix "$output_csv" 
        line_2=$(awk 'NR==2 {print; exit}' "$output_csv")
        echo "After dos2unix, line 2: $line_2"
        exit

        
        
        # Try several methods to get line 2
        existing_line_sed=$(sed -n '2p' /tmp/temp_csv_file.csv)
        echo "Existing line sed: $existing_line_sed"
        existing_line_awk=$(awk 'NR==2 {print; exit}' /tmp/temp_csv_file.csv)
        existing_line_head=$(head -n 2 /tmp/temp_csv_file.csv | tail -n 1)
        existing_line_tail=$(tail -n +2 /tmp/temp_csv_file.csv | head -n 1)
        
        echo "Line 2 via sed: ${existing_line_sed:0:20}..."
        echo "Line 2 via awk: ${existing_line_awk:0:20}..."
        echo "Line 2 via head/tail: ${existing_line_head}..."
        echo "Line 2 via tail: ${existing_line_tail}..."
        exit 0

        if [ -n "$existing_line" ]; then
            echo "Found existing data line for patient $patient_id at timepoint $timepoint"

            # Calculate how many FA columns we need based on num_bins
            if [ $num_bins -eq 10 ]; then
                # For 10 bins: 2 columns (patient_id, timepoint) + 40 FA columns
                fa_columns=42
            elif [ $num_bins -eq 5 ]; then
                # For 5 bins: 2 columns (patient_id, timepoint) + 20 FA columns
                fa_columns=22
            fi
            # Extract just FA part (patient_id, timepoint, FA values)
            fa_part=$(echo "$existing_line" | cut -d',' -f1-$fa_columns)
            # Initialise data line with existing FA values
            data_line="${fa_part}"

            #Process only MD values
            parameter="MD"
            echo "Extracting $parameter values..."

            # Iterate through all labels
            for label in ant post baseline_ant baseline_post; do
                # Process all bins for this parameter/label combination
                for ((i=1; i<=$num_bins; i++)); do
                # Handle special case for MD with filtering
                    if [ $get_all_values = 'false' ]; then
                        if [ "$filter_fa_values" = "true" ]; then
                        mean_value=$(extract_filtered_md_value "$roi_dir/FA/${label}_ring${i}_FA.nii.gz" "$roi_dir/$parameter/${label}_ring${i}_${parameter}.nii.gz" "$roi_dir/$parameter/${label}_ring${i}.nii.gz")
                        else
                        mean_value=$(fslmeants -i "$roi_dir/$parameter/${label}_ring${i}_${parameter}.nii.gz" -m "$roi_dir/$parameter/${label}_ring${i}.nii.gz")
                        fi
                        data_line="${data_line},${mean_value}"
                    else
                        # Get all values using showall option
                        fslmeants -i "$roi_dir/$parameter/${label}_ring${i}_${parameter}.nii.gz" \
                                -m "$roi_dir/$parameter/${label}_ring${i}.nii.gz" \
                                -o DTI_Processing_Scripts/results/temp.csv --showall
                        
                        # Extract values, removing empty lines
                        md_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
                        
                        # Format as array by wrapping in quotes and brackets
                        md_array="\"[${md_values}]\""
                        
                        data_line="${data_line},${md_array}"
                    fi
                done
            done

            # Remove existing line from output CSV
            grep -v "^$patient_id,$timepoint" "$output_csv" > "${output_csv}.tmp"
            mv "${output_csv}.tmp" "$output_csv"

            # Also update master CSV
            grep -v "^$patient_id,$timepoint" "$master_csv" > "${master_csv}.tmp"
            mv "${master_csv}.tmp" "$master_csv"

            echo "Updated MD values for patient $patient_id at timepoint $timepoint"
        else
            echo "Warning: No existing data line found for patient $patient_id at timepoint $timepoint"
            echo "Falling back to normal processing for both FA and MD values"
            # Fall back to normal processing (next code section)
            md_extraction_overwrite="false"
        fi
    else
        echo "Warning: Output CSV does not exist for MD overwrite"
        echo "Falling back to normal processing for both FA and MD values"
        # Fall back to normal processing (next code section)
        md_extraction_overwrite="false"
    fi
fi





# Only do normal processing if not doing MD overwrite or fallback required
if [ "$md_extraction_overwrite" = "false" ]; then
    # Iterate through parameters (FA, MD)
    for parameter in FA MD; do
    echo "Extracting $parameter values..."
        
        # Iterate through all labels
        for label in ant post baseline_ant baseline_post; do
            # Process all bins for this parameter/label combination
            for ((i=1; i<=$num_bins; i++)); do
                # Handle special case for FA with filtering
                if [ $get_all_values = 'false' ]; then
                    if [ "$parameter" = "FA" ] && [ "$filter_fa_values" = "true" ]; then
                        mean_value=$(extract_filtered_fa_value "$roi_dir/$parameter/${label}_ring${i}_${parameter}.nii.gz" "$roi_dir/$parameter/${label}_ring${i}.nii.gz")
                    elif [ "$parameter" = "MD" ] && [ "$filter_fa_values" = "true" ]; then
                        mean_value=$(extract_filtered_md_value "$roi_dir/FA/${label}_ring${i}_FA.nii.gz" "$roi_dir/$parameter/${label}_ring${i}_${parameter}.nii.gz" "$roi_dir/$parameter/${label}_ring${i}.nii.gz")
                    else 
                        mean_value=$(fslmeants -i "$roi_dir/$parameter/${label}_ring${i}_${parameter}.nii.gz" -m "$roi_dir/$parameter/${label}_ring${i}.nii.gz")
                    fi
                    data_line="${data_line},${mean_value}"
                else
                    # Get all values using showall option
                    fslmeants -i "$roi_dir/$parameter/${label}_ring${i}_${parameter}.nii.gz" \
                            -m "$roi_dir/$parameter/${label}_ring${i}.nii.gz" \
                            -o DTI_Processing_Scripts/results/temp.csv --showall
                
                    # Extract values, removing empty lines
                    fa_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
                    
                    # Format as array by wrapping in quotes and brackets
                    fa_array="\"[${fa_values}]\""
                    
                    data_line="${data_line},${fa_array}"
                fi
            done
        done
    done
fi


# echo "Extracting FA values..."

# # Extract FA values for anterior rings
# for ((i=1; i<=$num_bins; i++)); do
#   # fslmeants -i "$roi_dir/FA/ant_ring${i}_FA.nii.gz" -m "$roi_dir/FA/ant_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
#   # fa_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
#   # # Format as array by wrapping in quotes and brackets
#   # fa_array="\"[${fa_values}]\""
#   # data_line="${data_line},${fa_array}"

#   # Just gets the mean value without --showall
#   if [ "$filter_fa_values" = "true" ]; then
#       fa_mean=$(extract_filtered_fa_value "$roi_dir/FA/ant_ring${i}_FA.nii.gz" "$roi_dir/FA/ant_ring${i}.nii.gz")
#   else
#       fa_mean=$(fslmeants -i "$roi_dir/FA/ant_ring${i}_FA.nii.gz" -m "$roi_dir/FA/ant_ring${i}.nii.gz")
#   fi
#   data_line="${data_line},${fa_mean}"
# done

# # Extract FA values for posterior rings
# for ((i=1; i<=$num_bins; i++)); do
#   # fslmeants -i "$roi_dir/FA/post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/post_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
#   # fa_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
#   # fa_array="\"[${fa_values}]\""
#   # data_line="${data_line},${fa_array}"

#   # Just gets the mean value without --showall
#   if [ "$filter_fa_values" = "true" ]; then
#       fa_mean=$(extract_filtered_fa_value "$roi_dir/FA/post_ring${i}_FA.nii.gz" "$roi_dir/FA/post_ring${i}.nii.gz")
#   else
#       fa_mean=$(fslmeants -i "$roi_dir/FA/post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/post_ring${i}.nii.gz")
#   fi
#   #fa_mean=$(fslmeants -i "$roi_dir/FA/post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/post_ring${i}.nii.gz")
#   data_line="${data_line},${fa_mean}"
# done

# # Extract FA values for baseline anterior rings
# for ((i=1; i<=$num_bins; i++)); do
#   # fslmeants -i "$roi_dir/FA/baseline_ant_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_ant_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
#   # fa_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
#   # fa_array="\"[${fa_values}]\""
#   # data_line="${data_line},${fa_array}"

#   # Just gets the mean value without --showall

#   if [ "$filter_fa_values" = "true" ]; then
#       fa_mean=$(extract_filtered_fa_value "$roi_dir/FA/baseline_ant_ring${i}_FA.nii.gz" "$roi_dir/FA/baseline_ant_ring${i}.nii.gz")
#   else
#       fa_mean=$(fslmeants -i "$roi_dir/FA/baseline_ant_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_ant_ring${i}.nii.gz")
#   fi
#   # fa_mean=$(fslmeants -i "$roi_dir/FA/baseline_ant_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_ant_ring${i}.nii.gz")
#   data_line="${data_line},${fa_mean}"
# done
# # Extract FA values for baseline posterior rings
# for ((i=1; i<=$num_bins; i++)); do
#   # fslmeants -i "$roi_dir/FA/baseline_post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_post_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
#   # fa_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
#   # fa_array="\"[${fa_values}]\""
#   # data_line="${data_line},${fa_array}"

#   # Just gets the mean value without --showall
#   if [ "$filter_fa_values" = "true" ]; then
#       fa_mean=$(extract_filtered_fa_value "$roi_dir/FA/baseline_post_ring${i}_FA.nii.gz" "$roi_dir/FA/baseline_post_ring${i}.nii.gz")
#   else
#       fa_mean=$(fslmeants -i "$roi_dir/FA/baseline_post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_post_ring${i}.nii.gz")
#   fi
#   # fa_mean=$(fslmeants -i "$roi_dir/FA/baseline_post_ring${i}_FA.nii.gz" -m "$roi_dir/FA/baseline_post_ring${i}.nii.gz")
#   data_line="${data_line},${fa_mean}"
# done

# # Extract MD values
# echo "Extracting MD values..."
# # Extract MD values for anterior rings
# for ((i=1; i<=$num_bins; i++)); do
#   # fslmeants -i "$roi_dir/MD/ant_ring${i}_MD.nii.gz" -m "$roi_dir/MD/ant_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
#   # md_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
#   # md_array="\"[${md_values}]\""
#   # data_line="${data_line},${md_array}"

#   # Just gets the mean value without --showall
#   md_mean=$(fslmeants -i "$roi_dir/MD/ant_ring${i}_MD.nii.gz" -m "$roi_dir/MD/ant_ring${i}.nii.gz")
#   data_line="${data_line},${md_mean}"
# done

# # Extract MD values for posterior rings
# for ((i=1; i<=$num_bins; i++)); do
#   # fslmeants -i "$roi_dir/MD/post_ring${i}_MD.nii.gz" -m "$roi_dir/MD/post_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
#   # md_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
#   # md_array="\"[${md_values}]\""
#   # data_line="${data_line},${md_array}"

#   # Just gets the mean value without --showall
#   md_mean=$(fslmeants -i "$roi_dir/MD/post_ring${i}_MD.nii.gz" -m "$roi_dir/MD/post_ring${i}.nii.gz")
#   data_line="${data_line},${md_mean}"
# done

# # Extract MD values for baseline anterior rings
# for ((i=1; i<=$num_bins; i++)); do
#   # fslmeants -i "$roi_dir/MD/baseline_ant_ring${i}_MD.nii.gz" -m "$roi_dir/MD/baseline_ant_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
#   # md_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
#   # md_array="\"[${md_values}]\""
#   # data_line="${data_line},${md_array}"

#   # Just gets the mean value without --showall
#   md_mean=$(fslmeants -i "$roi_dir/MD/baseline_ant_ring${i}_MD.nii.gz" -m "$roi_dir/MD/baseline_ant_ring${i}.nii.gz")
#   data_line="${data_line},${md_mean}"
# done

# # Extract MD values for baseline posterior rings
# for ((i=1; i<=$num_bins; i++)); do
#   # fslmeants -i "$roi_dir/MD/baseline_post_ring${i}_MD.nii.gz" -m "$roi_dir/MD/baseline_post_ring${i}.nii.gz" -o DTI_Processing_Scripts/results/temp.csv --showall
#   # md_values=$(cat DTI_Processing_Scripts/results/temp.csv | grep -v "^$" | tail -1)
#   # md_array="\"[${md_values}]\""
#   # data_line="${data_line},${md_array}"

#   # Just gets the mean value without --showall
#   md_mean=$(fslmeants -i "$roi_dir/MD/baseline_post_ring${i}_MD.nii.gz" -m "$roi_dir/MD/baseline_post_ring${i}.nii.gz")
#   data_line="${data_line},${md_mean}"
# done

# Clean up temporary files
rm -f DTI_Processing_Scripts/results/temp.csv


# Write data to CSV
echo "Writing data to CSV..."
echo "$data_line" >> $output_csv

# Write to master CSV here
# # Write DIRECTLY to the master CSV with file locking
# echo "Writing data to master CSV..."
# (
#     # Use flock for safe concurrent access
#     flock -w 30 200
#     echo "$data_line" >> "$master_csv"
#     # Explicitly flush file system buffers
#     sync
# ) 200>"${master_csv}.lock"

# rm -f "${master_csv}.lock"

# Before appending
echo "About to append to master CSV: $master_csv"
echo "Data line content: $data_line"

# When appending
echo "$data_line" >> "$master_csv"

# After appending
echo "Verifying append operation..."
tail -n 1 "$master_csv"


# echo "$patient_id,$timepoint,$fa_ant_r1,$fa_ant_r2,$fa_ant_r3,$fa_ant_r4,$fa_ant_r5,$fa_post_r1,$fa_post_r2,$fa_post_r3,$fa_post_r4,$fa_post_r5,$fa_base_ant_r1,$fa_base_ant_r2,$fa_base_ant_r3,$fa_base_ant_r4,$fa_base_ant_r5,$fa_base_post_r1,$fa_base_post_r2,$fa_base_post_r3,$fa_base_post_r4,$fa_base_post_r5,$md_ant_r1,$md_ant_r2,$md_ant_r3,$md_ant_r4,$md_ant_r5,$md_post_r1,$md_post_r2,$md_post_r3,$md_post_r4,$md_post_r5,$md_base_ant_r1,$md_base_ant_r2,$md_base_ant_r3,$md_base_ant_r4,$md_base_ant_r5,$md_base_post_r1,$md_base_post_r2,$md_base_post_r3,$md_base_post_r4,$md_base_post_r5" >> $output_csv
echo "Extraction complete for patient $patient_id at timepoint $timepoint"


# End of script