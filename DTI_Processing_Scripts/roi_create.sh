#!/bin/bash
module load fsl
# Input parameters
patient_id=$1
timepoint=$2
tp_base=$3
mask_path=$4    # Path to the mask file
fa_path=$5      # Path to the FA image
md_path=$6      # Path to the MD image
bin_size=$7 # Size of the bin for the rings
num_bins=$8 # Number of bins for the rings
csv_path=${9:-"DTI_Processing_Scripts/LEGACY_DTI_coords_transformed_manually_adjusted.csv"}  # Optional CSV path, defaults to coordinates.csv
filtered_fa_values=${10:-"false"}  # Optional filter flag, defaults to false

# Check if all required parameters are provided
if [ -z "$patient_id" ] || [ -z "$timepoint" ] || [ -z "$mask_path" ] || [ -z "$fa_path" ] || [ -z "$md_path" ]; then
    echo "Usage: $0 <patient_id> <timepoint> <mask_path> <fa_path> <md_path> [csv_path]"
    echo "Example: $0 SUBJ001 baseline /path/to/mask.nii.gz /path/to/fa.nii.gz /path/to/md.nii.gz"
    exit 1
fi

# Check if files exist
for file in "$mask_path" "$fa_path" "$md_path" "$csv_path"; do
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file"
        exit 1
    fi
done

echo "hello"


# Get coordinates from CSV using the patient_id and timepoint
ant_x=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $12}')
ant_y=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $13}')
ant_z=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $14}')
post_x=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $15}')
post_y=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $16}')
post_z=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $17}')
baseline_ant_x=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $18}')
baseline_ant_y=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $19}')
baseline_ant_z=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $20}')
baseline_post_x=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $21}')
baseline_post_y=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $22}')
baseline_post_z=$(grep "$patient_id,$timepoint" $csv_path | awk -F, '{print $23}')

# Verify that coordinates were found
if [ -z "$ant_x" ] || [ -z "$ant_y" ] || [ -z "$ant_z" ] || [ -z "$post_x" ] || [ -z "$post_y" ] || [ -z "$post_z" ] || \
   [ -z "$baseline_ant_x" ] || [ -z "$baseline_ant_y" ] || [ -z "$baseline_ant_z" ] || \
   [ -z "$baseline_post_x" ] || [ -z "$baseline_post_y" ] || [ -z "$baseline_post_z" ]; then
    echo "Error: Could not find coordinates for patient $patient_id at timepoint $timepoint in $csv_path"
    exit 1
fi


# Create spherical ROIs for each point
if [[ "$patient_id" =~ ^[0-9]+$ ]]; then
    # Patient ID contains only numbers
    output_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox"
    base_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}"
    if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
        output_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox_NEW"
    fi


else
    output_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox"
    base_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi"
    if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
        output_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox_NEW"
    fi
fi



# if filtered_fa_values is true, add "_filtered" to the output directory
if [ "$filtered_fa_values" == "true" ]; then
    output_dir="${output_dir}_filtered"
fi

# if output_dir contains "NEW" and already exists, delete it
if [[ "$output_dir" == *"NEW"* ]] && [ -d "$output_dir" ]; then
    echo "Deleting existing directory: $output_dir"
    rm -rf "$output_dir"
fi

mkdir -p $output_dir

# if there is a directory with the same bin_size but lower num_bins, copy the files from that directory to the new dir (output_dir)
# search for search_dir based on output dir "/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${find_what_this_is}x${bin_size}vox"
# Search for directories with same bin_size but lower num_bins
echo "Looking for existing ROI directories with same bin size ($bin_size) but lower number of bins..."
# Find all matching directories
for dir in "$base_dir"/roi_files_*x${bin_size}vox; do
    if [ -d "$dir" ]; then
        # Extract the number of bins from the directory name
        dir_num_bins=$(echo "$dir" | grep -o 'roi_files_[0-9]*x' | sed 's/roi_files_//g' | sed 's/x//g')
        
        # If the directory has a lower number of bins, copy its files
        if [ "$dir_num_bins" -lt "$num_bins" ]; then
            echo "Found directory with $dir_num_bins bins for bin size $bin_size: $dir"
            echo "Copying these files recursively to $output_dir"
             # Use find to copy files recursively while preserving structure
            find "$dir" -type f -print0 | while IFS= read -r -d '' file; do
                # Get relative path
                rel_path="${file#$dir/}"
                # Create target directory if needed
                target_dir="$output_dir/$(dirname "$rel_path")"
                mkdir -p "$target_dir"
                # Copy if target doesn't exist
                if [ ! -f "$output_dir/$rel_path" ]; then
                    cp "$file" "$output_dir/$rel_path"
                fi
            done
        fi
    fi
done



# Create point ROIs
echo "Creating point ROIs..."




# if $output_dir/ant_point does not exist, perform fslmaths
if [ ! -f "$output_dir/ant_point.nii.gz" ]; then
    #echo "Creating point ROI for anterior point..."
    fslmaths $mask_path -mul 0 -add 1 -roi $ant_x 1 $ant_y 1 $ant_z 1 0 1 $output_dir/ant_point -odt float
fi
#fslmaths $mask_path -mul 0 -add 1 -roi $ant_x 1 $ant_y 1 $ant_z 1 0 1 $output_dir/ant_point -odt float
# if $output_dir/post_point does not exist, perform fslmaths
if [ ! -f "$output_dir/post_point.nii.gz" ]; then
    #echo "Creating point ROI for posterior point..."
    fslmaths $mask_path -mul 0 -add 1 -roi $post_x 1 $post_y 1 $post_z 1 0 1 $output_dir/post_point -odt float
fi
#fslmaths $mask_path -mul 0 -add 1 -roi $post_x 1 $post_y 1 $post_z 1 0 1 $output_dir/post_point -odt float

# if $output_dir/baseline_ant_point does not exist, perform fslmaths
if [ ! -f "$output_dir/baseline_ant_point.nii.gz" ]; then
    #echo "Creating point ROI for baseline anterior point..."
    fslmaths $mask_path -mul 0 -add 1 -roi $baseline_ant_x 1 $baseline_ant_y 1 $baseline_ant_z 1 0 1 $output_dir/baseline_ant_point -odt float
fi
#fslmaths $mask_path -mul 0 -add 1 -roi $baseline_ant_x 1 $baseline_ant_y 1 $baseline_ant_z 1 0 1 $output_dir/baseline_ant_point -odt float

if [ ! -f "$output_dir/baseline_post_point.nii.gz" ]; then
    #echo "Creating point ROI for baseline posterior point..."
    fslmaths $mask_path -mul 0 -add 1 -roi $baseline_post_x 1 $baseline_post_y 1 $baseline_post_z 1 0 1 $output_dir/baseline_post_point -odt float
fi
#fslmaths $mask_path -mul 0 -add 1 -roi $baseline_post_x 1 $baseline_post_y 1 $baseline_post_z 1 0 1 $output_dir/baseline_post_point -odt float


# Debug information
echo "Ant coordinates: $ant_x $ant_y $ant_z"
echo "Post coordinates: $post_x $post_y $post_z"
echo "Baseline ant coordinates: $baseline_ant_x $baseline_ant_y $baseline_ant_z"
echo "Baseline post coordinates: $baseline_post_x $baseline_post_y $baseline_post_z"



echo "Creating ROI rings within mask..."

# Function to create rings for a specific point and metric
create_metric_rings() {
    local point_name=$1
    local metric_path=$2
    local metric_name=$3
    
    # Create directories for metric-specific output
    mkdir -p $output_dir/${metric_name}
    echo "Output directory for $patient_id $timepoint $metric_name: $output_dir/${metric_name}"
    echo "bin size: $bin_size"

    # Variable to track the last existing whole sphere
    local last_whole_sphere=""
    
    
    for ((i=1; i<=$num_bins; i++)); do
        radius=$((i*$bin_size))
        prev_radius=$(($radius-$bin_size))
        echo "Processing ring $i: prev_radius: $prev_radius; current radius: $radius"

        # Check if the ring and metric files already exist
        ring_file="$output_dir/${metric_name}/${point_name}_ring${i}.nii.gz"
        metric_file="$output_dir/${metric_name}/${point_name}_ring${i}_${metric_name}.nii.gz"
        whole_file="$output_dir/${metric_name}/${point_name}_whole_${i}.nii.gz"

        # NEW: If all files exist, we can skip this iteration
        if [ -f "$ring_file" ] && [ -f "$metric_file" ] && [ -f "$whole_file" ]; then
            echo "Files for ring $i already exist, skipping..."
            last_whole_sphere="$whole_file"
            continue
        fi
        
        if [ $prev_radius -eq 0 ]; then
            # First ring - create spherical dilation then mask with the brain mask
            fslmaths $output_dir/${point_name}_point -kernel sphere $radius -dilM -mul $mask_path $output_dir/${metric_name}/${point_name}_ring${i}
            # Extract metric values within the ring
            fslmaths $output_dir/${metric_name}/${point_name}_ring${i} -mul $metric_path $output_dir/${metric_name}/${point_name}_ring${i}_${metric_name}
            # Save a copy of whole sphere for subsequent rings
            fslmaths $output_dir/${metric_name}/${point_name}_ring${i} $output_dir/${metric_name}/${point_name}_whole_${i}
            # Track the whole sphere file
            last_whole_sphere="$whole_file"
        else

            # Check if we have a reference to the last whole sphere
            if [ -z "$last_whole_sphere" ]; then
                # If not, look for the previous whole sphere file
                prev_whole="$output_dir/${metric_name}/${point_name}_whole_$((i-1)).nii.gz"
                if [ ! -f "$prev_whole" ]; then
                    echo "ERROR: Cannot find previous whole sphere file: $prev_whole"
                    echo "Cannot continue processing for this point. Exiting function."
                    return 1
                fi
                last_whole_sphere="$prev_whole"
            fi

            # Subsequent rings (donuts)
            fslmaths $output_dir/${point_name}_point -kernel sphere $radius -dilM -mul $mask_path $output_dir/${metric_name}/${point_name}_whole_temp
            # Subtract the previous sphere to get a ring
            fslmaths $output_dir/${metric_name}/${point_name}_whole_temp -sub $output_dir/${metric_name}/${point_name}_whole_$((i-1)) $output_dir/${metric_name}/${point_name}_ring${i}
            # Extract metric values within the ring
            fslmaths $output_dir/${metric_name}/${point_name}_ring${i} -mul $metric_path $output_dir/${metric_name}/${point_name}_ring${i}_${metric_name}
            # Save a copy of this whole sphere for the next iteration
            fslmaths $output_dir/${metric_name}/${point_name}_whole_temp $output_dir/${metric_name}/${point_name}_whole_${i}
            # Clean up temporary files
            rm -f $output_dir/${metric_name}/${point_name}_whole_temp.*
        fi
    done
}

# Process for all points with FA
for point in "ant" "post" "baseline_ant" "baseline_post"; do
    # if files dont exist, do function
    if [ ! -f "$output_dir/FA/${point}_whole_${num_bins}.nii.gz" ]; then
        #echo "Creating rings for $point with FA..."
        create_metric_rings $point $fa_path "FA"
    fi
done

# Process for all points with MD
for point in "ant" "post" "baseline_ant" "baseline_post"; do
    # if files dont exist, do function
    if [ ! -f "$output_dir/MD/${point}_whole_${num_bins}.nii.gz" ]; then
        #echo "Creating rings for $point with MD..."
        create_metric_rings $point $md_path "MD"
    fi
done


# Print confirmation message
echo "ROIs created successfully for patient $patient_id at timepoint $timepoint"
echo "Output stored in $output_dir"