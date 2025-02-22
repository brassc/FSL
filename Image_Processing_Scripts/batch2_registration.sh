#!/bin/bash

# Load FSL module
module load fsl

# Base path for GUPI directories
GUPI_BASE_PATH="/rds-d5/user/cmb247/hpc-work/Feb2025_working"

# Function to print usage
usage() {
    echo "Usage: $0 [-g GUPI] [-list list_file]"
    echo "  -g    : Single GUPI directory to process"
    echo "  -list : File containing list of GUPI directories"
    exit 1
}

# Parse command line arguments
while getopts "g:list:" opt; do
    case $opt in
        g)
            single_gupi=$OPTARG
            ;;
        list)
            list_file=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

# Function to extract hour number from filename
get_hour_number() {
    local filename=$1
    if [[ $filename =~ Hour-([0-9]{5}) ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}



# Function to process a single GUPI
process_gupi() {
    local gupi_name=$1
    local gupi_dir="${GUPI_BASE_PATH}/${gupi_name}"
    local bet_dir="${gupi_dir}/BET_Output"


    if [ ! -d "$gupi_dir" ]; then
        echo "Error: GUPI directory not found: ${gupi_dir}"
        return 1
    fi
    echo "GUPI dir: $gupi_dir"
    echo "BET dir: $bet_dir"
    
    
    # Check if BET_Output directory exists
    if [ ! -d "$bet_dir" ]; then
        echo "Error: BET_Output directory not found in ${gupi_dir}"
        return 1
    fi
    
    # Find all Hour-XXXXX files and extract the earliest one
    # First try to find modified files with Hour- pattern
    earliest_image=$(find "$bet_dir" -name "*Hour-[0-9]*modified*" | grep -v "mask" | sort | head -n 1)
    #earliest_image=$(find "$bet_dir" -name "*Hour-[0-9]*" | sort | head -n 1)
    # If no modified files found, fall back to non-modified files
    if [ -z "$earliest_image" ]; then
        echo "No modified Hour- files found, falling back to non-modified files..."
        earliest_image=$(find "$bet_dir" -name "*Hour-[0-9]*" | grep -v -e "modified" -e "mask" | sort | head -n 1)
    fi
    
    if [ -z "$earliest_image" ]; then
        echo "Error: No Hour-XXXXX files found in ${bet_dir}"
        return 1
    fi
    
    echo "Reference image: ${earliest_image}"
    
    # Get all unique hour numbers in order
    declare -a hour_numbers
    while IFS= read -r file; do
        hour_num=$(get_hour_number "$file")
        if [ ! -z "$hour_num" ]; then
            if [[ ! " ${hour_numbers[@]} " =~ " ${hour_num} " ]]; then
                hour_numbers+=("$hour_num")
            fi
        fi
    done < <(find "$bet_dir" -name "*Hour-[0-9]*" | grep -v "mask")
    
    # Sort hour numbers
    IFS=$'\n' hour_numbers=($(sort <<<"${hour_numbers[*]}"))
    unset IFS
    echo "Hour numbers found (in order): ${hour_numbers[*]}"
    
    # Create BET_Reg directory if it doesn't exist
    reg_dir="${gupi_dir}/BET_Reg"
    mkdir -p "$reg_dir"


    # Process images by hour, preferring modified versions
    for hour in "${hour_numbers[@]}"; do
        # Skip if this is the reference image's hour
        if [[ "$earliest_image" =~ Hour-${hour} ]]; then
            continue
        fi
        
        # Check for modified version first
        modified_image=$(find "$bet_dir" -name "*modified*Hour-${hour}*" | grep -v "mask")
        
        
        if [ ! -z "$modified_image" ]; then
            image="$modified_image"
        else
            # Use non-modified version if no modified exists
            image=$(find "$bet_dir" -name "*Hour-${hour}*" | grep -v -e "modified" -e "mask")
        fi
        
        if [ ! -z "$image" ]; then
            base_name=$(basename "${image%.nii.gz}")
            echo $base_name
            
            output_name="${reg_dir}/${base_name}_registered.nii.gz"
            #output_name="${image%.nii.gz}_registered.nii.gz"
            echo "ref image: $earliest_image"
            echo "image to reg: $image"
            
            
            echo "Registering Hour-${hour} image (${image}) to reference..."
            flirt -in "$image" \
                  -ref "$earliest_image" \
                  -out "$output_name" \
                  -omat "${reg_dir}/${base_name}_to_ref.mat" \
                  #-dof 12 \
                  #-interp trilinear
           echo "Copying original bet image to reg dir..."
           cp $earliest_image $reg_dir
        fi
    done
    exit 1
  
    
    echo "Registration complete for ${gupi_dir}"
}


# Check if either -g or -list was provided
if [ -z "$single_gupi" ] && [ -z "$list_file" ]; then
    usage
fi

# Process single GUPI if specified
if [ ! -z "$single_gupi" ]; then
    process_gupi "$single_gupi"
fi

# Process list of GUPIs if specified
if [ ! -z "$list_file" ]; then
    if [ ! -f "$list_file" ]; then
        echo "Error: List file ${list_file} not found"
        exit 1
    fi
    
    while read -r gupi; do
        echo "Processing GUPI: ${gupi}"
        process_gupi "$gupi"
    done < "$list_file"
fi




