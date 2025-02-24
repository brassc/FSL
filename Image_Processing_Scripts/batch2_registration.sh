#!/bin/bash

# Load FSL module
module load fsl

# Base path for GUPI directories
GUPI_BASE_PATH="/rds-d5/user/cmb247/hpc-work/Feb2025_working"

# Function to print usage
usage() {
    echo "Usage: $0 [-g GUPI] [-list list_file]"
    echo "  -g    : Single GUPI directory to process"
    echo "  -l : File containing list of GUPI directories"
    exit 1
}

# Parse command line arguments
while getopts "g:l:" opt; do
    case $opt in
        g)
            single_gupi=$OPTARG
            ;;
        l)
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

    # First find the earliest hour number from all images (modified or not)
    earliest_hour=$(find "$bet_dir" -name "*Hour-[0-9]*" | grep -v "mask" | sort | head -n 1 | xargs basename | grep -o "Hour-[0-9]*" | grep -o "[0-9]*")

    if [ -z "$earliest_hour" ]; then
        echo "Error: No Hour-XXXXX files found in ${bet_dir}"
        return 1
    fi
    
    # Now try to find modified version of earliest hour
    earliest_image=$(find "$bet_dir" -name "*Hour-${earliest_hour}*modified*" | grep -v "mask" | head -n 1)
    earliest_mask=$(find "$bet_dir" -name "*Hour-${earliest_hour}*modifiedmask*" | head -n 1)

    # If no modified version exists, use non-modified version
    if [ -z "$earliest_image" ]; then
        echo "No modified version found for earliest hour, using non-modified version..."
        earliest_image=$(find "$bet_dir" -name "*Hour-${earliest_hour}*" | grep -v -e "modified" -e "mask" | head -n 1)
        earliest_mask=$(find "$bet_dir" -name "*Hour-${earliest_hour}*mask*" | grep -v "modified" | head -n 1)
    fi



    echo "Reference image: ${earliest_image}"
    echo "Reference mask: ${earliest_mask}"
    
    
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
        modified_image=$(find "$bet_dir" -name "*Hour-${hour}*modified*" | grep -v "mask" | head -n 1)
        modified_mask=$(find "$bet_dir" -name "*Hour-${hour}*modifiedmask*" | head -n 1)
        
        
        if [ ! -z "$modified_image" ]; then
            image="$modified_image"
            mask="$modified_mask"
        else
            # Use non-modified version if no modified exists
            image=$(find "$bet_dir" -name "*Hour-${hour}*" | grep -v -e "modified" -e "mask" | head -n 1)
            mask=$(find "$bet_dir" -name "*Hour-${hour}*mask" | grep -v "modified" | head -n 1)
        fi
        
        if [ ! -z "$image" ]; then
            base_name=$(basename "${image%.nii.gz}")
            base_name_mask=$(basename "${mask%.nii.gz}")
            echo $base_name
            
            output_name="${reg_dir}/${base_name}_registered.nii.gz"
            output_mask_name="${reg_dir}/${base_name_mask}_registered.nii.gz"
            #output_name="${image%.nii.gz}_registered.nii.gz"
            echo "ref image: $earliest_image"
            echo "image to reg: $image"
            
            omat="${reg_dir}/${base_name}_to_ref.mat"
            
            # Check if registration has already been done, if so skip flirt but do fnirt
            if [ -f "$output_name" ]; then
                echo "Linear registration already done for Hour-${hour}, skipping..."
                
            else
                
                echo "Registering Hour-${hour} image (${image}) to reference..."
                flirt -in "$image" \
                    -ref "$earliest_image" \
                    -out "$output_name" \
                    -omat "$omat" \
                    #-dof 12 \
                    #-interp trilinear
                if [ $? -ne 0 ]; then
                    echo "Error: FLIRT failed for Hour-${hour}"
                    continue
                else
                    echo "binarising mask"
                    fslmaths $output_name -bin "${reg_dir}/${base_name}_registeredmask.nii.gz"
                fi
                
            fi

            

            # Check if fnirt has already been done, if so skip
            if [ -f "${reg_dir}/${base_name}_registered_fnirt.nii.gz" ]; then
                echo "fnirt already done for Hour-${hour}, skipping..."
            else
                echo "Performing fnirt on Hour-${hour} image... for GUPI ${gupi_name}"
                
                fnirt \
                    --ref="$earliest_image" \
                    --in="$output_name" \
                    --aff="$omat" \
                    --cout="${reg_dir}/${base_name}_to_ref_warp" \
                    --iout="${reg_dir}/${base_name}_registered_fnirt.nii.gz"

                # if iout file exists, binarise it
                if [ -f "${reg_dir}/${base_name}_registered_fnirt.nii.gz" ]; then
                    echo "binarising mask"
                    fslmaths "${reg_dir}/${base_name}_registered_fnirt.nii.gz" -bin "${reg_dir}/${base_name}_registeredmask_fnirt.nii.gz"
                fi
                
            fi
            
        fi
    done
    # check if original image and mask have been copied to reg dir
    if [ -f "${reg_dir}/${earliest_image}" ]; then
        echo "Original image already copied to reg dir, skipping..."
    else
        echo "Copying original bet image and mask to reg dir..."
        cp $earliest_image $reg_dir
        cp $earliest_mask $reg_dir
    fi    
  
    
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




