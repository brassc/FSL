#!/bin/bash

# Load FSL module
module load fsl
# load ants for nonlinear reg
module load ants

# Base path for GUPI directories
GUPI_BASE_PATH="/rds-d5/user/cmb247/hpc-work/Feb2025_working"

# Function to print usage
usage() {
    echo "Usage: $0 [-g GUPI] [-l list_file] [-o]"
    echo "  -g    : Single GUPI directory to process"
    echo "  -l : File containing list of GUPI directories"
    echo " -o : Overwrite existing FNIRT outputs (will prompt for confirmation)"
    exit 1
}

# Parse command line arguments
overwite_fnirt=false
while getopts "g:l:o" opt; do
    case $opt in
        g)
            single_gupi=$OPTARG
            ;;
        l)
            list_file=$OPTARG
            ;;
        o)
            overwrite_fnirt=true
            ;;
        *)
            usage
            ;;
    esac
done

# If overwrite_fnirt is true, prompt for confirmation
if [ "$overwrite_fnirt" = true ]; then
    read -p "Are you sure you want to overwrite existing FNIRT outputs? (y/n): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 1
    fi
    echo "Proceeding with FNIRT overwrite."
fi


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
    local bias_dir="${gupi_dir}/bias_corr"


    if [ ! -d "$gupi_dir" ]; then
        echo "Error: GUPI directory not found: ${gupi_dir}"
        return 1
    fi
    echo "GUPI dir: $gupi_dir"
    echo "BET dir: $bet_dir"
    echo "Bias dir: $bias_dir"
    
    
    # Check if BET_Output directory exists
    if [ ! -d "$bet_dir" ]; then
        echo "Error: BET_Output directory not found in ${gupi_dir}"
        return 1
    fi

    # Check if bias_corr directory exists
    if [ ! -d "$bias_dir" ]; then
        echo "Error: bias_corr directory not found in ${gupi_dir}"
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

    # Create bias_reg directory if it doesn't exist
    bias_reg_dir="${gupi_dir}/bias_reg"
    mkdir -p "$bias_reg_dir"

    #set bias_corr dir
    bias_corr_dir="${gupi_dir}/bias_corr"


    # Process images by hour, preferring modified versions
    for hour in "${hour_numbers[@]}"; do
        # Skip if this is the reference image's hour
        if [[ "$earliest_image" =~ Hour-${hour} ]]; then
            continue
        fi

        # find non-BET bias image
        bias_image=$(find "$bias_corr_dir" -name "T1_bias-Hour-${hour}*" | head -n 1)
        
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

            

            # # Check if fnirt has already been done, if so skip
            # if [ -f "${reg_dir}/${base_name}_registered_ants.nii.gz" ]; then
            #     if [ "$overwrite_fnirt" = true ]; then
            #         echo "Overwriting existing ANTs output for Hour-${hour}"
            #         echo "Performing nonlinear registration on GUPI ${gupi_name} Hour-${hour} image..."

            #         # UPDATE FNIRT IN BOTH PLACES FOR OVERWRITE AND FOR NOT OVERWRITE
            #         antsRegistration \
            #             --dimensionality 3 \
            #             --float 0 \
            #             --output [${reg_dir}/${base_name}_to_ref_,${reg_dir}/${base_name}_registered_ants.nii.gz] \
            #             --interpolation Linear \
            #             --use-histogram-matching 1 \
            #             --winsorize-image-intensities [0.005,0.995] \
            #             --initial-moving-transform [${earliest_image},${output_name},1] \
            #             --transform Rigid[0.1] \
            #             --metric MI[${earliest_image},${output_name},1,32,Regular,0.25] \
            #             --convergence [1000x500x250x100,1e-6,10] \
            #             --shrink-factors 8x4x2x1 \
            #             --smoothing-sigmas 3x2x1x0vox \
            #             --transform Affine[0.1] \
            #             --metric MI[${earliest_image},${output_name},1,32,Regular,0.25] \
            #             --convergence [1000x500x250x100,1e-6,10] \
            #             --shrink-factors 8x4x2x1 \
            #             --smoothing-sigmas 3x2x1x0vox \
            #             --transform SyN[0.1,3,0] \
            #             --metric CC[${earliest_image},${output_name},1,4] \
            #             --convergence [100x70x50x20,1e-6,10] \
            #             --shrink-factors 8x4x2x1 \
            #             --smoothing-sigmas 3x2x1x0vox
            #         # fnirt \
            #         #     --ref="$earliest_image" \
            #         #     --in="$output_name" \
            #         #     --aff="$omat" \
            #         #     --cout="${reg_dir}/${base_name}_to_ref_warp" \
            #         #     --iout="${reg_dir}/${base_name}_registered_fnirt.nii.gz" \
            #         #     --lambda=1500,750,400,200 \
            #         #     --warpres=30,30,30

            #         # if iout file exists, binarise it
            #         if [ -f "${reg_dir}/${base_name}_registered_ants.nii.gz" ]; then
            #             echo "binarising mask"
            #             fslmaths "${reg_dir}/${base_name}_registered_ants.nii.gz" -bin "${reg_dir}/${base_name}_registeredmask_ants.nii.gz"
            #         fi   
            #     else
            #         echo "fnirt already done for Hour-${hour}, skipping..."
            #     fi
            # else
            #     echo "Performing nonlinear registration on GUPI ${gupi_name} Hour-${hour} image..."
                
            #     # UPDATE FNIRT IN BOTH PLACES FOR OVERWRITE AND FOR NOT OVERWRITE
            #     antsRegistration \
            #             --dimensionality 3 \
            #             --float 0 \
            #             --output [${reg_dir}/${base_name}_to_ref_,${reg_dir}/${base_name}_registered_ants.nii.gz] \
            #             --interpolation Linear \
            #             --use-histogram-matching 1 \
            #             --winsorize-image-intensities [0.005,0.995] \
            #             --initial-moving-transform [${earliest_image},${output_name},1] \
            #             --transform Rigid[0.1] \
            #             --metric MI[${earliest_image},${output_name},1,32,Regular,0.25] \
            #             --convergence [1000x500x250x100,1e-6,10] \
            #             --shrink-factors 8x4x2x1 \
            #             --smoothing-sigmas 3x2x1x0vox \
            #             --transform Affine[0.1] \
            #             --metric MI[${earliest_image},${output_name},1,32,Regular,0.25] \
            #             --convergence [1000x500x250x100,1e-6,10] \
            #             --shrink-factors 8x4x2x1 \
            #             --smoothing-sigmas 3x2x1x0vox \
            #             --transform SyN[0.1,3,0] \
            #             --metric CC[${earliest_image},${output_name},1,4] \
            #             --convergence [100x70x50x20,1e-6,10] \
            #             --shrink-factors 8x4x2x1 \
            #             --smoothing-sigmas 3x2x1x0vox
            #     # fnirt \
            #     #     --ref="$earliest_image" \
            #     #     --in="$output_name" \
            #     #     --aff="$omat" \
            #     #     --cout="${reg_dir}/${base_name}_to_ref_warp" \
            #     #     --iout="${reg_dir}/${base_name}_registered_fnirt.nii.gz" \
            #     #     --lambda=1500,750,400,200 \
            #     #     --warpres=30,30,30

            #     # if iout file exists, binarise it
            #     if [ -f "${reg_dir}/${base_name}_registered_ants.nii.gz" ]; then
            #         echo "binarising mask"
            #         fslmaths "${reg_dir}/${base_name}_registered_ants.nii.gz" -bin "${reg_dir}/${base_name}_registeredmask_ants.nii.gz"
            #     fi
                
            # fi
            # Move original images along linear transform to new directory $bias_reg_dir
            # get basename of bias image
            base_name_bias_img=$(basename "${bias_image%.nii.gz}")
            if [ -f "${bias_reg_dir}/${base_name_bias_img}_registered.nii.gz" ]; then
                echo "Original image already transformed along linear transform to bias_reg, skipping..."
            else
                # apply $omat to original image
                echo "bias non-bet image to register: $bias_image"
                
                echo "Applying linear transform to original image..."
                flirt -in $bias_image \
                    -ref "$earliest_image" \
                    -out "${bias_reg_dir}/${base_name_bias_img}_registered.nii.gz" \
                    -applyxfm -init "$omat" \
                    #-dof 12 \
                    #-interp trilinear
                
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

    # check if original image has been copied to bias_reg dir from bias_corr
    earliest_nonbet_image=$(find "$bias_corr_dir" -name "*Hour-${earliest_hour}*" | head -n 1)
    if [ -f "${bias_reg_dir}/${earliest_nonbet_image}" ]; then
        echo "Original non-bet image already copied to bias_reg dir, skipping..."
    else
        echo "Copying original non-bet image to bias_reg dir..."
        cp $earliest_nonbet_image $bias_reg_dir
    fi
  
    
    echo "Registration complete for ${gupi_dir}"
    #fsleyes ${gupi_dir}/bias_reg/*.nii.gz
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




