#!/bin/bash
module load fsl


# Define input DTI data file
patient_id="19978"
patient_timepoint="acute"
working_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/DTI_Space"
# Check if the directory exists
if [ ! -d "$working_directory" ]; then
    # Directory doesn't exist, create it
    mkdir -p "$working_directory"
    if [ $? -ne 0 ]; then
        echo "Failed to create the directory."
        exit 1
    else
        echo "Directory created successfully."
    fi
else
    echo "Directory already exists."
fi


## Check if the user has provided a timepoint argument
#if [ $# -eq 0 ]; then
#    echo "Usage: $0 <timepoint>"
#    exit 1
#fi

## Extract the timepoint argument
#timepoint=$1


#Find DWI raw data
files=$(find "/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/OG_Scans/" -name "*DWI*$patient_timepoint*.nii*")


# Loop through each file found
for file in $files; do
    # Extract basename and full path
    basename=$(basename "$file")
    fullpath=$(realpath "$file")
    
    # Set the DTI_DATA variable
    DTI_DATA="${fullpath}"
    
    # Output the variable
    echo "DTI_DATA: $DTI_DATA"
done

if [ ! -f "$DTI_DATA" ]; then
    echo "The DTI data file does not exist at the specified path: $DTI_DATA"
    exit 1 
fi


# Find T1 BET image
files=$(find "/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1w_time1_registered_scans" -name "$patient_timepoint*restored*bet-*.nii.gz")


# Loop through each file found
for file in $files; do
    # Extract basename and full path
    basename=$(basename "$file")
    fullpath=$(realpath "$file")
    
    # Set the DTI_DATA variable
    T1_BET="${fullpath}"
    
    # Output the variable
    echo "T1_BET: $T1_BET"
done


if [ ! -f "$T1_BET" ]; then
    echo "The T1 BET data file does not exist at the specified path: $T1_BET"
    exit 1
fi

# Find T1 BET image
files=$(find "/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1w_time1_registered_scans" -name "$patient_timepoint*bet_mask*.nii.gz")


# Loop through each file found
for file in $files; do
    # Extract basename and full path
    basename=$(basename "$file")
    fullpath=$(realpath "$file")
    
    # Set the DTI_DATA variable
    T1_BET_MASK="${fullpath}"
    
    # Output the variable
    echo "T1_BET_MASK: $T1_BET_MASK"
done


if [ ! -f "$T1_BET_MASK" ]; then
    echo "The T1 BET mask file does not exist at the specified path: $T1_BET_MASK"
    flag=2
 
fi






# Define output names
#B0_IMAGE="${working_directory}/${patient_id}_${patient_timepoint}_b0_image.nii.gz"
#B0_BRAIN="${working_directory}/${patient_id}_${patient_timepoint}_nodif_brain.nii.gz"
#B0_MASK="${working_directory}/${patient_id}_${patient_timepoint}_nodif_brain_mask.nii.gz"
LOG_FILE="${working_directory}/${patient_id}_${patient_timepoint}_T1toDTI_reg_log.txt"
OUTPUT_IMAGE="$working_directory/registered_${patient_timepoint}_T1_BET_to_DWI.nii.gz"
OUTPUT_MATRIX="$working_directory/registered_${patient_timepoint}_T1_BET_to_DWI_mat.mat"
T1XFM_BET_MASK="$working_directory/registered_${patient_timepoint}_T1_BET_to_DWI_mask.nii.gz"
echo "T1XFM_BET_MASK: $T1XFM_BET_MASK"
DTI_BET_MASK="$working_directory/registered_${patient_timepoint}_DWI_BET_mask.nii.gz"

# Register T1_BET to DTI_data
flirt -in $T1_BET -ref $DTI_DATA -out $OUTPUT_IMAGE -omat $OUTPUT_MATRIX -dof 6

# Create binary mask from output image (T1 BET xfm)
fslmaths $OUTPUT_IMAGE -bin $T1XFM_BET_MASK

# multiply T1 BET xfm by DTI_DATA to get DTI_BET image
fslmaths $DTI_DATA -mul $T1XFM_BET_MASK $DTI_BET_MASK



chmod +x T1toDTIspace.sh

