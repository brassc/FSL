#!/bin/bash
module load fsl


# Define input DTI data file
patient_id="19978"
patient_timepoint="acute"
working_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/DTI_Space"

## Check if the user has provided a timepoint argument
#if [ $# -eq 0 ]; then
#    echo "Usage: $0 <timepoint>"
#    exit 1
#fi

## Extract the timepoint argument
#timepoint=$1


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





#Find DWI raw data
files=$(find "/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/OG_Scans/" -name "*DWI*$patient_timepoint*.nii*")


# Loop through each file found
for file in $files; do
    # Extract basename and full path
    basename=$(basename "$file")
    fullpath=$(realpath "$file")
    basepath="${fullpath%.nii.gz}"

    # Assume .bvecs and .bvals share the same basename with the DWI data
    bvecs_path="${basepath}.bvec"
    bvals_path="${basepath}.bval"
    
    # Set the DTI_DATA, BVECS, and BVALS variables
    DTI_DATA="$fullpath"
    BVECS="$bvecs_path"
    BVALS="$bvals_path"
    
    # Output the variables
    echo "DTI_DATA: $DTI_DATA"
    echo "BVECS: $BVECS"
    echo "BVALS: $BVALS"
done

if [ ! -f "$DTI_DATA" ]; then
    echo "The DTI data file does not exist at the specified path: $DTI_DATA"
    exit 1 
fi

if [ ! -f "$BVECS" ]; then
    echo "The DTI bvecs data file does not exist at the specified path: $BVECS"
    exit 1 
fi

if [ ! -f "$BVALS" ]; then
    echo "The DTI bvals data file does not exist at the specified path: $BVALS"
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
DTIFIT_LOG="${working_directory}/${patient_id}_${patient_timepoint}_dtifit.log"
OUTPUT_IMAGE="$working_directory/${patient_id}_${patient_timepoint}_T1_BET_DWIspace.nii.gz"
OUTPUT_MATRIX="$working_directory/${patient_id}_${patient_timepoint}_T1_BET_DWIspace_mat.mat"
T1XFM_BET_MASK="$working_directory/${patient_id}_${patient_timepoint}_T1_BET_DWIspace_mask.nii.gz"
echo "T1XFM_BET_MASK: $T1XFM_BET_MASK"
DTI_BET_IMAGE="$working_directory/${patient_id}_${patient_timepoint}_DWI_BET_image.nii.gz"
DTIFIT_OUTPUT="$working_directory/${patient_id}_${patient_timepoint}_DTIFIT_output"

# Register T1_BET to DTI_data
echo -e "\nRegistering T1_BET to DTI_DATA to create T1XFM_BET\n"
flirt -in $T1_BET -ref $DTI_DATA -out $OUTPUT_IMAGE -omat $OUTPUT_MATRIX -dof 6

# Create binary mask from T1XFM_BET
echo -e "\nCreating binary mask from output image T1XFM_BET\n"
fslmaths $OUTPUT_IMAGE -bin $T1XFM_BET_MASK

# multiply T1 BET xfm by DTI_DATA to get DTI_BET image
echo -e "Extracting BET brain region from DTI image using mask\n"
fslmaths $DTI_DATA -mul $T1XFM_BET_MASK $DTI_BET_IMAGE

# Perform dtifit
echo "Performing dtifit using DTI_BET..."
dtifit -k $DTI_BET_IMAGE -o $DTIFIT_OUTPUT -m $T1XFM_BET_MASK -r $BVECS -b $BVALS > "$DTIFIT_LOG" 2>&1
echo -e "dtifit output written to log file $DTIFIT_LOG\n"
echo "***SCRIPT COMPLETE***"

fsleyes "$OUTPUT_IMAGE" "${DTIFIT_OUTPUT}_FA.nii.gz" "${DTIFIT_OUTPUT}_V1.nii.gz"






chmod +x T1toDTIspace.sh

