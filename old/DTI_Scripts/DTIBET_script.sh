#!/bin/bash
module load fsl


# Define input DTI data file
patient_id="19978"
patient_timepoint="acute"
working_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/DTI_Space"
bet_params="-f 0.2 -R -m"


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

# Define output names
B0_IMAGE="${working_directory}/${patient_id}_${patient_timepoint}_b0_image.nii.gz"
B0_BRAIN="${working_directory}/${patient_id}_${patient_timepoint}_nodif_brain.nii.gz"
B0_MASK="${working_directory}/${patient_id}_${patient_timepoint}_nodif_brain_mask.nii.gz"
LOG_FILE="${working_directory}/${patient_id}_${patient_timepoint}_DTI_BET_log.txt"

# Step 1: Extract the b0 image from the DTI dataset
echo "Extracting b0 image..."
fslroi $DTI_DATA $B0_IMAGE 0 1
echo "b0 DATA: $B0_IMAGE"


# Step 2: Run BET on the b0 image
echo "Running BET on b0 image..."
echo "bet $B0_IMAGE $B0_BRAIN $BET_PARAMS" >> $LOG_FILE
bet $B0_IMAGE $B0_BRAIN $BET_PARAMS

# The above command also generates a brain mask named automatically as "${B0_BRAIN}_mask.nii.gz" due to -m parameter
# which is actually $B0_MASK as per our defined variable.

# Optional: Uncomment the following line to visually inspect the results with FSLeyes
fsleyes $B0_IMAGE $B0_BRAIN &

echo "BET processing completed."



chmod +x DTIBET_script.sh

