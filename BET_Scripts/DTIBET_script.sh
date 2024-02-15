#!/bin/bash

# Define input DTI data file
DTI_DATA="data.nii.gz"

# Define output names
B0_IMAGE="b0_image.nii.gz"
B0_BRAIN="b0_image_brain.nii.gz"
B0_MASK="b0_image_brain_mask.nii.gz"
LOG_FILE="DTI_BET_log.txt"

# Step 1: Extract the b0 image from the DTI dataset
echo "Extracting b0 image..."
fslroi $DTI_DATA $B0_IMAGE 0 1

# Step 2: Run BET on the b0 image
echo "Running BET on b0 image..."
echo bet $B0_IMAGE $B0_BRAIN -f 0.3 -g 0.1 -m >> $LOG_FILE

# The above command also generates a brain mask named automatically as "${B0_BRAIN}_mask.nii.gz" due to -m parameter
# which is actually $B0_MASK as per our defined variable.

# Optional: Uncomment the following line to visually inspect the results with FSLeyes
#fsleyes $B0_IMAGE $B0_BRAIN &

echo "BET processing completed."

chmod +x DTIBET_script.sh

