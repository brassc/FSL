# BASEPATH='/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI/'
# SOURCE_DIR='$patient_id/<whatever name>/<hour_directories>/dwi'
# <hour_directories> there are some number of these, all starting with Hour
# DEST_DIR="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id"

# For all patient_id directories in BASEPATH, copy dwi dir (source dir) to destination directory

#!/bin/bash

# Define paths
BASEPATH='/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI/'
DEST_BASE="/home/cmb247/rds/hpc-work/April2025_DWI"

# Check if base directories exist
if [ ! -d "$BASEPATH" ]; then
    echo "Error: Source directory $BASEPATH does not exist."
    exit 1
fi

if [ ! -d "$DEST_BASE" ]; then
    echo "Creating destination directory $DEST_BASE"
    mkdir -p "$DEST_BASE"
fi

# Loop through all patient directories in BASEPATH
for patient_dir in "$BASEPATH"*/; do
    # Extract patient_id (directory name)
    patient_id=$(basename "$patient_dir")
    echo "Processing patient: $patient_id"
    
    # Create destination directory for this patient if it doesn't exist
    patient_dest="$DEST_BASE/$patient_id"
    if [ ! -d "$patient_dest" ]; then
        echo "Creating directory: $patient_dest"
        mkdir -p "$patient_dest"
    fi
    
    # Find all dwi directories for this patient
    # This recursively searches through all subdirectories including Hour* directories
    find "$patient_dir" -type d -path "*/Hour*/dwi" | while read dwi_dir; do
        # Get the Hour directory name for reference
        hour_dir=$(basename "$(dirname "$dwi_dir")")
        
        # Create a more descriptive destination name that includes the hour info
        dest_with_hour="$patient_dest/${hour_dir}_dwi"
        
        echo "Copying $dwi_dir to $dest_with_hour"
        
        # Copy the directory with all contents
        cp -r "$dwi_dir" "$dest_with_hour"
        
        # Check if copy was successful
        if [ $? -eq 0 ]; then
            echo "Successfully copied $dwi_dir to $dest_with_hour"
        else
            echo "Error copying $dwi_dir to $dest_with_hour"
        fi
    done
done

echo "DWI data copy complete"