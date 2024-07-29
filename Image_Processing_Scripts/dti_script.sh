#!/bin/bash

# Script to flatten cortex using freesurfer

# Load modules
module load freesurfer

export SUBJECTS_DIR=/home/cmb247/Desktop/Project_3/Freesurfer/
echo "SUBJECTS_DIR is: $SUBJECTS_DIR"

patient_id=19978
timepoint=ultra-fast



# Select input_basename based on timepoint

# Function to find T1 input scan basename based on timepoint
find_input_basename() {
    local input_basename
    local found=0
    
    
    while IFS= read -r file; do
        basename=$(basename "$file")
        if [[ "$basename" == "T1"* && "$basename" == *"$timepoint"* && "$basename" == *"restore_registered.nii.gz" ]]; then
            if [[ "$timepoint" == "fast" && "$basename" != *"ultra-fast"* ]]; then
                input_basename=$basename
                found=1
                echo "$input_basename"
                break
            elif [[ "$timepoint" != "fast" || "$basename" != *"ultra-fast"* ]]; then
                input_basename=$basename
                found=1
                echo "$input_basename"
                break
            fi
        fi
    done < <(find "$input_directory" -type f)

    if [ $found -eq 0 ]; then
        echo "Error: No matching file found for patient_id $patient_id and timepoint $timepoint."
        exit 1
    fi
}

# Function to find input mask basename based on timepoint
find_mask_input_basename() {
    local input_basename
    local found=0
    
    
    while IFS= read -r file; do
        basename=$(basename "$file")
        if [[ "$basename" == "T1"* && "$basename" == *"$timepoint"* && "$basename" == *"restore_registered_bet_mask"*".nii.gz" ]]; then
            if [[ "$timepoint" == "fast" && "$basename" != *"ultra-fast"* ]]; then
                input_basename=$basename
                found=1
                echo "$input_basename"
                break
            elif [[ "$timepoint" != "fast" || "$basename" != *"ultra-fast"* ]]; then
                input_basename=$basename
                found=1
                echo "$input_basename"
                break
            fi
        fi
    done < <(find "$input_bet_directory" -type f)

    if [ $found -eq 0 ]; then
        echo "Error: No matching file found for patient_id $patient_id and timepoint $timepoint."
        exit 1
    fi
}



input_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1w_time1_bias_corr_registered_scans/"
input_bet_directory="/home/cmb247/Desktop/Project_3/BET_Extractions/$patient_id/T1w_time1_bias_corr_registered_scans/BET_Output/"

# Get T1 input basename
input_basename=$(find_input_basename)
# Get BET mask input basename
mask_basename=$(find_mask_input_basename)
destination_dir="/home/cmb247/Desktop/Project_3/Freesurfer/${patient_id}_$timepoint/mri/"

# Check if the destination directory exists, if not, create it
if [ ! -d "${destination_dir}orig/" ]; then
    mkdir -p "${destination_dir}orig/"
fi


# Copy the file to the destination directory
#cp "$input_directory$input_basename" "${destination_dir}orig/"
echo "T1 copied to ${destination_dir}orig/"

# Copy brain mask to destination dir
#cp "${input_bet_directory}$mask_basename" "${destination_dir}orig/"
echo "BET mask copied to $destination_dir"



# Converting to freesurfer .mgz format
echo "Converting .nii to freesurfer .mgz..."
#mri_convert "${destination_dir}orig/$input_basename" "${destination_dir}T1.mgz"
#mri_convert "${destination_dir}orig/$mask_basename" "${destination_dir}brainmask.mgz"
echo "conversion complete."

# apply mask:
echo "applying brainmask to T1 to create brain.mgz..."
#mri_mask "${destination_dir}T1.mgz" "${destination_dir}brainmask.mgz" "${destination_dir}brain.mgz"
echo "BET complete."

# rename T1 as orig and nu, place inside orig folder as 001
#cp "${destination_dir}T1.mgz" "${destination_dir}orig/001.mgz"
#cp "${destination_dir}T1.mgz" "${destination_dir}nu.mgz"
#mv "${destination_dir}T1.mgz" "${destination_dir}orig.mgz"

# DO AUTORECON1 OPERATIONS:
# Normalise the intensities of bias corrected image
echo "normalising intensities..."
#mri_normalize -g 1 -mprage -mask "${destination_dir}/brainmask.mgz" "${destination_dir}/orig/001.mgz" "${destination_dir}/norm.mgz"


# register brain.mgz to talairach
#talairach_avi --i "${destination_dir}/brain.mgz" --xfm "${destination_dir}/transforms/talairach.xfm"



# Surface generation
#recon-all -s "${patient_id}_${timepoint}" -fill
#recon-all -s "${patient_id}_${timepoint}" -tessellate
#recon-all -s "${patient_id}_${timepoint}" -smooth1
#recon-all -s "${patient_id}_${timepoint}" -inflate1
#recon-all -s "${patient_id}_${timepoint}" -qsphere
#recon-all -s "${patient_id}_${timepoint}" -fix
#recon-all -s "${patient_id}_${timepoint}" -white
#recon-all -s "${patient_id}_${timepoint}" -smooth2
#recon-all -s "${patient_id}_${timepoint}" -inflate2







# Starting recon-all
echo "Starting recon-all with BET brain.mgz, without registration"
#recon-all -s "${patient_id}_${timepoint}" -autorecon1 -notalairach -noskullstrip 
echo "Stage 1 complete, starting stage 2..."
echo "force generate aseg.presurf.mgz..."
#mri_watershed -surf "${destination_dir}aseg.presurf.mgz" "${destination_dir}T1.mgz" "${destination_dir}brainmask.mgz"
# White matter segmentation
#echo "starting white matter seg..."
#recon-all -s "${patient_id}_${timepoint}" -autorecon2 -noaseg -nofix


#echo "completed autorecon2"
#recon-all -s "${patient_id}_${timepoint}" -autorecon2-wm -noaseg
#echo "completed wm"
# Create surface models
#echo "starting surface models -autorecon2-pial"
#recon-all -s "${patient_id}_${timepoint}" -autorecon2-pial
#echo "completed surface modelling"
# Run the remaining surface-based steps
#echo "transforming to sphere..."
#recon-all -s "${patient_id}_${timepoint}" -sphere -surfreg -jacobian_white -avgcurv -cortparc
#recon-all -s "${patient_id}_${timepoint}" -autorecon2 -noskullstrip 


echo "Stage 2 complete, starting stage 3..."
recon-all -s "${patient_id}_${timepoint}" -autorecon3 -noaseg
echo "recon-all complete!"







