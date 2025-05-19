#!/bin/bash
module load ants
module load fsl

# Define paths to input and reference images
RAW_T1="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/12519/ultra-fast/Hour_00030.8831-Date_20070312/U-ID11206/T1w_space/T1w_trio_P00030_12519_ultra-fast_20070312_U-ID11206.nii.gz"
CROPPED_T1="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/12519/ultra-fast/Hour_00030.8831-Date_20070312/U-ID11206/nipype/DATASINK/T1space/neckCropping/T1w_cropped.nii.gz"
CROPPED_T1_CENTER="/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI/2ZFz639/Sub-040-2ZFz639/Hour-00376_E21717/anat/nipype/DATASINK/T1space/neckCropping/T1w_cropped.nii.gz"
DTI_REF="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/12519/ultra-fast/Hour_00030.8831-Date_20070312/U-ID11206/nipype/DATASINK/DTIspace/dti/dtifitWLS_FA.nii.gz"
DTI_REF_CENTER="/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI/2ZFz639/Sub-040-2ZFz639/Hour-00376_E21717/dwi/proc_set1_nobzero/nipype/DATASINK/DTIspace/dti/dtifitWLS_FA.nii.gz"
#CAMCAN_AFF_REF="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/12519/ultra-fast/Hour_00030.8831-Date_20070312/U-ID11206/nipype/DATASINK/CamCANspace/alternative2021/antsAff_onCamCAN_T1w.nii.gz"
#AFFINE_TRANSFORM="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/12519/ultra-fast/Hour_00030.8831-Date_20070312/U-ID11206/nipype/DATASINK/CamCANspace/alternative2021/antsAff_onCamCAN_T1w_Composite.h5"
SYN_TRANSFORM_T1_DTI="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/12519/ultra-fast/Hour_00030.8831-Date_20070312/U-ID11206/nipype/DATASINK/T1space/ants_rig_DTItoT1_InverseComposite.h5"
#SYN_TRANSFORM_DTI_T1="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/12519/ultra-fast/Hour_00030.8831-Date_20070312/U-ID11206/nipype/DATASINK/T1space/ants_rig_DTItoT1_Composite.h5"
SYN_TRANSFORM_T1_DTI_CENTER="/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI/2ZFz639/Sub-040-2ZFz639/Hour-00376_E21717/dwi/proc_set1_nobzero/nipype/DATASINK/T1space/ants_rig_DTItoT1_InverseComposite.h5"

OUTPUT="/home/cmb247/Desktop/12519-ultra-fast-DTI-reg-test.nii.gz"
OUTPUT_CENTER="/home/cmb247/Desktop/12519-ultra-fast-DTI-reg-test_CENTER_version.nii.gz"

WM_MASK_CENTER="/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI/2ZFz639/Sub-040-2ZFz639/Hour-00376_E21717/anat/nipype/DATASINK/T1space/malpem/tissueMap_WM.nii.gz"

WM_MASK_LEGACY="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/12519/ultra-fast/Hour_00030.8831-Date_20070312/U-ID11206/nipype/DATASINK/T1space/malpem/tissueMap_WM.nii.gz"

OUTPUT_WM="/home/cmb247/Desktop/12519-ultra-fast-T1_WM_2_DTI-reg-test_LEGACY_version.nii.gz"

OUTPUT_WM_CENTER="/home/cmb247/Desktop/12519-ultra-fast-T1_WM_2_DTI-reg-test_CENTER_version.nii.gz"

echo "Starting ANTs transformation using both affine and SyN transforms"

# Apply the ANTs transformations in sequence (affine first, then SyN)
echo "Applying ANTs transformations..."
#antsApplyTransforms -d 3 \
#  -i "$CROPPED_T1" \
#  -o "$OUTPUT" \
#  -r "$DTI_REF" \
#  -t "$SYN_TRANSFORM_T1_DTI"

#antsApplyTransforms -d 3 \
#  -i "$WM_MASK_LEGACY" \
#  -o "$OUTPUT_WM" \
#  -r "$DTI_REF" \
#  -t "$SYN_TRANSFORM_T1_DTI"




antsApplyTransforms -d 3 \
  -i "$CROPPED_T1_CENTER" \
  -o "$OUTPUT_CENTER" \
  -r "$DTI_REF_CENTER" \
  -t "$SYN_TRANSFORM_T1_DTI_CENTER"

antsApplyTransforms -d 3 \
  -i "$WM_MASK_CENTER" \
  -o "$OUTPUT_WM_CENTER" \
  -r "$DTI_REF_CENTER" \
  -t "$SYN_TRANSFORM_T1_DTI_CENTER"



# Check if transformation was successful
if [ -f "$OUTPUT" ]; then
  echo "Transformed image saved to: $OUTPUT"
  
  # Open FSLeyes to check the results
  echo "Opening FSLeyes to compare transformed image with reference..."
  #fsleyes "$OUTPUT" -cm red-yellow "$DTI_REF" -cm blue-lightblue -a 50 "$OUTPUT_WM" &
else
  echo "Transformation failed. Output file not created."
  exit 1
fi


##################################################

# Input T1 coordinates to transform
T1_X=133
T1_Y=158
T1_Z=87

# Create and transform ROI
ROI="/home/cmb247/Desktop/point_roi.nii.gz"
TRANSFORMED_ROI="/home/cmb247/Desktop/transformed_roi.nii.gz"

# Create ROI at T1 coordinates
fslmaths "$CROPPED_T1" -mul 0 "$ROI"
#fslmaths "$ROI" -add 1 -roi $((T1_X-1)) 3 $((T1_Y-1)) 3 $((T1_Z-1)) 3 0 1 "$ROI"
fslmaths "$ROI" -add 1 -roi $T1_X 2 $T1_Y 2 $T1_Z 2 0 1 "$ROI"

# Transform ROI to DTI space
antsApplyTransforms -d 3 \
  -i "$ROI" \
  -o "$TRANSFORMED_ROI" \
  -r "$DTI_REF" \
  -t "$SYN_TRANSFORM_T1_DTI" \
  -n NearestNeighbor

# Get coordinates in DTI space
MAX_COORDS=$(fslstats "$TRANSFORMED_ROI" -x)

echo "T1 coordinates ($T1_X, $T1_Y, $T1_Z) → DTI coordinates: $MAX_COORDS"

echo "fsleyes /home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/12519/ultra-fast/Hour_00030.8831-Date_20070312/U-ID11206/nipype/DATASINK/DTIspace/dti/dtifitWLS_FA.nii.gz -cm greyscale /home/cmb247/Desktop/transformed_roi.nii.gz -cm red-yellow"






exit 0
####################################################




echo "Process completed."
