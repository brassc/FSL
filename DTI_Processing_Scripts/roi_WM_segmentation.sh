#!/bin/bash
module load ants
module load fsl

# Input parameters
patient_id=$1
timepoint=$2
tp_base=$3     # Base directory for the timepoint
bin_size=$4    # Size of the bin for the rings
num_bins=$5    # Number of bins for the rings
filter_fa_values=${6:-"true"}  # Optional filter flag, defaults to true

# Set ROI dir
if [[ "$patient_id" =~ ^[0-9]+$ ]]; then
    # Patient ID contains only numbers (LEGACY)
    base_dir="/home/cmb247/rds/rds-uda-2-pXaBn8E6hyM/users/cmb247/cmb247_working/DECOMPRESSION_Legacy_CB/hemi/$patient_id/$timepoint"
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox"
    
    # Adjust directory name based on bin configuration
    if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox_NEW"
    elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${timepoint}/roi_files_${num_bins}x${bin_size}vox_NEW"
    fi
    
    # Find the DTIspace directory
    dti_dir=$(find "$base_dir" -type d -name "DTIspace" | head -n 1)
    t1_space_dir=$(find "$base_dir" -type d -name "T1space" | head -n 1)
    wm_mask_path="$t1_space_dir/malpem/tissueMap_WM.nii.gz"
    dti_ref="$dti_dir/dti/dtifitWLS_FA.nii.gz"
    transform_path="$t1_space_dir/ants_rig_DTItoT1_InverseComposite.h5"
    
else
    # Patient ID contains letters and numbers (CENTER)
    base_dir="/home/cmb247/rds/hpc-work/Feb2025_data/CT_Brass/Charlotte_brass_Feb2025/MRI/$patient_id"
    roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox"
    
    # Adjust directory name based on bin configuration
    if [ $num_bins -eq 5 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox_NEW"
    elif [ $num_bins -eq 10 ] && [ $bin_size -eq 4 ]; then
        roi_dir="/home/cmb247/rds/hpc-work/April2025_DWI/$patient_id/${tp_base}_dwi/roi_files_${num_bins}x${bin_size}vox_NEW"
    fi
    
    # Find the directories
    tp_dir=$(find "$base_dir" -type d -name "Sub-*" -exec find {} -type d -name "Hour-${timepoint}_*" \; | head -n 1)
    dti_dir="${tp_dir}/dwi/proc_set1_nobzero/nipype/DATASINK/DTIspace"
    t1_space_dir="${tp_dir}/anat/nipype/DATASINK/T1space"
    wm_mask_path="$t1_space_dir/malpem/tissueMap_WM.nii.gz"
    dti_ref="$dti_dir/dti/dtifitWLS_FA.nii.gz"
    transform_path="$tp_dir/dwi/proc_set1_nobzero/nipype/DATASINK/T1space/ants_rig_DTItoT1_InverseComposite.h5"
fi

# Add filtered suffix if needed
if [ "$filter_fa_values" == "true" ]; then
    roi_dir="${roi_dir}_filtered"
fi

# Create WM mask directory in ROI directory
wm_mask_dir="$roi_dir/WM_mask_DTI_space"
mkdir -p $wm_mask_dir

echo "Processing patient $patient_id at timepoint $timepoint"
echo "ROI directory: $roi_dir"
echo "WM mask path: $wm_mask_path"
echo "DTI reference: $dti_ref"
echo "Transform path: $transform_path"

# Check if required files exist
if [ ! -f "$wm_mask_path" ]; then
    echo "Error: WM mask not found at $wm_mask_path"
    exit 1
fi

if [ ! -f "$dti_ref" ]; then
    echo "Error: DTI reference not found at $dti_ref"
    exit 1
fi

if [ ! -f "$transform_path" ]; then
    echo "Error: Transform not found at $transform_path"
    exit 1
fi

exit 0

# Apply transform to register WM mask to DTI space
echo "Registering WM mask to DTI space..."
wm_mask_dti="$wm_mask_dir/wm_mask_dti.nii.gz"

antsApplyTransforms -d 3 \
  -i "$wm_mask_path" \
  -o "$wm_mask_dti" \
  -r "$dti_ref" \
  -t "$transform_path" \
  -n NearestNeighbor

# Binarize the WM mask
echo "Binarizing WM mask..."
wm_mask_bin="$wm_mask_dir/wm_mask_dti_bin.nii.gz"
fslmaths "$wm_mask_dti" -bin "$wm_mask_bin"

# Process all ROIs for all parameters (FA, MD)
for parameter in FA MD; do
    echo "Processing $parameter ROIs with WM mask..."
    mkdir -p "$wm_mask_dir/$parameter"
    
    for label in ant post baseline_ant baseline_post; do
        echo "Processing $label rings..."
        
        for ((i=1; i<=$num_bins; i++)); do
            echo "Processing ring $i..."
            
            # Paths to existing ROI files
            roi_ring="$roi_dir/$parameter/${label}_ring${i}.nii.gz"
            roi_metric="$roi_dir/$parameter/${label}_ring${i}_${parameter}.nii.gz"
            
            # Check if the input files exist
            if [ ! -f "$roi_ring" ] || [ ! -f "$roi_metric" ]; then
                echo "Warning: Required files not found for $label ring $i, skipping..."
                continue
            fi
            
            # Output paths for WM-masked files
            wm_roi_ring="$wm_mask_dir/$parameter/${label}_ring${i}_wm.nii.gz"
            wm_roi_metric="$wm_mask_dir/$parameter/${label}_ring${i}_${parameter}_wm.nii.gz"
            
            # Create WM-masked ROI ring
            fslmaths "$roi_ring" -mul "$wm_mask_bin" "$wm_roi_ring"
            
            # Create WM-masked metric values
            fslmaths "$roi_metric" -mul "$wm_mask_bin" "$wm_roi_metric"
        done
    done
done

echo "WM mask registration and ROI processing complete for patient $patient_id at timepoint $timepoint"
echo "Results stored in $wm_mask_dir"
exit 0








exit 0














########################################################################################################
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
