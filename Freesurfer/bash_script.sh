#!/bin/bash

module load freesurfer
module load fsl

# use srun to run this script!!!
# srun -p sapphire --pty --nodes=1 --cpus-per-task=64 --time=12:00:00 bash
# check here for flags https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all

#set SUBJECTS_DIR
export SUBJECTS_DIR=/home/cmb247/Desktop/Project_3/Freesurfer/
echo "SUBJECTS_DIR is set to: $SUBJECTS_DIR"

patient_id="19978"
timepoint="acute"
nifti_dir="/home/cmb247/Desktop/Project_3/BET_Extractions/${patient_id}/T1w_time1_bias_corr_registered_scans/"
input_T1="${nifti_dir}T1w_time1_bias_corr_T1w_verio_P00030_19978_acute_20111102_U-ID22791_bias_corr_restore_registered.nii.gz"

##set up folder structure
#recon-all -i "${nifti_dir}T1w_time1_bias_corr_T1w_verio_P00030_19978_acute_20111102_U-ID22791_bias_corr_restore_registered.nii.gz" -s "${patient_id}_${timepoint}"
## supply existing bet mask
#mri_convert "${nifti_dir}BET_Output/T1w_time1_bias_corr_T1w_verio_P00030_19978_acute_20111102_U-ID22791_bias_corr_restore_registered_bet_rbc_-f_0.54_cropped_56.nii.gz" "${SUBJECTS_DIR}${patient_id}_$timepoint/mri/brainmask.mgz" # T1w_time1_bias_corr_T1w_verio_P00030_19978_acute_20111102_U-ID22791_bias_corr_restore_registered_bet_rbc_-f_0.54_cropped_56modified.nii.gz



#resample to be 256 256 256






# step 1: 
#recon-all -s 19978_acute -autorecon1 -noskullstrip -openmp 2
 
##recon-all -s 19978_acute -autorecon2 -openmp 2 #-notalairach -nogcareg -nocanorm -nocareg -noskull-lta -nocalabel -nosegstats -noaseg -nofill -notessellate -nosmooth1 -nosmooth2 -openmp 2 #-noaseg #-notess # -cw256

#recon-all -s 19978_acute -autorecon2-wm -hemi lh -openmp 8
#read -p "Left hemisphere completed. Press Enter to continue with the right hemisphere..."
#recon-all -s 19978_acute -autorecon2-wm -hemi rh -openmp 8
#read -p "wm reconstruction finished. Press Enter to continue"
#exit
##recon-all -s 19978_acute -autorecon2-wm -notalairach -nogcareg -nocanorm  -nosegstats -noaseg -nofill -notessellate -nosmooth1 -nosmooth2 -openmp 8 #-noaseg #-notess # -cw256

#recon-all -s 19978_acute -autorecon2-cp -hemi lh -openmp 8
#read -p "autorecon2-cp completed for left hemisphere. Press Enter to continue to right hemisphere..."
#recon-all -s 19978_acute -autorecon2-cp -hemi rh -openmp 8
#read -p "autorecon2-cp completed for both hemispheres. Press Enter to continue to pial reconstruction..."
#recon-all -s 19978_acute -autorecon2-pial -hemi lh -openmp 8
#read -p "autorecon2-pial completed for left hemisphere. Press Enter to continue to right hemisphere..."
#recon-all -s 19978_acute -autorecon2-pial -hemi rh -openmp 8
#read -p "autorecon2-pial completed for both hemispheres. Press Enter to continue to -autorecon3..."
#recon-all -s 19978_acute -autorecon3 -hemi lh -openmp 8 #-surfonly 
#read -p "autorecon3 completed for left hemisphere. Press Enter to continue to -autorecon3 for right hemisphere..." # failed at -aseg
#recon-all -s 19978_acute -autorecon3 -hemi rh -openmp 8 #-surfonly 
#echo "recon-all pipeline complete."

# Sphere flattening
# Flatten sphere using freesurfer's mri_flatten command
mri_flatten -s 19978_acute -t1 -n 20 -w 20 -l 20 -hemi lh -openmp 8
read -p "Flattening complete for lh. Press Enter to continue to right hemisphere..."
mri_flatten -s 19978_acute -t1 -n 20 -w 20 -l 20 -hemi rh -openmp 8





#mri_convert "${nifti_dir}BET_Output/manual_corpus_callosum.nii.gz" "${SUBJECTS_DIR}19978_acute/mri/manual_cc.mgz"

#mri_fwhm --i "${SUBJECTS_DIR}19978_acute/mri/manual_cc.mgz" --o "${SUBJECTS_DIR}19978_acute/mri/smoothed_manual_cc.mgz" --fwhm 2

#mri_convert --conform --smooth 2 "${SUBJECTS_DIR}19978_acute/mri/manual_cc.mgz" "${SUBJECTS_DIR}19978_acute/mri/smoothed_manual_cc.mgz"

# Step 1: Smooth the binary mask with Gaussian smoothing (e.g., sigma = 1.0)
#fslmaths "${nifti_dir}BET_Output/manual_corpus_callosum.nii.gz" -s 1.0 "${nifti_dir}BET_Output/smoothed_manual_corpus_callosum.nii.gz"

# Step 2: Threshold the smoothed result to binarize it again (e.g., threshold at 0.5)
#fslmaths "${nifti_dir}BET_Output/smoothed_manual_corpus_callosum.nii.gz" -thr 0.5 -bin "${nifti_dir}BET_Output/smoothed_bin_corpus_callosum.nii.gz"

#mri_convert "${nifti_dir}BET_Output/smoothed_bin_corpus_callosum.nii.gz" "${SUBJECTS_DIR}19978_acute/mri/smoothed_cc_bin.mgz"

#mri_paint "${SUBJECTS_DIR}19978_acute/mri/filled.mgz" "${SUBJECTS_DIR}19978_acute/mri/smoothed_cc_bin.mgz" 251
#exit



#recon-all -s 19978_ultra-fast -autorecon1 -notalairach -noskullstrip
#recon-all -s 19978_ultra-fast -autorecon2-cp -noaseg -nofix
#recon-all -s 19978_ultra-fast -autorecon2 -noaseg -nofix
#recon-all -s 19978_ultra-fast -autorecon3 -noavgcurv -nocortparc -nocortparc2 -noparcstats
