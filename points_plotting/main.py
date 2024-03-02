# main.py program from which functions are called. References medical data and therefore must be run from where this data is stored/accessed. 

# libraries

import numpy as np
import os
#import matplotlib.pyplot as plt
#import nibabel as nib

#user defined functions
from extract_data_make_plots import extract_data_make_plots
from automatic_boundary import auto_boundary_detect

# Patient info
patient_id='19978'

patient_timepoint='acute'
nifti_file_path ='/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_acute_20111102_U-ID22791_registered.nii.gz'
bet_mask_file_path="/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/acute_restored_bet_mask-f0.5-R.nii.gz"
slice_selected=np.array([2.641497, -2.877373, -12.73399,1]) # Scanner coordinates
#voxel loc: 91 119 145
extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=False, deformed_order=2)
auto_boundary_detect(patient_id, patient_timepoint, bet_mask_file_path, x_offset=120, array_save_name='deformed_boundary.npz')
auto_boundary_detect(patient_id, patient_timepoint, bet_mask_file_path, x_offset=50, array_save_name='baseline_boundary.npz')

patient_timepoint='fast'
nifti_file_path ='/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_fast_20111027_U-ID22723_registered.nii.gz'
slice_selected=np.array([2.641497, -2.877373, -12.73399,1]) # Scanner coordinates
#voxel loc: 91 119 145

extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=True, deformed_order=4)
#auto_boundary_detect(patient_id, patient_timepoint, bet_mask_file_path, x_offset=120, array_save_name='deformed_boundary.npz')
#auto_boundary_detect(patient_id, patient_timepoint, bet_mask_file_path, x_offset=50, array_save_name='baseline_boundary.npz')

patient_id='19344'

patient_timepoint='acute'
nifti_file_path = '/home/cmb247/Desktop/Project_3/BET_Extractions/19344/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00244_19344_acute_20110323_U-ID21262_registered.nii.gz'
slice_selected=np.array([29.85494, -76.71205, 22.77727]) # Scanner coordinates

extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=False, deformed_order=2)

patient_timepoint='fast'
nifti_file_path='/home/cmb247/Desktop/Project_3/BET_Extractions/19344/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00244_19344_fast_20110308_U-ID21134_registered.nii.gz'

#extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=False, deformed_order=2)














