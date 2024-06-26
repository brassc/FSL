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
auto_boundary_detect(patient_id, patient_timepoint, bet_mask_file_path, x_offset=120, array_save_name='deformed_arrays.npz') #overwrite manual .npz deformed array
#auto_boundary_detect(patient_id, patient_timepoint, bet_mask_file_path, x_offset=50, array_save_name='baseline_arrays.npz')
#extract data again using auto extracted points
extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=False, deformed_order=2)

## x_offset VARIABLE SETS X DIMENSION FOR ZOOMED IN PLOT WINDOW
"""
patient_timepoint='fast'
nifti_file_path ='/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_fast_20111027_U-ID22723_registered.nii.gz'
#slice_selected=np.array([2.641497, -2.877373, -12.73399,1]) # Scanner coordinates
#voxel loc: 91 119 145
slice_selected=int(145)

extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=True, deformed_order=4)
#auto_boundary_detect(patient_id, patient_timepoint, bet_mask_file_path, x_offset=120, array_save_name='deformed_boundary.npz')
#auto_boundary_detect(patient_id, patient_timepoint, bet_mask_file_path, x_offset=50, array_save_name='baseline_boundary.npz')

patient_id='19344'

patient_timepoint='acute'
nifti_file_path = '/home/cmb247/Desktop/Project_3/BET_Extractions/19344/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00244_19344_acute_20110323_U-ID21262_registered.nii.gz'
slice_selected=np.array([29.85494, -76.71205, 22.77727, 1]) # Scanner coordinates

extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=True, deformed_order=4)

patient_timepoint='fast'
nifti_file_path='/home/cmb247/Desktop/Project_3/BET_Extractions/19344/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00244_19344_fast_20110308_U-ID21134_registered.nii.gz'
slice_selected=int(155)

extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=True, deformed_order=4)

patient_id='22725'
patient_timepoint='acute'
nifti_file_path=f"/home/cmb247/Desktop/Project_3/BET_Extractions/{patient_id}/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00370_22725_acute_20131118_U-ID29681_registered.nii.gz"
slice_selected=int(169)

extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=False, deformed_order=2)


patient_id='13990'
patient_timepoint='acute'
nifti_file_path='/home/cmb247/Desktop/Project_3/BET_Extractions/13990/T1w_time1_registered_scans/T1w_time1.T1w_trio_P00030_13990_acute_20080226_U-ID14047_registered.nii.gz'
slice_selected=int(170)

extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected, scatter=False, deformed_order=2)


"""










