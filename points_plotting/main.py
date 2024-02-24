# main.py program from which functions are called. References medical data and therefore must be run from where this data is stored/accessed. 

# libraries
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

#user defined functions
from load_nifti import load_nifti
from polynomial_plot import create_polynomial_from_csv
from polynomial_plot import fit_poly
from symmetry_line import get_mirror_line
from symmetry_line import reflect_across_line
from save_variables import save_arrays_to_directory
#from extract_slice import extract_and_display_slice
from make_patient_dir import ensure_directory_exists
from extract_data_make_plots import extract_data_make_plots

# Patient info
patient_id='19978'
patient_timepoint='acute'
nifti_file_path ='/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_acute_20111102_U-ID22791_registered.nii.gz'
slice_selected=np.array([2.641497, -2.877373, -12.73399,1]) # Scanner coordinates
#voxel loc: 91 119 145

extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected)














