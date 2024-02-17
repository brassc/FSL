# main.py program from which functions are called

import nibabel as nib
import scipy as sp
import numpy as np
import pandas as pd
from PIL import Image

from extract_slice import extract_and_display_slice


nifti_file_path ='/home/cmb247/Desktop/Project_3/BET_Extractions/19978/T1w_time1_registered_scans/T1w_time1.T1w_verio_P00030_19978_acute_20111102_U-ID22791_registered.nii.gz'
        # Define the scanner coordinates or voxel location
scanner_coords = [2.641497, -2.877373, -12.73399]  
#voxel loc: 91 119 145

extract_and_display_slice(nifti_file_path, scanner_coords)








