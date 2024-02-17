# This python program imports .nii files to python, extracts a slice and prints it.

import nibabel as nib
import scipy as sp
import numpy as np
import pandas as pd
from PIL import Image


# Load jpg using PIL
image = Image.open('Kings_Mist.jpg')

# Display image
image.show()

"""
#convert PIL image to np array
image_array=np.array(image)

#create NIfTI image header
header = nib.Nifti1Header()

#create a NIfTI image object
nifti_img=nib.Nifti1Image(image_array, header)

#Save the NIfTI image
nib.save(nifti_img, 'image.nii.gz')

"""

