# Image Processing Scripts #
This folder contains bet script for processing original images. This processing pipeline includes bias field correction using FSL fast, then registering each patient scan to first time point available using FLIRT. 

Pipeline:
1. FSL fast with -B option (output bias corrected images to directory OG_Scans_Bias_Corrected)
2. FSL flirt to register bias corrected images to first time point available for each patient


