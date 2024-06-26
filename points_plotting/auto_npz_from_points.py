import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.integrate import quad
import csv
import pandas as pd

# User defined functions
from make_patient_dir import ensure_directory_exists
from load_nifti import load_nifti
from polynomial_plot import create_polynomial_from_csv
from polynomial_plot import fit_poly
from polynomial_plot import approx_poly
from symmetry_line import get_mirror_line
from symmetry_line import reflect_across_line
from save_variables import save_arrays_to_directory
from reorient import switch_orientation # there is also a reverse_switch_orientation function
from reorient import switch_sign
from load_np_data import load_data_readout
from translate_rotate import move
from translate_rotate import center
from automatic_boundary import auto_boundary_detect


def poi_csv(poi_csv, affine):
    poi_df=pd.read_csv(poi_csv)
    print(poi_df)
    
    ## POINTS OF INTEREST

    transformed_points = []
    for index, row in poi_df.iterrows():
        point = np.array([row[0], row[1], row[2], 1])
        transformed_point = np.linalg.inv(affine).dot(point)
        transformed_points.append(transformed_point)

    # extract x and y coordinates
    transformed_points=np.array(transformed_points)
    x_coords = transformed_points[:, 0]
    y_coords = transformed_points[:, 1]

    return x_coords, y_coords



    

def auto_npz_from_points(patient_id, patient_timepoint, nifti_file_path, slice_selected=0, scatter=False):
    # loads or creates points directory path and associated points files based on patient ID and timepoint
    directory_path = ensure_directory_exists(patient_id, patient_timepoint)

    poi_log_file_path=f"{directory_path}/points.csv"#'/home/cmb247/repos/FSL/points_plotting/points.csv'
    baseline_poi_log_file_path=f"{directory_path}/baseline_points.csv"#'/home/cmb247/repos/FSL/points_plotting/baseline_points.csv'

    poi_voxels_file_path=f"{directory_path}/points_voxel_coords.csv"#'/home/cmb247/repos/FSL/points_plotting/points_voxel_coords.csv'
    baseline_poi_voxels_file_path=f"{directory_path}/baseline_points_voxel_coords.csv"#'/home/cmb247/repos/FSL/points_plotting/baseline_points_voxel_coords.csv'

    img, save_directory = load_nifti(nifti_file_path)


    # get the affine transformation matrix 
    affine=img.affine

    # Get image data
    data = img.get_fdata()

    ## SELECT AND PLOT BASE SLICE

    if isinstance(slice_selected, np.ndarray):
        # Do something if slice_selected is an array
        print("slice_selected is a numpy array:", slice_selected)
        # Define the scanner (RAS, anatomical, imaging space) coordinates or voxel location
        scanner_coords = slice_selected 
        #voxel loc: 91 119 145

        # Inverse affine to convert RAS/anatomical coords to voxel coords
        inv_affine=np.linalg.inv(img.affine)

        # convert RAS/anatomical coords to voxel coords
        voxel_coords=inv_affine.dot(scanner_coords)[:3]

        # Extract the axial slice at the z voxel index determined from the scanner coordinates
        z_index=int(voxel_coords[2])
    else:
        # Do something else if slice_selected is not an array
        print("slice_selected is an integer:", slice_selected)
        z_index=int(slice_selected)

    
    print(z_index)
    slice_data=data[:,:, z_index]

    # plot the axial slice
    plt.imshow(slice_data.T, cmap='gray', origin='lower')

    xa_coords, ya_coords = poi_csv(poi_log_file_path, affine)
    xb_coords, yb_coords = poi_csv(baseline_poi_log_file_path, affine) #ya and yb coords should be the same

    #Find mirrorline (average of first and last points in xa and xb respectively)
    m, c, Y = get_mirror_line(yb_coords, xa_coords, xb_coords)
    yl_values = np.linspace(Y[0]+50, Y[-1]-65, 100) #extend Y fit line
    xl_values = m * yl_values + c # x values and y values for mirrorline plot

    #Reflection of baseline side
    xr_coords = reflect_across_line(m, c, xb_coords, yb_coords) # yr coords are same as ya and yb coords

    ## SAVE ARRAYS TO .npz FILES
    #save_auto_arrays_to_directory(patient_id, patient_timepoint, xa_coords, ya_coords, xb_coords, yb_coords, xr_coords)


    print(f"xa_coords:{xa_coords}")
    print(f"ya_coords:{ya_coords}")
    
    data_readout_dir = f"/home/cmb247/repos/FSL/points_plotting/data_readout/{patient_id}_{patient_timepoint}"
    print(f"saving arrays to directory {data_readout_dir}")


    
    save_arrays_to_directory(data_readout_dir, 'auto_deformed_array.npz',
                                xx_coords=xa_coords, yy_coords=ya_coords)
    save_arrays_to_directory(data_readout_dir, 'auto_baseline_array.npz', 
                                xx_coords=xb_coords, yy_coords=yb_coords)
    save_arrays_to_directory(data_readout_dir, 'auto_reflected_array.npz',
                                xx_coords=xr_coords, yy_coords=ya_coords)
    
    return 0



    
