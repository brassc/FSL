import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib



def extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected):
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

    # Define the scanner (RAS, anatomical, imaging space) coordinates or voxel location
    scanner_coords = slice_selected 
    #voxel loc: 91 119 145

    # Inverse affine to convert RAS/anatomical coords to voxel coords
    inv_affine=np.linalg.inv(img.affine)

    # convert RAS/anatomical coords to voxel coords
    voxel_coords=inv_affine.dot(scanner_coords)[:3]

    # Extract the axial slice at the z voxel index determined from the scanner coordinates
    z_index=int(voxel_coords[2])
    slice_data=data[:,:, z_index]

    # plot the axial slice
    plt.imshow(slice_data.T, cmap='gray', origin='lower')


    ## POLYNOMIAL AND MIRROR LINE FITTING FROM POI

    # Deformed side
    poly_func, x_values, y_values, xa_coords, ya_coords = create_polynomial_from_csv(poi_log_file_path, affine)

    #Baseline side
    polyb_func, xb_values, yb_values, xb_coords, yb_coords = create_polynomial_from_csv(baseline_poi_log_file_path, affine)

    #Find mirrorline (average of first and last points in xa and xb respectively)
    m, c, Y = get_mirror_line(yb_coords, xa_coords, xb_coords)
    yl_values = np.linspace(Y[0]+50, Y[-1]-65, 100) #extend Y fit line
    xl_values = m * yl_values + c # x values and y values for mirrorline plot

    #Reflection of baseline side
    xr_coords = reflect_across_line(m, c, xb_coords, yb_coords)
    polyr_func, xr_values, yr_values=fit_poly(yb_coords, xr_coords)

    # Save np arrays to to file.npz in given directory 'data_readout' using np.savez
    save_arrays_to_directory('data_readout', 'deformed_arrays.npz',
                            poly_func=poly_func, x_values=x_values, y_values=y_values, xx_coords=xa_coords, yy_coords=ya_coords)

    save_arrays_to_directory('data_readout', 'baseline_arrays.npz',
                            poly_func=polyb_func, x_values=xb_values, y_values=yb_values, xx_coords=xb_coords, yy_coords=yb_coords)

    save_arrays_to_directory('data_readout', 'reflected_baseline_arrays.npz',
                            poly_func=polyr_func, x_values=xr_values, y_values=yr_values, xx_coords=xr_coords, yy_coords=yb_coords)


    # Plot the fitted polynomial curve
    plt.plot(x_values, y_values, color='red', linewidth=0.75, label='Deformed Polynomial')
    #plt.scatter(xa_coords, ya_coords, c='red', s=4) # plot expansion points
    #plt.scatter(xb_coords, yb_coords, c='r', s=2) # plot baseline points
    plt.plot(xb_values, yb_values, color='cyan', linewidth=0.75, label='Baseline Polynomial')
    plt.plot(xl_values, yl_values, color='w', linestyle='--', linewidth=0.5, dashes =[10,5], label='Mirror') # plot mirror line
    #plt.scatter(xr_coords, yb_coords, color='blue', s=1) # plot mirrored points
    plt.plot(xr_values, yr_values, color='cyan',linewidth=0.75, label='Mirrored fit polynomial')

    # Save plot and show
    save_path=os.path.join(save_directory, 'slice_plot.png')
    print('Plot saved to '+ save_path)
    plt.savefig(save_path)
    plt.show()


    # Plot polynomial area between on scan
    # plot the axial slice
    plt.imshow(slice_data.T, cmap='gray', origin='lower')
    plt.fill_betweenx(yb_values, x_values, xr_values, color='orange', alpha=0.5)
    save_path2=os.path.join(save_directory, 'slice_plot.png')
    print('Plot saved to '+ save_path2)
    plt.savefig(save_path2)
    plt.show()


    return 0