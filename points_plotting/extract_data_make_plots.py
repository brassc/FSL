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



def extract_data_make_plots(patient_id, patient_timepoint, nifti_file_path, slice_selected=0, scatter=False, deformed_order=2):
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


    ## POLYNOMIAL AND MIRROR LINE FITTING FROM POI

    # Deformed side
    print('deformed side equation:')
    poly_func, x_values, y_values, xa_coords, ya_coords = create_polynomial_from_csv(poi_log_file_path, affine, order=deformed_order)

    #Baseline side
    print('baseline side equation:')
    polyb_func, xb_values, yb_values, xb_coords, yb_coords = create_polynomial_from_csv(baseline_poi_log_file_path, affine, order=2)

    #Find mirrorline (average of first and last points in xa and xb respectively)
    m, c, Y = get_mirror_line(yb_coords, xa_coords, xb_coords)
    yl_values = np.linspace(Y[0]+50, Y[-1]-65, 100) #extend Y fit line
    xl_values = m * yl_values + c # x values and y values for mirrorline plot

    #Reflection of baseline side
    xr_coords = reflect_across_line(m, c, xb_coords, yb_coords)
    print('reflected baseline equation:')
    polyr_func, xr_values, yr_values=fit_poly(yb_coords, xr_coords, order=2)



    # AREA CALC
    #change orientation 
    # new x coord is vertical, v new y coord is horizontal, h
    # DEFORMED VALUES
    vd_values, hd_values, vd_coords, hd_coords = switch_orientation(x_values, y_values, xa_coords, ya_coords)
    # BASELINE VALUES
    vb_values, hb_values, vb_coords, hb_coords = switch_orientation(xb_values, yb_values, xb_coords, yb_coords)
    # REFLECTED VALUES
    vr_values, hr_values, vr_coords, hr_coords = switch_orientation(xr_values, yr_values, xr_coords, yb_coords)

    result, _ = quad(poly_func, hd_values[0], hd_values[-1])
    result_r, _ = quad(polyr_func, hd_values[0], hd_values[-1])
    area_between = np.abs(result - result_r)
    area_between=round(area_between, 2) 
    result_rounded=round(result,2) 
    result_rounded_r=round(result_r, 2)  
    print('deformed area = ', result_rounded)
    print('baseline reflection area = ', result_rounded_r)
    print('**** total deformed AREA IS ****')    
    print(area_between)

    # SAVE AREA TO FILE
    # Flag to check if patient_id and patient_timepoint have been found in the file
    current_directory = os.getcwd()
    if os.path.basename(current_directory) == 'points_plotting':
        csv_file_path = 'area_data.csv'
    else:
        print('please run from points_plotting directory in FSL repo.')
    
    # initialise found flag 
    found = False

    # Read existing contents of the CSV file
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for row in rows:
            # Check if the row contains the same patient_id and patient_timepoint
            if row[0] == patient_id and row[1] == patient_timepoint:
                # Replace this row with the new data
                row[:] = [patient_id, patient_timepoint, area_between]
                found = True
                break

    # If patient_id and patient_timepoint were not found, append a new row
    if not found:
        rows.append([patient_id, patient_timepoint, area_between])

    # Write the updated contents back to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
        


    # Save np arrays to to file.npz in given directory data_readout_dir using np.savez
    data_readout_dir=f"data_readout/{patient_id}_{patient_timepoint}"
    save_arrays_to_directory(data_readout_dir, 'deformed_arrays.npz',
                            poly_func=poly_func, x_values=x_values, y_values=y_values, xx_coords=xa_coords, yy_coords=ya_coords)

    save_arrays_to_directory(data_readout_dir, 'baseline_arrays.npz',
                            poly_func=polyb_func, x_values=xb_values, y_values=yb_values, xx_coords=xb_coords, yy_coords=yb_coords)

    save_arrays_to_directory(data_readout_dir, 'reflected_baseline_arrays.npz',
                            poly_func=polyr_func, x_values=xr_values, y_values=yr_values, xx_coords=xr_coords, yy_coords=yb_coords)


    # Plot the fitted polynomial curve
    plt.plot(x_values, y_values, color='red', linewidth=0.75, label='Deformed Polynomial')
    #plt.scatter(xa_coords, ya_coords, c='red', s=4) # plot expansion points
    #plt.scatter(xb_coords, yb_coords, c='r', s=2) # plot baseline points
    plt.plot(xb_values, yb_values, color='cyan', linewidth=0.75, label='Baseline Polynomial')
    plt.plot(xl_values, yl_values, color='w', linestyle='--', linewidth=0.5, dashes =[10,5], label='Mirror') # plot mirror line
    #plt.scatter(xr_coords, yb_coords, color='blue', s=1) # plot mirrored points
    plt.plot(xr_values, yr_values, color='cyan',linewidth=0.75, label='Mirrored fit polynomial')

    if scatter:
        # plot scatter points
        print('plotting scatter points')
        plt.scatter(xa_coords, ya_coords, c='red', s=4) # plot expansion points
        plt.scatter(xb_coords, yb_coords, c='r', s=2) # plot baseline points
        plt.scatter(xr_coords, yb_coords, color='blue', s=1) # plot mirrored points
    else: 
        print('plot without scatter points')
        
    plt.xlim(0)
    plt.ylim(0)
    # Save plot and show
    save_path=os.path.join(save_directory, f"{patient_id}_{patient_timepoint}_polynomial_slice_plot.png")
    print('Plot saved to '+ save_path)
    plt.savefig(save_path)
    plt.show()


    # Plot polynomial area between on scan
    # plot the axial slice
    plt.imshow(slice_data.T, cmap='gray', origin='lower')
    plt.fill_betweenx(yb_values, x_values, xr_values, color='orange', alpha=0.5)
    plt.text(3, 250, f'Area = {area_between} mm^2', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.xlim(0)
    plt.ylim(0)
    save_path2=os.path.join(save_directory, f"{patient_id}_{patient_timepoint}_area_slice_plot.png")
    print('Plot saved to '+ save_path2)
    plt.savefig(save_path2)
    plt.show()


    return 0

def rt_data(patient_id, patient_timepoint):
    data_readout_loc = f"points_plotting/data_readout/{patient_id}_{patient_timepoint}"
    
    polyd_func, x_values, y_values, xa_coords, ya_coords = load_data_readout(data_readout_loc, 'deformed_arrays.npz')
    polyb_func, xb_values, yb_values, xb_coords, yb_coords = load_data_readout(data_readout_loc, 'baseline_arrays.npz')
    polyr_func, xr_values, yr_values, xr_coords, yb_coords = load_data_readout(data_readout_loc, 'reflected_baseline_arrays.npz')

    #change orientation 
    # new x coord is vertical, v new y coord is horizontal, h
    # DEFORMED VALUES
    vd_values, hd_values, vd_coords, hd_coords = switch_orientation(x_values, y_values, xa_coords, ya_coords)
    # BASELINE VALUES
    vb_values, hb_values, vb_coords, hb_coords = switch_orientation(xb_values, yb_values, xb_coords, yb_coords)
    # REFLECTED VALUES
    vr_values, hr_values, vr_coords, hr_coords = switch_orientation(xr_values, yr_values, xr_coords, yb_coords)

    
    # find average (centerline), find if vd_coord < or > centerline
    avg_x = ((xa_coords + xb_coords) / 2)
   #avg_x = pd.DataFrame({'avg_x': avg_x})
    print(avg_x)
    if (xa_coords[0] <= avg_x[0]):
        vd_values, hd_values, vd_coords, hd_coords = switch_sign(vd_values, hd_values, vd_coords, hd_coords)
        vr_values, hr_values, vr_coords, hr_coords = switch_sign(vr_values, hr_values, vr_coords, hr_coords)
    else:
        print('no sign change required')

    return vd_values, hd_values, vd_coords, hd_coords, vr_values, hr_values, vr_coords, hr_coords, hb_values, polyd_func, polyr_func


def rt_data_make_plots(patient_id, patient_timepoint):

    vd_values, hd_values, vd_coords, hd_coords, vr_values, hr_values, vr_coords, hr_coords, hb_values, polyd_func, polyr_func = rt_data(patient_id, patient_timepoint)


    # CALCULATE AREA
    aread, _ = quad(polyd_func, hb_values[0], hb_values[-1])
    arear, _ = quad(polyr_func, hb_values[0], hb_values[-1])
    area_betw = np.abs(aread-arear)
    area_betw=round(area_betw, 2)
    print(area_betw)


    # MAKE NICE PLOT

    plt.plot(hr_values, vr_values, c='cyan')
    plt.scatter(hr_coords, vr_coords, s=2, color='cyan')
    plt.plot(hd_values, vd_values, c='red')
    plt.scatter(hd_coords, vd_coords, s=2, color='red')
    plt.fill_between(hd_values, vd_values, vr_values, color='orange', alpha=0.5 )
    plt.text(70, 165, f'Area = {area_betw} mm^2', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(hr_coords[0], vr_coords[0], 'hr_coords[0], vr_coords[0]')
    #plt.scatter(hr_coords[-1], vr_coords[-1])
    plt.text(hr_coords[-1], vr_coords[-1], 'hr_coords[-1], vr_coords[-1]')
    plt.show()


    # TRANSLATE and ROTATE SO THAT POINTS LIE ON HORIZONTAL AXIS
    tr_hr_coords, tr_vr_coords, _ = move(hr_coords, vr_coords)
    tr_hd_coords, tr_vd_coords, _ = move(hd_coords, vd_coords)
    tr_hr_values, tr_vr_values, polyr_func = move(hr_values, vr_values)
    tr_hd_values, tr_vd_values, polyd_func = move(hd_values, vd_values)
    # compare translating points like this with fitting a new polynomial:
    polyd2_func, tr2_vd_values, tr2_hd_values = fit_poly(tr_hd_coords, tr_vd_coords, order=2)


    # SAVE ARRAYS TO DIRECTORY
    # Save np arrays to to file.npz in given directory data_readout_dir using np.savez
    data_readout_dir=f"points_plotting/data_readout/{patient_id}_{patient_timepoint}"
    save_arrays_to_directory(data_readout_dir, 'deformed_rt_array.npz',
                                poly_func=polyd_func, x_values=tr_hd_values, 
                                    y_values=tr_vd_values, xx_coords=tr_hd_coords, yy_coords=tr_vd_coords)

        #save_arrays_to_directory(data_readout_dir, 'baseline__rt_arrays.npz', poly_func=polyb_func, x_values=xb_values, y_values=yb_values, xx_coords=xb_coords, yy_coords=yb_coords)

    save_arrays_to_directory(data_readout_dir, 'reflected_baseline_rt_array.npz',
                                poly_func=polyr_func, x_values=tr_hr_values, 
                                    y_values=tr_vr_values, xx_coords=tr_hr_coords, yy_coords=tr_vr_coords)


    # PLOT TRANSLATED AND ROTATED FUNCTION
    plt.scatter(tr_hr_coords, tr_vr_coords, c='cyan', s=2)
    plt.scatter(tr_hd_coords, tr_vd_coords, c='red', s=2)
    plt.plot(tr_hd_values, tr_vd_values, c='red')
    plt.plot(tr_hr_values, tr_vr_values, c='cyan')
    #plt.plot(tr2_hd_values, tr2_vd_values, c='blue')
    plt.xlim(0)
    plt.ylim(0)

    # Calculate the range of values along each axis
    x_range = np.max(hd_values) - np.min(hd_values)
    y_range = np.max(vd_values)+100 - np.min(vd_values)-100

    # Calculate the aspect ratio
    aspect_ratio = 20/9
    # Set aspect ratio
    plt.gca().set_aspect(0.5)

    plt.savefig(f"points_plotting/plots_no_patient_data/{patient_id}_{patient_timepoint}_rt_plot.png")
    plt.show()
        
    
    
    return 0

def rt_data_poly_plot(patient_id, patient_timepoint):

    _,_, vd_coords, hd_coords, _, _, vr_coords, hr_coords, hb_values, _, _ = rt_data(patient_id, patient_timepoint)

    poly_func, vd_values, hd_values = fit_poly(hd_coords, vd_coords, order=2)

    plt.plot(hd_values, vd_values, color='red')
    plt.show()

    return hd_values, vd_values


def test_fun(patient_id, patient_timepoint):
    data_readout_loc = f"points_plotting/data_readout/{patient_id}_{patient_timepoint}"
    
    _, _, _, xa_coords, ya_coords = load_data_readout(data_readout_loc, 'deformed_arrays.npz')
    _, _, _, xb_coords, yb_coords = load_data_readout(data_readout_loc, 'baseline_arrays.npz')
    _, _, _, xr_coords, yb_coords = load_data_readout(data_readout_loc, 'reflected_baseline_arrays.npz')

    tr_h_coords, tr_v_coords, _ = move(ya_coords, xa_coords, poly=0)
    print(tr_h_coords, tr_v_coords)
    #Centered coords:
    ctr_h_coords, c_val = center(tr_h_coords)
    h_values, v_fitted, h_optimal, a_optimal, b_optimal = approx_poly(ctr_h_coords, tr_v_coords)

    # Plot the original data points and the fitted curve
    #plt.scatter(ya_coords, xa_coords, label='Original data points', color='r', s=2)
    plt.scatter(ctr_h_coords, tr_v_coords, label='translated and rotated data points', color='r', s=2)
    plt.plot(h_values, v_fitted, label='Fitted curve', color='red')
    plt.scatter(ctr_h_coords[0], tr_v_coords[0], c='black')
    plt.scatter(ctr_h_coords[-1], tr_v_coords[-1], c='orange', label='h[-1], v[-1]')
    plt.xlabel('y')
    plt.ylabel('x')

    
    #plt.xlim(0)
    legend_text = f'h_optimal: {h_optimal:.2f}\na_optimal: {a_optimal:.2f}\nb_optimal: {b_optimal:.2f}'
    plt.text(0.95, 1.02, legend_text, transform=plt.gca().transAxes)
    plt.ylim(0)
    plt.title('Fitting function of form y=h sqrt[a^2-x^2]')
    plt.legend()
    plt.grid(True)
    plt.show()

    return 0
