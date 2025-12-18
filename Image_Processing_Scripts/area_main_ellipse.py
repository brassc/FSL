import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy as sp


# Filenames
input_filename = 'batch2_ellipse_data.pkl' # input
ellipse_area_filename = 'batch2_ellipse_area_data.csv' # output

# Import data from pickle file instead of CSV
print("Loading data from pickle file...")
data = pd.read_pickle(f'Image_Processing_Scripts/{input_filename}')
print("Original columns:")
print(data.columns)


# Create a new dataframe for ellipse data
ellipse_df = pd.DataFrame(columns=['patient_id', 'timepoint', 'side', 
                                  'ellipse_area_def', 'ellipse_area_ref', 'ellipse_area_diff'])

# check if data['ellipse_*_*ef'] is a string or numpy array
print(type(data['ellipse_h_def'].iloc[0]))

# Process each row
for i in range(len(data)):
    # Get ellipse contour data
    h_def_ellipse = data['ellipse_h_def'].iloc[i]
    v_def_ellipse = data['ellipse_v_def'].iloc[i]
    h_ref_ellipse = data['ellipse_h_ref'].iloc[i]
    v_ref_ellipse = data['ellipse_v_ref'].iloc[i]
    
    # Sort by x-coordinate (horizontal) for trapezoidal integration
    def sort_by_x(h_arr, v_arr):
        sorted_pairs = sorted(zip(h_arr, v_arr))
        h_sorted, v_sorted = map(np.array, zip(*sorted_pairs))
        return h_sorted, v_sorted
    
    # Sort data for integration
    h_def_sorted, v_def_sorted = sort_by_x(h_def_ellipse, v_def_ellipse)
    h_ref_sorted, v_ref_sorted = sort_by_x(h_ref_ellipse, v_ref_ellipse)
    
    # Calculate area using trapezoidal rule
    print(f"Starting integration for ellipse of patient {data['patient_id'].iloc[i]} at timepoint {data['timepoint'].iloc[i]}...")
    ellipse_area_def = sp.integrate.trapezoid(y=v_def_sorted, x=h_def_sorted)
    ellipse_area_ref = sp.integrate.trapezoid(y=v_ref_sorted, x=h_ref_sorted)
    
    # Calculate area difference
    ellipse_area_diff = ellipse_area_def - ellipse_area_ref
    
    print(f"Ellipse areas calculated - Deformed: {ellipse_area_def}, Reference: {ellipse_area_ref}, Difference: {ellipse_area_diff}")
    
    # Add to ellipse dataframe
    ellipse_df = pd.concat([ellipse_df, pd.DataFrame([{
        'patient_id': data['patient_id'].iloc[i],
        'timepoint': data['timepoint'].iloc[i],
        'side': data['side'].iloc[i],
        'ellipse_area_def': ellipse_area_def,
        'ellipse_area_ref': ellipse_area_ref,
        'ellipse_area_diff': ellipse_area_diff
    }])], ignore_index=True)

    
    # Visualize ellipses and areas
    plt.figure(figsize=(8, 4))
        
    # Create common x for filling between (plotting)
    common_x = np.union1d(h_def_sorted, h_ref_sorted)
    common_x = np.sort(common_x)
    
    # Interpolate y values to common x-axis
    def interpolate(x_old, y_old, x_new):
        return np.interp(x_new, x_old, y_old)
    
    y_def_interp = interpolate(h_def_sorted, v_def_sorted, common_x)
    y_ref_interp = interpolate(h_ref_sorted, v_ref_sorted, common_x)
    
    # Fill between the curves
    plt.fill_between(common_x, y_def_interp, y_ref_interp, color='orange', alpha=0.5)


    # Plot ellipse points
    plt.scatter(h_def_sorted, v_def_sorted, color='red', s=2, label='Deformed Ellipse')
    plt.scatter(h_ref_sorted, v_ref_sorted, color='blue', s=2, label='Reference Ellipse')
    
    #plt.text(0.5, 0.95, f"Ellipse area difference: {ellipse_area_diff:.2f}", 
    #         fontsize=12, transform=plt.gca().transAxes, ha='center')
    plt.ylim(bottom=None, top=60)
    plt.gca().set_aspect('equal', adjustable='datalim')
    
    plt.title(f"Ellipses for {data['patient_id'].iloc[i]} at timepoint {data['timepoint'].iloc[i]}")
    plt.legend(loc='upper right')
    
    plt.close()
    
    
    # Uncomment to save plots
    # plt.savefig(f"Image_Processing_Scripts/ellipse_plots/{data['patient_id'].iloc[i]}_{data['timepoint'].iloc[i]}.png")
    
    #plt.close()

# Print the ellipse dataframe
print("\nEllipse dataframe:")
print(ellipse_df)

# Save the ellipse dataframe
ellipse_df.to_csv(f'Image_Processing_Scripts/{ellipse_area_filename}', index=False)
print(f"\nEllipse data saved to Image_Processing_Scripts/{ellipse_area_filename}")