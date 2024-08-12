import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy as sp

from ellipse_plot_main import convert_to_numpy_array


"""
def convert_to_numpy_array(s):
    # Remove extra whitespace and split by spaces
    # Convert the resulting list of strings to a list of integers
    s=s.strip('[]')

    def convert_value(value):
        try:
            # Attempt to convert to int
            return int(value)
        except ValueError:
            # If it fails, convert to float
            return float(value)
    
    return np.array([convert_value(value) for value in s.split()])
"""

# import data
data=pd.read_csv('Image_Processing_Scripts/ellipse_data.csv')
print(data.columns)
# drop rows 'angle', 'ellipse_h_def','ellipse_v_def', 'h_param_def', 'a_param_def', 'ellipse_h_ref','ellipse_v_ref', 'h_param_ref', 'a_param_ref'
data=data.drop(columns=['angle', 'h_def_cent', 'v_def_cent',
       'h_ref_cent', 'v_ref_cent', 'ellipse_h_def','ellipse_v_def', 'h_param_def', 'a_param_def', 'ellipse_h_ref','ellipse_v_ref', 'h_param_ref', 'a_param_ref'])
print(data.columns)

# convert columns to numpy arrays
data['h_def_rot'] = data['h_def_rot'].apply(convert_to_numpy_array)
data['v_def_rot'] = data['v_def_rot'].apply(convert_to_numpy_array)
data['h_ref_rot'] = data['h_ref_rot'].apply(convert_to_numpy_array)
data['v_ref_rot'] = data['v_ref_rot'].apply(convert_to_numpy_array)

# Create new column to store area of deformed contour
if 'area_def' not in data.columns:
    data['area_def'] = pd.Series([np.array([])] * len(data['h_def_rot']), index=data.index)
print(data.columns)

# Make h values absolute
data['h_def_rot'] = data['h_def_rot'].apply(np.abs)
data['v_def_rot'] = data['v_def_rot'].apply(np.abs)
print(data['h_def_rot'])
print(data['v_def_rot'])

# sort the data wrt to 'h_def_rot' array line by line to prepare for integration
for i in range(len(data)):
    # Zip the arrays together and sort by h_def_rot
    sorted_pairs = sorted(zip(data.loc[i, 'h_def_rot'], data.loc[i, 'v_def_rot']))
    # Unzip the sorted pairs back into h_def_rot and v_def_rot
    h_sorted, v_sorted = map(np.array, zip(*sorted_pairs))
    # Assign the sorted arrays back to the DataFrame
    data.at[i, 'h_def_rot'] = h_sorted
    data.at[i, 'v_def_rot'] = v_sorted

print(data['h_def_rot'])
print(data['v_def_rot'])
# integrate wrt y line by line
print("Starting integration...")
for i in range(0, len(data['area_def'])):#range(5,6):#
    #print(data.loc[i, 'h_def_rot'])
    #print(data.loc[i, 'v_def_rot'])
    
    data.loc[i, 'area_def'] = sp.integrate.trapz(y=data.loc[i, 'v_def_rot'], x=data.loc[i, 'h_def_rot']) 
    
    plt.scatter(data.loc[i, 'h_def_rot'], data.loc[i, 'v_def_rot'], color='red')
    plt.fill_between(data.loc[i, 'h_def_rot'], data.loc[i, 'v_def_rot'], color='orange', alpha=0.5)
    plt.title(f"Deformed contour for {data.loc[i, 'patient_id']} at timepoint {data.loc[i, 'timepoint']}")
    plt.text(0.5, 0.5, f"Area: {data.loc[i, 'area_def']:.2f}", fontsize=12, transform=plt.gca().transAxes)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.figsize=(12, 8)

    plt.show()

print("Integration complete.")
print(data['area_def'])