import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy as sp

from ellipse_plot_main import convert_to_numpy_array


# import data
data=pd.read_csv('Image_Processing_Scripts/ellipse_data.csv')
print(data.columns)
# drop rows 'angle', 'ellipse_h_def','ellipse_v_def', 'h_param_def', 'a_param_def', 'ellipse_h_ref','ellipse_v_ref', 'h_param_ref', 'a_param_ref'
data=data.drop(columns=['angle', 'h_def_cent', 'v_def_cent',
       'h_ref_cent', 'v_ref_cent', 'ellipse_h_def','ellipse_v_def', 'h_param_def', 'a_param_def', 'ellipse_h_ref','ellipse_v_ref', 'h_param_ref', 'a_param_ref'])
print(data.columns)

new_df = pd.DataFrame(columns=['patient_id', 'timepoint', 'side', 'area_def', 'area_ref', 'area_diff'])

# convert _tr columns to numpy arrays 
data['h_def_tr'] = data['h_def_tr'].apply(convert_to_numpy_array)
data['v_def_tr'] = data['v_def_tr'].apply(convert_to_numpy_array)
data['h_ref_tr'] = data['h_ref_tr'].apply(convert_to_numpy_array)
data['v_ref_tr'] = data['v_ref_tr'].apply(convert_to_numpy_array)

def initialize_columns(data, name):
    h_col = f'h_oriented_{name}'
    v_col = f'v_oriented_{name}'
    area_col = f'area_{name}'
    common_x_col = f'common_x_{name}'
    if h_col not in data.columns:
        data[h_col] = pd.Series([np.array([])] * len(data[f'h_{name}_rot']), index=data.index)
    if v_col not in data.columns:
        data[v_col] = pd.Series([np.array([])] * len(data[f'v_{name}_rot']), index=data.index)
    if area_col not in data.columns:
        data[area_col] = np.float16()
    return h_col, v_col, area_col



for i in range (0, len(data)):
    plt.scatter(data['h_def_tr'].iloc[i], data['v_def_tr'].iloc[i], color='red', s=2)
    plt.scatter(data['h_ref_tr'].iloc[i], data['v_ref_tr'].iloc[i], color='cyan', s=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Contour for {data['patient_id'].iloc[i]} at timepoint {data['timepoint'].iloc[i]}")
    plt.close()
   

    for name in ['def', 'ref']:
        h_col, v_col, area_col = initialize_columns(data, name)

        #for i in range(0, len(data)):
        horizontal=data[f'v_{name}_tr'].iloc[i]
        vertical=data[f'h_{name}_tr'].iloc[i]
        if data['side'].iloc[i] == 'L':
            # flip data
            horizontal = -horizontal
            horizontal = horizontal + np.abs(np.min(horizontal))
        elif data['side'].iloc[i] == 'R':
            vertical = -vertical
        data.at[i, h_col] = horizontal
        data.at[i, v_col] = vertical
        #print(data.columns)

        ## sort the data wrt to 'h_def_rot' array line by line to prepare for integration
        #for i in range(len(data)):
        # Zip the arrays together and sort by h_def_rot
        sorted_pairs = sorted(zip(data.loc[i, h_col], data.loc[i, v_col]))
        # Unzip the sorted pairs back into h_def_rot and v_def_rot
        h_sorted, v_sorted = map(np.array, zip(*sorted_pairs))
        # Assign the sorted arrays back to the DataFrame
        data.at[i, h_col] = h_sorted
        data.at[i, v_col] = v_sorted
    
        


        # Get area underneath, store in area column
        print(f"Starting integration for patient {data['patient_id'].iloc[i]} at timepoint {data['timepoint'].iloc[i]} {name} configuration...")
        #for i in range(0, len(data)):#range(5,6):#
        data.loc[i, f'area_{name}'] = sp.integrate.trapezoid(y=data.loc[i, v_col], x=data.loc[i, h_col]) 
        print(f"area calculated for {name}: {data.loc[i, f'area_{name}']}")
    """
    def flip_and_shift(horizontal, vertical, side):
        if side == 'L':
            # flip data
            horizontal = -horizontal
            horizontal = horizontal + np.abs(np.min(horizontal))
        elif side == 'R':
            vertical = -vertical
        return horizontal, vertical
    """
    # Create and fill area difference column
    if 'area_diff' not in data.columns:
        data['area_diff'] = np.float16()
    data.loc[i, 'area_diff'] = data.loc[i, 'area_def'] - data.loc[i, 'area_ref']

    #interpolate for common x for fill between
    # Create common x for filling between (plotting)
    common_x=np.union1d(data['h_oriented_def'].iloc[i], data['h_oriented_ref'].iloc[i])
    common_x=np.sort(common_x)
    if 'common_x' not in data.columns:
        data[f'common_x'] = pd.Series([np.array([])] * len(data), index=data.index)
    data.at[i, f'common_x'] = common_x
    
    # Interpolate y values to common x-axis
    def interpolate(x_old, y_old, x_new):
        return np.interp(x_new, x_old, y_old)
    y_def_interp = interpolate(data['h_oriented_def'].iloc[i], data['v_oriented_def'].iloc[i], data['common_x'].iloc[i])
    y_ref_interp = interpolate(data['h_oriented_ref'].iloc[i], data['v_oriented_ref'].iloc[i], data['common_x'].iloc[i])
    print("length of y_def_interp: ", len(y_def_interp))
    print("length of y_ref_interp: ", len(y_ref_interp))
    print("length of common_x: ", len(data['common_x'].iloc[i]))
    
        
        

    #horizontal, vertical = flip_and_shift(data['v_def_tr'].iloc[i], data['h_def_tr'].iloc[i], data['side'].iloc[i])
    plt.scatter(data['h_oriented_def'].iloc[i], data['v_oriented_def'].iloc[i], color='red', s=2)
    plt.scatter(data['h_oriented_ref'].iloc[i], data['v_oriented_ref'].iloc[i], color='cyan', s=2)
    plt.fill_between(data['common_x'].iloc[i], y_def_interp, y_ref_interp, color='orange', alpha=0.5)
    plt.text(0.5, 0.5, f"Difference in area: {(data['area_def'].iloc[i]-data['area_ref'].iloc[i]):.2f}", fontsize=12, transform=plt.gca().transAxes)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Contour for {data['patient_id'].iloc[i]} at timepoint {data['timepoint'].iloc[i]}")
    plt.close()

    # add to new_df
    new_df = new_df._append({'patient_id': data['patient_id'].iloc[i], 'timepoint': data['timepoint'].iloc[i], 'side': data['side'].iloc[i], 'area_def': data['area_def'].iloc[i], 'area_ref': data['area_ref'].iloc[i], 'area_diff': data['area_diff'].iloc[i]}, ignore_index=True)

print(new_df)
new_df.to_csv('Image_Processing_Scripts/area_data.csv', index=False)
sys.exit()







plt.scatter(data['h_def_tr'].iloc[0], data['v_def_tr'].iloc[0], color='red', s=2)
plt.show()
horizontal=data['v_def_tr'].iloc[0]
vertical=data['h_def_tr'].iloc[0]
if data['side'].iloc[0] == 'L':
    # flip data
    horizontal = -horizontal
    # if minimum point in horizontal < 0, add the minimum point to all values in horizontal
    if np.min(horizontal) < 0:
        horizontal = horizontal + np.abs(np.min(horizontal))

plt.scatter(horizontal, vertical, color='red', s=2)
plt.show()


sys.exit()
# convert columns to numpy arrays
data['h_def_rot'] = data['h_def_rot'].apply(convert_to_numpy_array)
data['v_def_rot'] = data['v_def_rot'].apply(convert_to_numpy_array)
data['h_ref_rot'] = data['h_ref_rot'].apply(convert_to_numpy_array)
data['v_ref_rot'] = data['v_ref_rot'].apply(convert_to_numpy_array)

# Create new column to store area of deformed contour
if 'area_def' not in data.columns:
    data['area_def'] = pd.Series([np.array([])] * len(data['h_def_rot']), index=data.index)
print(data.columns)


if 'h_def_rot_sorted' not in data.columns:
    data['h_def_rot_sorted'] = pd.Series([np.array([])] * len(data['h_def_rot']), index=data.index)
if 'v_def_rot_sorted' not in data.columns:
    data['v_def_rot_sorted'] = pd.Series([np.array([])] * len(data['v_def_rot']), index=data.index)

# Make h values greater than 0
if data['side'].iloc[0] == 'L':
    # Get second to last valu (anterior coordinate) of array stored at data['h_def_rot'].iloc[0]
    anterior_coord = data['h_def_rot'].iloc[0][-2]
    # If the anterior coordinate is greater than 0, then the values are already in the correct orientation
    if anterior_coord > 0:
        print("Values are already in the correct orientation.")
    else:
        # add anterior coordinate to all values in h_def_rot array
        data['h_def_rot_sorted'] = data['h_def_rot'].apply(lambda x: x + np.abs(anterior_coord))
        print(f"h_def_rot_sorted: {data['h_def_rot_sorted'].iloc[0]}")

elif data['side'].iloc[0] == 'R':
    print("All positive values will already get a positive area.")
else:
    print("Invalid side value. Must be 'L' or 'R'.")
    sys.exit()



# sort the data wrt to 'h_def_rot' array line by line to prepare for integration
for i in range(len(data)):
    # Zip the arrays together and sort by h_def_rot
    sorted_pairs = sorted(zip(data.loc[i, 'h_def_rot_sorted'], data.loc[i, 'v_def_rot']))
    # Unzip the sorted pairs back into h_def_rot and v_def_rot
    h_sorted, v_sorted = map(np.array, zip(*sorted_pairs))
    # Assign the sorted arrays back to the DataFrame
    data.at[i, 'h_def_rot_sorted'] = h_sorted
    data.at[i, 'v_def_rot_sorted'] = v_sorted

print(data['h_def_rot'])
print(data['v_def_rot'])
# integrate wrt y line by line
print("Starting integration...")
for i in range(0, len(data['area_def'])):#range(5,6):#
    
    data.loc[i, 'area_def'] = sp.integrate.trapz(y=data.loc[i, 'v_def_rot_sorted'], x=data.loc[i, 'h_def_rot_sorted']) 
    
    plt.scatter(data.loc[i, 'h_def_rot_sorted'], data.loc[i, 'v_def_rot_sorted'], color='red')
    plt.fill_between(data.loc[i, 'h_def_rot_sorted'], data.loc[i, 'v_def_rot_sorted'], color='orange', alpha=0.5)
    plt.title(f"Deformed contour for {data.loc[i, 'patient_id']} at timepoint {data.loc[i, 'timepoint']}")
    plt.text(0.5, 0.5, f"Area: {data.loc[i, 'area_def']:.2f}", fontsize=12, transform=plt.gca().transAxes)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.figsize=(12, 8)

    plt.show()

print("Integration complete.")
print(data['area_def'])