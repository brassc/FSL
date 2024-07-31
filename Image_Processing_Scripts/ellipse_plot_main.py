import pandas as pd
pd.options.mode.copy_on_write = True # to avoid SettingWithCopyWarning
import numpy as np
import matplotlib.pyplot as plt
import sys

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


def transform_points(data):
    # move to origin
    data=data.copy()
    print('data type: ', type(data))
    print('side data type is: ', type(data['side'].iloc[0]))
    print('side value is: ', data['side'].iloc[0])
    if 'side' not in data:
        raise ValueError('Side column not found in data')
    print('initial data inside function', data)
    # Ensure 'h_<>ef_tr' column exists in the DataFrame
    if 'h_def_tr' not in data.columns:
        data['h_def_tr'] = pd.Series([np.array([])] * len(data['h_def']), index=data.index)
    if 'v_def_tr' not in data.columns:
        data['v_def_tr'] = pd.Series([np.array([])] * len(data['v_def']), index=data.index)
    if 'h_ref_tr' not in data.columns:
        data['h_ref_tr'] = pd.Series([np.array([])] * len(data['h_ref']), index=data.index)
    if 'v_ref_tr' not in data.columns:
        data['v_ref_tr'] = pd.Series([np.array([])] * len(data['v_ref']), index=data.index)

    print("Data columns:", data.columns)
    if (data['side'].iloc[0] == 'R'):
        #print(data['h_def'].iloc[0][-1])
        data['h_def_tr'] = data.apply(lambda row: row['h_def'] - row['h_def'][-1], axis=1)
        data['v_def_tr'] = data.apply(lambda row: row['v_def'] - row['v_def'][-1], axis=1)
        data['h_ref_tr'] = data.apply(lambda row: row['h_ref'] - row['h_ref'][-1], axis=1)
        data['v_ref_tr'] = data.apply(lambda row: row['v_ref'] - row['v_ref'][-1], axis=1)
        
    elif (data['side'].iloc[0] == 'L'):
        #print(data['h_def'].iloc[0][-2])
        print(data['h_def'].iloc[0])
        print(data['h_def'].iloc[0][-2])
        
        #data.at[0, 'h_def_tr'] = data['h_def'].iloc[0] - data['h_def'].iloc[0][-2]
        data['h_def_tr'] = data.apply(lambda row: row['h_def'] - row['h_def'][-2], axis=1)
        print(data['h_def_tr'].iloc[0])
        data['v_def_tr'] = data.apply(lambda row: row['v_def'] - row['v_def'][-2], axis=1)
        data['h_ref_tr'] = data.apply(lambda row: row['h_ref'] - row['h_ref'][-2], axis=1)
        data['v_ref_tr'] = data.apply(lambda row: row['v_ref'] - row['v_ref'][-2], axis=1)
        
    else:
        print('side data type is: *****', type(data['side']))
        raise ValueError('Side must be either "R" or "L"')
    print('inside function after data transformation', data)
    print(data.columns)
    return data

def rotate_points(data):
    # rotate points so that anterior point lies on x axis
    if (data['side'] == 'R').any():
        angle = np.arctan(data['v_def_tr'].iloc[0][-2]/data['h_def_tr'].iloc[0][-2])
        angle=angle*-1
    elif (data['side'] == 'L').any():
        angle = np.arctan(data['v_def_tr'].iloc[0][-1]/data['h_def_tr'].iloc[0][-1])
        angle=angle*-1
    else:
        raise ValueError('Side must be either "R" or "L"')

# Ensure 'h_<>ef_rot' column exists in the DataFrame
    if 'h_def_rot' not in data.columns:
        data['h_def_rot'] = pd.Series([np.array([])] * len(data['h_def']), index=data.index)
    if 'v_def_rot' not in data.columns:
        data['v_def_rot'] = pd.Series([np.array([])] * len(data['v_def']), index=data.index)
    if 'h_ref_rot' not in data.columns:
        data['h_ref_rot'] = pd.Series([np.array([])] * len(data['h_ref']), index=data.index)
    if 'v_ref_rot' not in data.columns:
        data['v_ref_rot'] = pd.Series([np.array([])] * len(data['v_ref']), index=data.index)
    
    # rotate points by this angle
     # rotation matrix for anticlockwise rotation
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)],
                            ])
    
    def_group_coords=np.vstack((data['h_def_tr'].values, data['v_def_tr'].values))
    def_rot_coords=np.dot(rotation_matrix, def_group_coords)
    #h_def_rot=def_rot_coords[0]
    #v_def_rot=def_rot_coords[1]
    # Perform the rotation and assign the results in one line using apply with a lambda function
    data['h_def_rot'] = data.apply(lambda row: np.dot(rotation_matrix, np.array([row['h_def_tr'], row['v_def_tr']]))[0], axis=1)
    data['v_def_rot'] = data.apply(lambda row: np.dot(rotation_matrix, np.array([row['h_def_tr'], row['v_def_tr']]))[1], axis=1)

    ref_group_coords=np.vstack((data['h_ref_tr'].values, data['v_ref_tr'].values))
    ref_rot_coords=np.dot(rotation_matrix, ref_group_coords)
    #h_ref_rot=ref_rot_coords[0]
    #v_ref_rot=ref_rot_coords[1]
    data['h_ref_rot'] = data.apply(lambda row: np.dot(rotation_matrix, np.array([row['h_ref_tr'], row['v_ref_tr']]))[0], axis=1)
    data['v_ref_rot'] = data.apply(lambda row: np.dot(rotation_matrix, np.array([row['h_ref_tr'], row['v_ref_tr']]))[1], axis=1)

    print("****HERE****")
    print(data.columns)
    print(data)


    print(f"Rotation angle: {angle}")

    # Assign rotated coordinates back to dataframe
    #data.at[0, 'h_def_rot'] = h_def_rot
    print("****HERE****")
    #print(type(h_def_rot))
    #print(len(h_def_rot))
    #print(h_def_rot)
    #data.at[0,'v_def_rot'] = v_def_rot
    #data.at[0,'h_ref_rot'] = h_ref_rot
    #data.at[0,'v_ref_rot'] = v_ref_rot

    return data


## MAIN SCRIPT TO PLOT ELLIPSE FORM
data = pd.read_csv('Image_Processing_Scripts/data_entries.csv')
side_data=pd.read_csv('Image_Processing_Scripts/included_patient_info.csv')
# filtered according to exclusion flag (first column)
side_data=side_data[side_data['excluded?'] == 0]
side_data = side_data.rename(columns={' side (L/R)': 'side'})
side_series = side_data['side'].reset_index(drop=True)
side_series=side_series.str.strip()
data=pd.concat([data, side_series], axis=1)

#Converting pd.Series to np for contour data
data['deformed_contour_x'] = data['deformed_contour_x'].apply(convert_to_numpy_array)
data['deformed_contour_y'] = data['deformed_contour_y'].apply(convert_to_numpy_array)
data['reflected_contour_x'] = data['reflected_contour_x'].apply(convert_to_numpy_array)
data['reflected_contour_y'] = data['reflected_contour_y'].apply(convert_to_numpy_array)
print(data.head())
print('Type of contour data in df', type(data['deformed_contour_x']))
print('Type of first element:', type(data['deformed_contour_x'].iloc[0]))


h_def=data['deformed_contour_x']
v_def=data['deformed_contour_y']
h_ref=data['reflected_contour_x']
v_ref=data['reflected_contour_y']
print('h_def:', h_def)

hv_df = pd.DataFrame({'h_def':h_def, 'v_def':v_def, 'h_ref':h_ref, 'v_ref':v_ref})
#print(hv_df.iloc[0,1])
total_df=pd.concat([data, hv_df], axis=1)
print("*********************")
print(total_df['h_def'].iloc[0])



# Transform data points such that posterior point lies on (0, 0) and anterior lies on y=0 (x axis) (for R side craniectomy) 
#   or anterior point lies on (0,0) and posterior lies on y=0 (x axis) (for L side craniectomy)
#       Recall baseline coords are at end of contour anterior, posterior (last two points in list in that order)
"""
# Get start and end points of contour
def get_start_end_points(contour):
    start = contour.iloc[-2]
    end = contour.iloc[-1]
    return start, end
"""
transformed_df = pd.DataFrame()
for i in range (len(total_df)):
    print(total_df.iloc[i])
    # drop data from total_df that isn't row i, save as data
    
    data = total_df.iloc[[i]].copy()
    data.columns=total_df.columns
    #print('side data is of type: ',type(data['side'].iloc[0]))
    
    transformed_data=transform_points(data)
    #print('transformed data:', transformed_data)
    print(transformed_data.columns)
    transformed_data=rotate_points(transformed_data)
    print(transformed_data.columns)


    # Centering about 0
    # Get smallest h_<>ef_rot value
    h_def_rot_min = transformed_data['h_def_rot'].iloc[0].min()
    #h_ref_rot_min = transformed_data['h_ref_rot'].iloc[0].min()
    # Get max h_<>ef_rot value
    h_def_rot_max = transformed_data['h_def_rot'].iloc[0].max()
    #h_ref_rot_max = transformed_data['h_ref_rot'].iloc[0].max()

    average_h = (h_def_rot_min + h_def_rot_max) / 2
    #average_h_ref_rot = (h_ref_rot_min + h_ref_rot_max) / 2

    if (transformed_data['side'] == 'R').any():
        transformed_data['h_def_rot'] = transformed_data['h_def_rot'] + average_h
        transformed_data['h_ref_rot'] = transformed_data['h_ref_rot'] + average_h
    elif (transformed_data['side'] == 'L').any():
        transformed_data['h_def_rot'] = transformed_data['h_def_rot'] - average_h
        transformed_data['h_ref_rot'] = transformed_data['h_ref_rot'] - average_h
    else:
        raise ValueError('Side must be either "R" or "L"')


    transformed_df = pd.concat([transformed_df, transformed_data], ignore_index=True)
    print(transformed_df.iloc[i])

    # plot data
    plt.scatter(transformed_df['h_def_rot'].iloc[i], transformed_df['v_def_rot'].iloc[i], color='red')
    plt.scatter(transformed_df['h_ref_rot'].iloc[i], transformed_df['v_ref_rot'].iloc[i], color='cyan')
    plt.title(f"{transformed_data['patient_id']} {transformed_data['timepoint']}")
    #plt.xlim(0)
    plt.ylim(0)
    plt.show()

    

print('*****')
print(total_df.columns)
print(transformed_df.columns)

#for i in range(len(side_data)):
    # if data has exclusion mark, pop it from the data
 #   if side_data.iloc[i, 0] == 1:
  #      side_data = side_data.drop(i)






# Find average of start and end y, position about center

# Make y values positive if necessary

# fit ellipse using least squares method

# Find change in area between two ellipses

# Plot change in area over time for each patient (x axis: time, y axis: area)

# Reverse transform data points, save to df / .csv

# Plot on image.