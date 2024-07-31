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
    #if (data['side'].iloc[0] == 'R'):
    #put posterior point to (0,0)
        #print(data['h_def'].iloc[0][-1])
    data['h_def_tr'] = data.apply(lambda row: row['h_def'] - row['h_def'][-1], axis=1)
    data['v_def_tr'] = data.apply(lambda row: row['v_def'] - row['v_def'][-1], axis=1)
    data['h_ref_tr'] = data.apply(lambda row: row['h_ref'] - row['h_ref'][-1], axis=1)
    data['v_ref_tr'] = data.apply(lambda row: row['v_ref'] - row['v_ref'][-1], axis=1)
    """    
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
    """
    #print('inside function after data transformation', data)
    #print(data.columns)
    return data

def rotate_points(data):
    # Ensure 'h_<>ef_rot' column exists in the DataFrame
    if 'h_def_rot' not in data.columns:
        data['h_def_rot'] = pd.Series([np.array([])] * len(data['h_def']), index=data.index)
    if 'v_def_rot' not in data.columns:
        data['v_def_rot'] = pd.Series([np.array([])] * len(data['v_def']), index=data.index)
    if 'h_ref_rot' not in data.columns:
        data['h_ref_rot'] = pd.Series([np.array([])] * len(data['h_ref']), index=data.index)
    if 'v_ref_rot' not in data.columns:
        data['v_ref_rot'] = pd.Series([np.array([])] * len(data['v_ref']), index=data.index)

    # Ensure 'angle' column exists in the DataFrame
    if 'angle' not in data.columns:
        data['angle'] = pd.Float32Dtype()   # Create a new column with float32 data type
    
    # rotate points so that anterior point lies on x axis
    angle = np.arctan(data['v_def_tr'].iloc[0][-2]/data['h_def_tr'].iloc[0][-2])
    
    if (data['side'] == 'L').any():
        #angle=angle*-1
        print('preprocessed angle:', angle)
        if angle < 0:
            angle=(2*np.pi)-angle
        else:
            angle=(np.pi)-angle
       
    elif(data['side'] == 'R').any():
        print('preprocessed angle:', angle)
        if angle < 0:
            angle=(np.pi)-angle
        else:
            angle=(2*np.pi)-angle
    else:
        raise ValueError('Side must be either "R" or "L"')

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
    
    # Assign angle to DataFrame
    data['angle'] = angle
    #print(f"coordinates: \n anterior: ({data['h_def_rot'].iloc[0][-2]}, {data['v_def_rot'].iloc[0][-2]} \n posterior: ({data['h_def_rot'].iloc[0][-1]}, {data['v_def_rot'].iloc[0][-1]})")
    print(f"Rotation angle: {angle}")
    return data

def center_points(data):
    # Centering about 0
    h_def_rot_min = data['h_def_rot'].iloc[0].min()  # Get smallest h_<>ef_rot value
    h_def_rot_max = data['h_def_rot'].iloc[0].max()  # Get max h_<>ef_rot value

    average_h = (h_def_rot_min + h_def_rot_max) / 2 # only one averager required - translate both h_def and h_ref by same amount
    
    if (data['side'] == 'R').any():
        data['h_def_rot'] = data['h_def_rot'] - average_h
        data['h_ref_rot'] = data['h_ref_rot'] - average_h
    elif (transformed_data['side'] == 'L').any():
        data['h_def_rot'] = data['h_def_rot'] - average_h
        data['h_ref_rot'] = data['h_ref_rot'] - average_h
    else:
        raise ValueError('Side must be either "R" or "L"')

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
#print(data.head())
#print('Type of contour data in df', type(data['deformed_contour_x']))
#print('Type of first element:', type(data['deformed_contour_x'].iloc[0]))

# Create new variables as copy of original contour data
h_def=data['deformed_contour_x']
v_def=data['deformed_contour_y']
h_ref=data['reflected_contour_x']
v_ref=data['reflected_contour_y']

# Create new data frame from these variables to add to original data frame
hv_df = pd.DataFrame({'h_def':h_def, 'v_def':v_def, 'h_ref':h_ref, 'v_ref':v_ref})
# Add data frames together
total_df=pd.concat([data, hv_df], axis=1)
#print(total_df['h_def'].iloc[0])



# Transform data points such that posterior point lies on (0, 0) and anterior lies on y=0 (x axis) (for R side craniectomy) 
#   or anterior point lies on (0,0) and posterior lies on y=0 (x axis) (for L side craniectomy)
#       Recall baseline coords are at end of contour anterior, posterior (last two points in list in that order)


# Initialise data frame to add to
transformed_df = pd.DataFrame()

# Loop through each row in the total_df
for i in range (len(total_df)):
    #print(total_df.iloc[i])
    
    # get copy of slice of total_df line by line
    data = total_df.iloc[[i]].copy()
    print(data)
    data.columns=total_df.columns
    
    #Plot original data
    plt.scatter(data['h_def'].iloc[0], data['v_def'].iloc[0], color='red', s=1)
    plt.scatter(data['h_ref'].iloc[0], data['v_ref'].iloc[0], color='cyan', s=1)
    plt.scatter(data['h_def'].iloc[0][-2], data['v_def'].iloc[0][-2], color='magenta', s=20) # anterior point
    plt.title(f"{data['patient_id'].iloc[0]} {data['timepoint'].iloc[0]}")
    # Set the aspect ratio of the plot to be equal
    plt.gca().set_aspect('equal', adjustable='box')
    plt.close()

    transformed_data=transform_points(data) # Translate function, puts in <>_<>ef_tr columns

    
    # Plot transformed data
    anterior_pt_h=transformed_data['h_ref_tr'].iloc[0][-2]
    anterior_pt_v=transformed_data['v_ref_tr'].iloc[0][-2]
    print('Anterior point:', anterior_pt_h)
    posterior_pt_h=transformed_data['h_def_tr'].iloc[0][-1]
    print('Posterior point:', posterior_pt_h)

    plt.scatter(transformed_data['h_def_tr'].iloc[0], transformed_data['v_def_tr'].iloc[0], color='red', s=1)
    plt.scatter(transformed_data['h_def_tr'].iloc[0][-2], transformed_data['v_def_tr'].iloc[0][-2], color='magenta', s=20) # anterior point
    plt.scatter(transformed_data['h_ref_tr'].iloc[0], transformed_data['v_ref_tr'].iloc[0], color='cyan', s=1)
    plt.title(f"{transformed_data['patient_id'].iloc[0]} {transformed_data['timepoint'].iloc[0]}")
    # Set the aspect ratio of the plot to be equal
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.xlim(0)
    #plt.ylim(0)
    plt.close()

   
    transformed_data=rotate_points(transformed_data) # Rotate function
    


    # plot rotated data
    plt.scatter(transformed_data['h_def_rot'].iloc[0], transformed_data['v_def_rot'].iloc[0], color='red', s=1)
    plt.scatter(transformed_data['h_def_rot'].iloc[0][-2], transformed_data['v_def_rot'].iloc[0][-2], color='magenta', s=20) # anterior point
    plt.scatter(transformed_data['h_ref_rot'].iloc[0], transformed_data['v_ref_rot'].iloc[0], color='cyan', s=1)
    plt.title(f"{transformed_data['patient_id']} {transformed_data['timepoint']}")
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.xlim(0)
    #plt.ylim(0)
    plt.close()

    
    transformed_data=center_points(transformed_data) # Center function (parks data back into rotated column)

    

    # plot data
    plt.scatter(transformed_data['h_def_rot'].iloc[0], transformed_data['v_def_rot'].iloc[0], color='red', s=1)
    plt.scatter(transformed_data['h_ref_rot'].iloc[0], transformed_data['v_ref_rot'].iloc[0], color='cyan', s=1)
    plt.scatter(transformed_data['h_def_rot'].iloc[0][-2], transformed_data['v_def_rot'].iloc[0][-2], color='magenta', s=20)
    plt.scatter(transformed_data['h_def_rot'].iloc[0][-1], transformed_data['v_def_rot'].iloc[0][-1], color='green', s=20)
    plt.title(f"{transformed_data['patient_id']} {transformed_data['timepoint']}")
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.xlim(0)
    #plt.ylim(0)
    plt.close()

    # Fit ellipse using least squares method - store data / parameters line by line

    # Find change in area between two ellipses

    # Store data as one big df
    transformed_df = pd.concat([transformed_df, transformed_data], ignore_index=True)
    #print(transformed_df.iloc[i])

    

print('*****')
#print(total_df.columns)
#print(transformed_df.columns)




# Plot change in area over time for each patient (x axis: time, y axis: area)

# Reverse transform data points, save to df / .csv

# Plot on image.