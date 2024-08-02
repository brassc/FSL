import pandas as pd
pd.options.mode.copy_on_write = True # to avoid SettingWithCopyWarning
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.patches import Ellipse
from scipy.linalg import eig
from scipy.optimize import curve_fit

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

def find_intersection_height(h_coords, v_coords):
    """
    Find the height at which a linear interpolation between two h_coords either side of the y axis cuts the y axis.

    Parameters:
        h_coords (array-like): Horizontal coordinates.
        v_coords (array-like): Vertical coordinates.

    Returns:
        intersection_height (float): Height at which the linear interpolation intersects the y-axis.
    """
    # Find indices of the points closest to the y-axis
    left_index = np.abs(h_coords).argmin()
    right_index = len(h_coords) - np.abs(h_coords[::-1]).argmin() - 2
    print(f"Left index: {left_index}\nRight index: {right_index}")

    # Perform linear interpolation between the points
    slope = (v_coords[right_index] - v_coords[left_index]) / (h_coords[right_index] - h_coords[left_index])
    intersection_height = v_coords[left_index] - slope * h_coords[left_index]

    return intersection_height

def funcb(x, h, a, b, c=0, d=0):
    # To ensure we only deal with the upper portion, we return NaN if the inside of the sqrt becomes negative
    with np.errstate(invalid='ignore'):
        y = h * np.sqrt(np.maximum(0, a**2 - (x-c)**2))*(1+(b/a)*x)+d
    return y

# Define the function that represents the upper portion of an ellipse
def func(x, h, a, c=0, d=0):
    # To ensure we only deal with the upper portion, we return NaN if the inside of the sqrt becomes negative
    with np.errstate(invalid='ignore'):
        y = h * np.sqrt(np.maximum(0, a**2 - (x - c)**2)) + d
    return y

def get_fit_params(data, name='<>ef_rot'): # name e.g. 'def_rot'
    h_name = 'h_' + name
    v_name = 'v_' + name
    # Define the weights           
    weights = np.ones_like(data[h_name].iloc[0])
    print(f"weights: {weights}")
    
    # BOUNDS FOR A
    lower_bound_a = np.abs(data[h_name].iloc[0][-1])
    upper_bound_a = np.abs(data[h_name].iloc[0][-2]*2 + 20)
    print(f"Lower bound a: {lower_bound_a} \nUpper bound a: {upper_bound_a}")

    # BOUNDS FOR H
    intersection_height = find_intersection_height(data[h_name].iloc[0], data[v_name].iloc[0])
    lower_bound_h = intersection_height / lower_bound_a
    print(f"Intersection height: {lower_bound_h}")

    # BOUNDS FOR B
    if data['side'].any()== 'L':
        lower_bound_b = -np.inf
        upper_bound_b = -0.2
        b=upper_bound_b
    else:
        lower_bound_b = 0
        upper_bound_b = np.inf
        b=lower_bound_b
        
    print(f"Side: {data['side'].iloc[0]}\nLower bound b: {lower_bound_b}\nUpper bound b: {upper_bound_b}")



    desired_width = np.abs(data[h_name].iloc[0][-1] - data[h_name].iloc[0][-2])  # Desired width for the function
    print(f"Desired width: {desired_width}")

    h = data[v_name].iloc[0].max()
        #a = h_coords.max() - h_coords.min()
    a = upper_bound_a #np.abs(transformed_data['h_def_rot'].iloc[0][-2] - transformed_data['h_def_rot'].iloc[0][-1])
    c = a / 2 # middle value

    lower_bounds = [lower_bound_h, lower_bound_a]#, -np.inf, -np.inf]
    upper_bounds = [np.inf, upper_bound_a]#, np.inf, np.inf]  
    #upper_bounds = [upper_bound_h, upper_bound_a, upper_bound_b, upper_bound_c, upper_bound_d]
    bounds = (lower_bounds, upper_bounds)
    initial_guess=(h, a) 
    print(f"lower bounds: {lower_bounds}")
    print(f"upper bounds: {upper_bounds}")
    print(f"Initial guess: {initial_guess}")

    return initial_guess, weights, bounds

# Approximates the difference between the difference of the first two elements and the last two elements. 
    # Used to provide estimation of what h would be (extend ellipse if necessary)
def difference_between_difference(h_values):
    difference = h_values[1]-h_values[0]
    difference2=h_values[2]-h_values[1]
    diff_diff=difference2-difference
    start_diff=diff_diff-difference

    end_diff=h_values[-2]-h_values[-1]
    end_diff2=h_values[-3]-h_values[-2]
    end_diff_diff=end_diff2-end_diff
    end_diff=end_diff_diff-end_diff
    return np.abs(start_diff), np.abs(end_diff)


def fit_ellipse(data):

    # Ensure 'ellipse_h_<>ef' and 'ellipse_v_<>ef' columns exist in the DataFrame
    if 'ellipse_h_def' not in data.columns: 
        data['ellipse_h_def'] = pd.Series([np.array([])] * len(data['h_def_rot']), index=data.index)
    if 'ellipse_v_def' not in data.columns:
        data['ellipse_v_def'] = pd.Series([np.array([])] * len(data['v_def_rot']), index=data.index)
    if 'ellipse_h_ref' not in data.columns:
        data['ellipse_h_ref'] = pd.Series([np.array([])] * len(data['h_ref_rot']), index=data.index)
    if 'ellipse_v_ref' not in data.columns:
        data['ellipse_v_ref'] = pd.Series([np.array([])] * len(data['v_ref_rot']), index=data.index)


    # Get intial guesses, weights and bounds for the fit
    def_initial_guesses, def_weights, def_bounds = get_fit_params(data, name='def_rot')
    ref_initial_guesses, ref_weights, ref_bounds = get_fit_params(data, name='ref_rot')

    
    def_params, def_covariance = curve_fit(func, data['h_def_rot'].iloc[0], data['v_def_rot'].iloc[0], p0=def_initial_guesses, bounds=def_bounds)
    ref_params, ref_covariance = curve_fit(func, data['h_ref_rot'].iloc[0], data['v_ref_rot'].iloc[0], p0=ref_initial_guesses, bounds=ref_bounds)

    print('***COVARIANCE: ***')
    np.linalg.cond(def_covariance)
    print(def_covariance)
    np.linalg.cond(ref_covariance)
    print(ref_covariance)
    print('***PARAMS: ***')
    #print(f"fitted parameters: h={params[0]}, a={params[1]}, b={params[2]}")
    print(f"patient id: {data['patient_id'].iloc[0]}, timepoint: {data['timepoint'].iloc[0]}")
    print(f"def fitted parameters: h={def_params[0]}, a={def_params[1]}")
    print(f"ref fitted parameters: h={ref_params[0]}, a={ref_params[1]}")

    # Extract the optimal parameter
    def check_params(params):

        if len(params) == 3:
            print('****3PARAMETERS****')
            h_optimal, a_optimal, b_optimal = params
            print(f"h_optimal (height at x=0): {h_optimal}")
            print(f"a_optimal (width): {a_optimal}")
            print(f"b_optimal (skew): {b_optimal}")
            return h_optimal, a_optimal, b_optimal
            
        elif len(params) == 2:
            h_optimal, a_optimal = params
            print(h_optimal)
            print(a_optimal)
            return h_optimal, a_optimal
        else:
            print('userdeferror: number of parameters ! = number of variables')
            return None
    
    h_optimal_def, a_optimal_def = check_params(def_params)
    h_optimal_ref, a_optimal_ref = check_params(ref_params)

    # plot ellipse
    def_h_values = np.linspace(min(data['h_def_rot'].iloc[0]), max(data['h_def_rot'].iloc[0]), 1000)
    print(f"min: {min(data['h_def_rot'].iloc[0])}, max: {max(data['h_def_rot'].iloc[0])}")
    #print(f"def_h_values: {def_h_values}")
    ref_h_values = np.linspace(min(data['h_ref_rot'].iloc[0]), max(data['h_ref_rot'].iloc[0]), 1000)
    if len(def_params) == 2:
        def_v_fitted = func(def_h_values, h_optimal_def, a_optimal_def)
    #elif len(ref_params) == 2:
    #    v_fitted = func(def_h_values, h_optimal_def, a_optimal_def, b_optimal_def)
    #elif len(params) == 3:
    #    v_fitted = funcb(h_values, h_optimal, a_optimal, b_optimal)
    else:
        print(f"userdeferror: v_fitted not calculated, number of parameters, {len(def_params)}!= number of function variables")
    
    if len(ref_params) == 2:
        ref_v_fitted = func(ref_h_values, h_optimal_ref, a_optimal_ref)
    else:
        print(f"userdeferror: v_fitted not calculated, number of parameters, {len(ref_params)}!= number of function variables")



    #
    print("**********TESTING")
    print(f"def_h_values len: {len(def_h_values)}")
    print(f"first few values of def_h_values: {def_h_values[:5]}")
    print(f"def_v_fitted len: {len(def_v_fitted)}")
    print(f"first few values of def_v_fitted: {def_v_fitted[:5]}")
    print(f"ref_h_values len: {len(ref_h_values)}")
    print(f"first few values of ref_h_values: {ref_h_values[:5]}")
    print(f"ref_v_fitted len: {len(ref_v_fitted)}")
    print(f"first few values of ref_v_fitted: {ref_v_fitted[:5]}")
    #remove 0 values from v_fitted (retaining a 0 at either end)
    # Identify the last non-zero element in v_fitted
    if def_v_fitted[0] == 0: # end points of array (or all points if not skew ellipse) are generally symmetrical
        last_non_zero_index_def = np.max(np.nonzero(def_v_fitted))
        first_non_zero_index_def = np.min(np.nonzero(def_v_fitted))
        first_index_def=first_non_zero_index_def-1
        last_index_def=last_non_zero_index_def+2
    else:
        first_index_def=0
        # add zero to def_v_fitted[0]
        def_v_fitted=np.insert(def_v_fitted, 0, 0)
        # add zero to def_v_fitted[-1]
        def_v_fitted=np.append(def_v_fitted, 0)
        # get corresponding h_values (approximate using linear interpolation)
        
        start_diff, end_diff=difference_between_difference(def_h_values)
        def_h_values=np.insert(def_h_values, 0, def_h_values[0]-start_diff)
        def_h_values=np.append(def_h_values, def_h_values[-1]-end_diff)

        last_index_def=len(def_v_fitted)
    
    if ref_v_fitted[0] == 0:
        last_non_zero_index_ref = np.max(np.nonzero(ref_v_fitted))
        first_non_zero_index_ref = np.min(np.nonzero(ref_v_fitted))
        first_index_ref=first_non_zero_index_ref-1
        last_index_ref=last_non_zero_index_ref+2
    else:
        print("inside else")
        first_index_ref=0
        # add zero to def_v_fitted[0]
        ref_v_fitted=np.insert(ref_v_fitted, 0, 0)
        # add zero to def_v_fitted[-1]
        ref_v_fitted=np.append(ref_v_fitted, 0)
        # get corresponding h_values (approximate using linear interpolation)
        ref_start_diff, ref_end_diff=difference_between_difference(ref_h_values)
        ref_h_values=np.insert(ref_h_values, 0, ref_h_values[0]-ref_start_diff)
        ref_h_values=np.append(ref_h_values, ref_h_values[-1]-ref_end_diff)


        last_index_ref=len(ref_v_fitted)


    # v_fitted filter / sliced between first and last index
    def_v_fitted_filtered = def_v_fitted[first_index_def:last_index_def]
    print(f"def_v_fitted_filtered len: {len(def_v_fitted_filtered)}")
    print(f"first few values of def_v_fitted_filtered: {def_v_fitted_filtered[:5]}")
    def_h_values_filtered = def_h_values[first_index_def:last_index_def]
    print(f"def_h_values_filtered len: {len(def_h_values_filtered)}")
    print(f"first few values of def_h_values_filtered: {def_h_values_filtered[:5]}")
    ref_v_fitted_filtered = ref_v_fitted[first_index_ref:last_index_ref]
    print(f"ref_v_fitted_filtered len: {len(ref_v_fitted_filtered)}")
    print(f"first few values of ref_v_fitted_filtered: {ref_v_fitted_filtered[:5]}")
    ref_h_values_filtered = ref_h_values[first_index_ref:last_index_ref]
    print(f"ref_h_values_filtered len: {len(ref_h_values_filtered)}")
    print(f"first few values of ref_h_values_filtered: {ref_h_values_filtered[:5]}")

    # get first character of name
    location_h_name_def=f"ellipse_h_def"
    location_v_name_def=f"ellipse_v_def"
    location_h_name_ref=f"ellipse_h_ref"
    location_v_name_ref=f"ellipse_v_ref"
    # Put into df

    print(f"data index: {data.index}")
    print(f"data index is 0: {0 in data.index}")
    if 0 in data.index and location_h_name_def in data.columns and location_v_name_def in data.columns:
        data.at[0, location_h_name_def] = def_h_values_filtered
        data.at[0, location_v_name_def] = def_v_fitted_filtered
    else:
        print("Index or column name does not exist.")
        print(f"Index: {0} in data: {0 in data.index}")
        data.at[0, location_h_name_def] = def_h_values_filtered
        data.at[0, location_v_name_def] = def_v_fitted_filtered
        
        """
        print(f"Column name: {location_h_name} in data: {location_h_name in data.columns}")

        print("resetting index...")
        data=data.reset_index(drop=True)
        print(f"Index: {0} in data: {0 in data.index}")
        print(f"Column name: {location_h_name} in data: {location_h_name in data.columns}")
        data.at[0, location_h_name] = h_values_filtered
        data.at[0, location_v_name] = v_fitted_filtered
        """
        #sys.exit()
    
    if 0 in data.index and location_h_name_ref in data.columns and location_v_name_ref in data.columns:
        data.at[0, location_h_name_ref] = ref_h_values_filtered
        data.at[0, location_v_name_ref] = ref_v_fitted_filtered
    else:
        print("Index or column name does not exist.")
        print(f"Index: {0} in data: {0 in data.index}")
        data.at[0, location_h_name_ref] = ref_h_values_filtered
        data.at[0, location_v_name_ref] = ref_v_fitted_filtered

    print("data after fitting ellipse: \n", data)
    #data.at[0, location_h_name] = h_values_filtered 
    #data.at[0,location_v_name] = v_fitted_filtered
    #print(f"data.at[0, {location_h_name}]: {data.at[0, location_h_name]}")
    #print(f"data.at[0, {location_v_name}]: {data.at[0, location_v_name]}")

    
    
    return data#, h_values_filtered, v_fitted_filtered



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
    print(f"transformed data shape: {transformed_data.shape}")
    
    
    # Plot transformed data
    plt.scatter(transformed_data['h_def_tr'].iloc[0], transformed_data['v_def_tr'].iloc[0], color='red', s=1)
    plt.scatter(transformed_data['h_def_tr'].iloc[0][-2], transformed_data['v_def_tr'].iloc[0][-2], color='magenta', s=20) # anterior point
    plt.scatter(transformed_data['h_ref_tr'].iloc[0], transformed_data['v_ref_tr'].iloc[0], color='cyan', s=1)
    plt.title(f"{transformed_data['patient_id'].iloc[0]} {transformed_data['timepoint'].iloc[0]}")
    # Set the aspect ratio of the plot to be equal
    plt.gca().set_aspect('equal', adjustable='box')
    plt.close()

   
    transformed_data=rotate_points(transformed_data) # Rotate function
    
    print(f"transformed data shape: {transformed_data.shape}")
    

    # plot rotated data
    plt.scatter(transformed_data['h_def_rot'].iloc[0], transformed_data['v_def_rot'].iloc[0], color='red', s=1)
    plt.scatter(transformed_data['h_def_rot'].iloc[0][-2], transformed_data['v_def_rot'].iloc[0][-2], color='magenta', s=20) # anterior point
    plt.scatter(transformed_data['h_ref_rot'].iloc[0], transformed_data['v_ref_rot'].iloc[0], color='cyan', s=1)
    plt.title(f"{transformed_data['patient_id']} {transformed_data['timepoint']}")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.close()

    
    transformed_data=center_points(transformed_data) # Center function (parks data back into rotated column)
    print(f"transformed data shape: {transformed_data.shape}")
    

    # plot data
    plt.scatter(transformed_data['h_def_rot'].iloc[0], transformed_data['v_def_rot'].iloc[0], color='red', s=1)
    plt.scatter(transformed_data['h_ref_rot'].iloc[0], transformed_data['v_ref_rot'].iloc[0], color='cyan', s=1)
    plt.scatter(transformed_data['h_def_rot'].iloc[0][-2], transformed_data['v_def_rot'].iloc[0][-2], color='magenta', s=20)
    plt.scatter(transformed_data['h_def_rot'].iloc[0][-1], transformed_data['v_def_rot'].iloc[0][-1], color='green', s=20)
    plt.title(f"{transformed_data['patient_id'].iloc[0]} {transformed_data['timepoint'].iloc[0]}")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.close()

    

    # Fit ellipse using least squares method - store data / parameters line by line
    # Fit ellipse through transformed_data['h_def_rot'] and transformed_data['v_def_rot']
    
    
    
    print(f"transformed data index: {transformed_data.index}")
    print(f"transformed data index is 0: {0 in transformed_data.index}")
    if not 0 in transformed_data.index:
        transformed_data=transformed_data.reset_index(drop=True)
        print(f"pre function reset index: {transformed_data.index}")


    ellipse_data = fit_ellipse(transformed_data)
    #ellipse_data, ref_h_values, ref_v_fitted = fit_ellipse(transformed_data)    
    #h_values_padded = np.append(h_values_filtered, h_values_filtered[-1] + 1)
    print(f"transformed_data_shape post ellipse: {ellipse_data.shape}")
    print(f"ellipse data: \n {ellipse_data}")
    #print(f"def_h_values: {def_h_values}")
    #print(f"def_v_fitted: {def_v_fitted}")
    #print(f"ref_h_values: {ref_h_values}")
    #print(f"ref_v_fitted: {ref_v_fitted}")
    print(f"ellipse_data columns: {ellipse_data.columns}")
    
    
    


    # PLOT FITTED ELLIPSE
    plt.scatter(ellipse_data['h_def_rot'].iloc[0], ellipse_data['v_def_rot'].iloc[0], label='translated and rotated data points', color='red', s=2)
    plt.plot(ellipse_data['ellipse_h_def'].iloc[0], ellipse_data['ellipse_v_def'].iloc[0], label='Fitted curve', color='red')
    plt.scatter(transformed_data['h_ref_rot'].iloc[0], transformed_data['v_ref_rot'].iloc[0], label='translated and rotated data points', color='cyan', s=2)
    plt.plot(ellipse_data['ellipse_h_ref'].iloc[0], ellipse_data['ellipse_v_ref'].iloc[0], label='Fitted curve', color='cyan')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{transformed_data['patient_id'].iloc[0]} {transformed_data['timepoint'].iloc[0]}")
    plt.show()
    

        
    # Store fitted ellipse data in DataFrame
    print("HERE HERE HERE")
    print(transformed_data.columns)
    
    #print(f"transformed_data.at[0,'ellipse_h_def']: {transformed_data.at[0,'ellipse_h_def']}")
    #transformed_data = put_in_df(transformed_data, def_h_values, def_v_fitted, ref_h_values, ref_v_fitted)
    
    #print(F"H VALUES FILTERED: {def_h_values}")
    #print("H_VALUES_FILTERED TYPE: ", type(def_h_values))
    #transformed_data.at[0, 'ellipse_h_def'] = def_h_values.copy()
    #transformed_data.at[0,'ellipse_v_def'] = def_v_fitted.copy()
    #transformed_data.at[0,'ellipse_h_ref'] = ref_h_values.copy()  
    #transformed_data.at[0,'ellipse_v_ref'] = ref_v_fitted.copy()
    
   
    #print(f"transformed_data first row: \n {transformed_data.iloc[0]}")
    
    new_row = ellipse_data.iloc[0]
    """
    for i in range(len(total_df)):
        
        print(f"Currently at patient id and timepoint {new_row['patient_id']} {new_row['timepoint']}")
        print("new_row: \n", new_row)
        if i < len(total_df)-1:
            input("Press Enter to continue... ")
            break
        else:
            print("End of loop.")
    #print(f"new_row: \n {new_row}")
    #sys.exit()
    """
    #plt.scatter(transformed_data['ellipse_h_def'].iloc[0], transformed_data['ellipse_v_def'].iloc[0], color='red', s=1)
    #plt.scatter(transformed_data['ellipse_h_ref'].iloc[0], transformed_data['ellipse_v_ref'].iloc[0], color='cyan', s=1)
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()

    transformed_df = pd.concat([transformed_df, new_row], axis=1, ignore_index=True)
    #print(f"transformed_df: \n {transformed_df}")
    #sys.exit()

    # Find change in area between two ellipses
    
    # Store data as one big df
    #transformed_df = transformed_df.append(transformed_data.iloc[0], ignore_index=True)
    #print(transformed_df.iloc[i])

    

print('*****')
#print(total_df.columns)
print(transformed_df.head)




# Plot change in area over time for each patient (x axis: time, y axis: area)

# Reverse transform data points, save to df / .csv

# Plot on image.
