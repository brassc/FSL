import pandas as pd
pd.options.mode.copy_on_write = True # to avoid SettingWithCopyWarning
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.patches import Ellipse
from scipy.linalg import eig
from scipy.optimize import curve_fit


def resample(contour_x, contour_y):
    # Check for empty or single-point contours
    #return contour_x, contour_y # skip resampling

    # if len(contour_x) <= 1:
    #     print("Warning: Contour has insufficient points for resampling")
    #     return contour_x, contour_y
    
    # # sort contour_x, contour_y by contour_x
    # sorted_indices = np.argsort(contour_x)
    # contour_x = contour_x[sorted_indices]
    # contour_y = contour_y[sorted_indices]

    # # calculate gaps between adjacent x points
    # x_gaps = np.diff(contour_x)

    # # identify large gaps that need resampling
    # # if min_gap_threshold is not provided, calculate it as a percentage of the range
    # x_range=max(contour_x)-min(contour_x)
    # min_gap_threshold = 0.1*x_range # 10% of the range

    # # Make sure we have a reasonable threshold
    # if min_gap_threshold <= 0:
    #     return contour_x, contour_y

    # large_gaps = np.where(x_gaps > min_gap_threshold)[0]

    # # if there are no large gaps, return the original contour
    # if len(large_gaps) == 0:
    #     return contour_x, contour_y
    
    # # resample the contour at the large gaps
    # resampled_x = contour_x.copy().tolist()
    # resampled_y = contour_y.copy().tolist()

    # # keep track of how many points added
    # points_added = 0

    # # For each large gap, add interpolated points
    # for idx in large_gaps:
    #     # current gap
    #     gap_size=x_gaps[idx]

    #     # calculate how manyt points to add proportional to the gap size
    #     num_points_to_add = max(1, int(gap_size / min_gap_threshold)-1)
    #     num_points_to_add = min(num_points_to_add, 3) # limit to 10 points avoid overfitting

    #     # current points
    #     x1, y1 = contour_x[idx], contour_y[idx]
    #     x2, y2 = contour_x[idx+1], contour_y[idx+1]

    #     # insert interpolated points
    #     for i in range(1, num_points_to_add+1):
    #         t= i/(num_points_to_add+1)
    #         new_x=x1+t*(x2-x1)
    #         new_y=y1+t*(y2-y1)

    #         # insert at correct position
    #         insert_idx = idx + 1 + points_added
    #         resampled_x.insert(insert_idx, new_x)
    #         resampled_y.insert(insert_idx, new_y)
    #         points_added += 1

    # # sort points correctly
    # sorted_indices = np.argsort(resampled_x)
    # resampled_x = np.array(resampled_x)[sorted_indices]
    # resampled_y = np.array(resampled_y)[sorted_indices]

    # print(f"added {points_added} points in large gaps")
    # plt.scatter(resampled_x, resampled_y, color='green', marker='o', linestyle='-', s=20)
    # plt.scatter(contour_x, contour_y, color='blue', s=5)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.close()

    




    # return np.array(resampled_x), np.array(resampled_y)



    if len(contour_x) <= 1:
        print("Warning: Contour has insufficient points for resampling")
        return contour_x, contour_y
    
    print(f"Contour points: {len(contour_x)}")

    plt.scatter(contour_x, contour_y, color='blue', s=5)
    plt.gca().set_aspect('equal', adjustable='box')
    
      
    # hardcoded positions for skull end points
    first_point_x=contour_x[-2]
    first_point_y=contour_y[-2]
    last_point_x=contour_x[-1]
    last_point_y=contour_y[-1]
    # plt.scatter(first_point_x, first_point_y, color='orange', s=50)
    # plt.scatter(last_point_x, last_point_y, color='orange', s=50)
    # plt.show()


    # sort (contour_x, contour_y) by contour_x
    sorted_indices = np.argsort(contour_x)
    contour_x = contour_x[sorted_indices]
    contour_y = contour_y[sorted_indices]
    
    # Create a continuous contour representation by connecting points
    # with straight lines and resampling
    resampled_x = []
    resampled_y = []
    
    # Calculate the total length of the contour
    total_length = 0
    segment_lengths = []
    for i in range(len(contour_x) - 1):
        dx = contour_x[i+1] - contour_x[i]
        dy = contour_y[i+1] - contour_y[i]
        segment_length = np.sqrt(dx**2 + dy**2)
        segment_lengths.append(segment_length)
        total_length += segment_length
    
    # Check if we have valid segment lengths
    if total_length <= 0 or len(segment_lengths) == 0:
        print("Warning: Zero total length or no segments found")
        return contour_x, contour_y
    
    # Determine number of points based on minimum segment length
    #min_segment_length = min(segment_lengths)
    sorted_segments = sorted([length for length in segment_lengths if length > 0])
    
    for segment_length in sorted_segments:
        try:
            num_points = max(int(total_length / segment_length), len(contour_x))
            break  # Success - exit the loop
        except (OverflowError, ValueError):
            continue


    print(f"Total contour length: {total_length}, Segments: {len(segment_lengths)}")
    print(f"Using {num_points} points based on minimum segment length of {segment_length:.4f}")

    # Explicitly add the first point to ensure it's preserved
    resampled_x.append(contour_x[0])
    resampled_y.append(contour_y[0])
    
    # Resample at equal intervals along the contour
    for i in range(num_points):
        # Position along the contour (normalized)
        target_dist = (i / (num_points - 1 if num_points > 1 else 1)) * total_length
        
        # Find which segment contains this position
        segment_idx = 0
        cumulative_length = 0
        while segment_idx < len(segment_lengths) and cumulative_length + segment_lengths[segment_idx] < target_dist:
            cumulative_length += segment_lengths[segment_idx]
            segment_idx += 1
        
        # Handle edge case
        if segment_idx >= len(segment_lengths):
            # Add the last point
            resampled_x.append(contour_x[-1])
            resampled_y.append(contour_y[-1])
            continue
        
        # Calculate position along current segment
        segment_pos = (target_dist - cumulative_length) / segment_lengths[segment_idx]
        
        # Get the segment's start and end points
        start_idx = segment_idx
        end_idx = segment_idx + 1
        
        # Ensure end_idx is valid
        if end_idx >= len(contour_x):
            # Just use the last point for open contours
            end_idx = len(contour_x) - 1
        
        # Interpolate to get coordinates
        x = contour_x[start_idx] + segment_pos * (contour_x[end_idx] - contour_x[start_idx])
        y = contour_y[start_idx] + segment_pos * (contour_y[end_idx] - contour_y[start_idx])
        
        resampled_x.append(x)
        resampled_y.append(y)
    
    # Explicitly add the last point to ensure it's preserved
    resampled_x.append(last_point_x)
    resampled_y.append(last_point_y)
    
    print(f"Generated {len(resampled_x)} resampled points")
    
    # Ensure we have multiple points
    if len(resampled_x) <= 1:
        print("Warning: Resampling produced too few points, returning original contour")
        return contour_x, contour_y
    
    # Downsample if necessary, preserve end points
    max_points = 10
    resample_x_copy = resampled_x.copy()
    resample_y_copy = resampled_y.copy()
    if len(resampled_x) > max_points:
        print(f"Downsampling from {len(resampled_x)} to {max_points} points")
        # Get first point
        first_x, first_y = resampled_x[0], resampled_y[0]
        # Find the last point where y=0 (or very close to 0)
                
        # Downsample the middle points
        step = (len(resampled_x) - 2) // (max_points - 2)
        middle_indices = range(1, len(resampled_x) - 1, step)
        middle_x = [resampled_x[i] for i in middle_indices]
        middle_y = [resampled_y[i] for i in middle_indices]
        
        # Combine the points
        resampled_x = [first_point_x] + middle_x + [last_point_x]
        resampled_y = [first_point_y] + middle_y + [last_point_y]
    plt.scatter(first_point_x, first_point_y, color='orange', s=50)
    plt.scatter(last_point_x, last_point_y, color='orange', s=50)
    plt.scatter(resampled_x, resampled_y, color='green', marker='o', linestyle='-', s=20)
    plt.scatter(contour_x, contour_y, color='blue', s=5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.close()

    
    return np.array(resampled_x), np.array(resampled_y)

def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def convert_to_numpy_array(s):
    # Remove extra whitespace and split by spaces
    # Convert the resulting list of strings to a list of integers
    print("s:", s)
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
    
    # Create new columns to store the centered points
    if 'h_def_cent' not in data.columns:
        data['h_def_cent'] = pd.Series([np.array([])] * len(data['h_def_rot']), index=data.index)
    if 'v_def_cent' not in data.columns:
        data['v_def_cent'] = pd.Series([np.array([])] * len(data['v_def_rot']), index=data.index)
    if 'h_ref_cent' not in data.columns:
        data['h_ref_cent'] = pd.Series([np.array([])] * len(data['h_ref_rot']), index=data.index)
    if 'v_ref_cent' not in data.columns:
        data['v_ref_cent'] = pd.Series([np.array([])] * len(data['v_ref_rot']), index=data.index)
    
    # Fill new v center columns even though these are the same as the v_rot columns
    data['v_def_cent'] = data['v_def_rot']
    data['v_ref_cent'] = data['v_ref_rot']

    if (data['side'] == 'R').any():
        data['h_def_cent'] = data['h_def_rot'] - average_h
        data['h_ref_cent'] = data['h_ref_rot'] - average_h
    elif (data['side'] == 'L').any():
        data['h_def_cent'] = data['h_def_rot'] - average_h
        data['h_ref_cent'] = data['h_ref_rot'] - average_h
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

    # Check if indices are the same
    if left_index == right_index:
        print("Warning: Left and right indices are the same. Using point height directly.")
        return v_coords[left_index]  # Just use the height at this point


    # Perform linear interpolation between the points
    try:
        slope = (v_coords[right_index] - v_coords[left_index]) / (h_coords[right_index] - h_coords[left_index])
        intersection_height = v_coords[left_index] - slope * h_coords[left_index]
    except:
        # use the height at the left index
        print("Warning: Zero division error. Using point height directly.")
        intersection_height=v_coords[left_index]
    
    print(f"Intersection height: {intersection_height}")
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

def get_fit_params(data, name='<>ef_cent'): # name e.g. 'def_rot'
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

def initialize_columns(data, name):
    h_col = f'ellipse_h_{name}'
    v_col = f'ellipse_v_{name}'
    h_param_col = f'h_param_{name}'
    a_param_col = f'a_param_{name}'
    if h_col not in data.columns:
        data[h_col] = pd.Series([np.array([])] * len(data[f'h_{name}_rot']), index=data.index)
    if v_col not in data.columns:
        data[v_col] = pd.Series([np.array([])] * len(data[f'v_{name}_rot']), index=data.index)
    if h_param_col not in data.columns:
        data[h_param_col] = pd.Float32Dtype()
    if a_param_col not in data.columns:
        data[a_param_col] = pd.Float32Dtype()
    return h_col, v_col, h_param_col, a_param_col

def fit_data(data, name):
    # Get initial guesses, weights and bounds for the fit
    initial_guesses, weights, bounds = get_fit_params(data, name=f'{name}_cent')

    # Perform the curve fitting
    params, covariance = curve_fit(
        func,
        data[f'h_{name}_cent'].iloc[0],
        data[f'v_{name}_cent'].iloc[0],
        p0=initial_guesses,
        bounds=bounds
    )
    # Display covariance matrix and condition number
    print(f'*** {name.upper()} COVARIANCE: ***')
    print('Condition Number:', np.linalg.cond(covariance))
    print('Covariance Matrix:\n', covariance)

    return params


def check_params(params):
    if len(params) == 3:
        print('**** 3 PARAMETERS ****')
        h_optimal, a_optimal, b_optimal = params
        print(f"h_optimal (height at x=0): {h_optimal}")
        print(f"a_optimal (width): {a_optimal}")
        print(f"b_optimal (skew): {b_optimal}")
        return h_optimal, a_optimal, b_optimal
    elif len(params) == 2:
        h_optimal, a_optimal = params
        print(f"h_optimal: {h_optimal}")
        print(f"a_optimal: {a_optimal}")
        return h_optimal, a_optimal
    else:
        print('userdeferror: number of parameters != number of variables')
        return None

def calculate_fitted_values(data, params, name):
    h_values = np.linspace(min(data[f'h_{name}_cent'].iloc[0]), max(data[f'h_{name}_cent'].iloc[0]), 1000)
    
    if len(params) == 2:
        v_fitted = func(h_values, *params)
    else:
        print(f"userdeferror: v_fitted not calculated, number of parameters, {len(params)} != number of function variables")
        return None, None
    return h_values, v_fitted

"""
def filter_fitted_values_old(h_values, v_fitted):
    if v_fitted[0] == 0:
        last_non_zero_index = np.max(np.nonzero(v_fitted))
        first_non_zero_index = np.min(np.nonzero(v_fitted))
        first_index = first_non_zero_index - 1
        last_index = last_non_zero_index + 2
    else:
        first_index = 0
        # Insert zero at the start
        v_fitted = np.insert(v_fitted, 0, 0)
        # Append zero at the end
        v_fitted = np.append(v_fitted, 0)

        # Approximate corresponding h_values using linear interpolation
        start_diff, end_diff = difference_between_difference(h_values)
        h_values = np.insert(h_values, 0, h_values[0] - start_diff)
        h_values = np.append(h_values, h_values[-1] + end_diff)

        last_index = len(v_fitted)
    
    # Slice the h_values and v_fitted between the first and last index
    h_values_filtered = h_values[first_index:last_index]
    v_fitted_filtered = v_fitted[first_index:last_index]

    return h_values_filtered, v_fitted_filtered
"""


def filter_fitted_values(h_values, v_fitted):
    # First apply the original filter_fitted_values_old function exactly as it is
    if v_fitted[0] == 0:
        last_non_zero_index = np.max(np.nonzero(v_fitted))
        first_non_zero_index = np.min(np.nonzero(v_fitted))
        first_index = first_non_zero_index - 1
        last_index = last_non_zero_index + 2
    else:
        first_index = 0
        # Insert zero at the start
        v_fitted = np.insert(v_fitted, 0, 0)
        # Append zero at the end
        v_fitted = np.append(v_fitted, 0)
        # Approximate corresponding h_values using linear interpolation
        start_diff, end_diff = difference_between_difference(h_values)
        h_values = np.insert(h_values, 0, h_values[0] - start_diff)
        h_values = np.append(h_values, h_values[-1] + end_diff)
        last_index = len(v_fitted)
    
    # Slice the h_values and v_fitted between the first and last index
    h_values_filtered = h_values[first_index:last_index]
    v_fitted_filtered = v_fitted[first_index:last_index]
    
    # Now add gradient-based filtering as a separate step
    # Calculate gradients on the already filtered data
    gradients = np.gradient(v_fitted_filtered, h_values_filtered)

    second_derivatives = np.gradient(gradients, h_values_filtered)

    # Define threshold for sudden changes in gradient
    second_derivative_threshold = 80#200  # Adjust this value based on your data
    
    # Define threshold for steep segments
    #max_gradient_threshold = 100.0  # For steep segments
    
    # Find valid indices where gradient is not too steep
    #valid_indices = np.where(np.abs(gradients) < max_gradient_threshold)[0]
    valid_indices = np.where(np.abs(second_derivatives) < second_derivative_threshold)[0]
    
    # Only apply gradient filtering if we have enough valid indices
    if len(valid_indices) > 10:
        # Create new filtered arrays
        h_gradient_filtered = []
        v_gradient_filtered = []
        
        # Only include points where gradient is acceptable
        for i in valid_indices:
            h_gradient_filtered.append(h_values_filtered[i])
            v_gradient_filtered.append(v_fitted_filtered[i])
        
        # Convert to numpy arrays
        h_gradient_filtered = np.array(h_gradient_filtered)
        v_gradient_filtered = np.array(v_gradient_filtered)
        
        return h_gradient_filtered, v_gradient_filtered
    
    # If gradient filtering didn't work, return the original filtered arrays
    return h_values_filtered, v_fitted_filtered


"""
def filter_fitted_values_working(h_values, v_fitted):
    # Find indices where v_fitted is greater than a small epsilon value
    epsilon = 1e-10  # Small value to account for floating-point precision
    valid_indices = np.where(v_fitted > epsilon)[0]
    
    if len(valid_indices) > 0:
        # Get the first and last valid indices
        first_index = valid_indices[0]
        last_index = valid_indices[-1]
        
        # Slice the arrays to keep only the valid portion
        h_values_filtered = h_values[first_index:last_index+1]
        v_fitted_filtered = v_fitted[first_index:last_index+1]
    else:
        # If no valid values, return empty arrays
        h_values_filtered = np.array([])
        v_fitted_filtered = np.array([])
    
    return h_values_filtered, v_fitted_filtered

"""
# Function to update DataFrame with fitted values
def update_dataframe(data, h_values_filtered, v_fitted_filtered, h_col, v_col, h_param_col, a_param_col, h_optimal, a_optimal):
    if 0 in data.index and h_col in data.columns and v_col in data.columns:
        data.at[0, h_col] = h_values_filtered
        data.at[0, v_col] = v_fitted_filtered
        data.at[0, h_param_col] = h_optimal
        data.at[0, a_param_col] = a_optimal
    else:
        print("Index or column name does not exist.")
        data.at[0, h_col] = h_values_filtered
        data.at[0, v_col] = v_fitted_filtered
        data.at[0, h_param_col] = h_optimal
        data.at[0, a_param_col] = a_optimal

    return data



def fit_ellipse(data):

    for name in ['def', 'ref']:
        # Initialize the columns
        h_col, v_col, h_param_col, a_param_col = initialize_columns(data, name)

        # Fit the data
        params = fit_data(data, name)

        # Display fitted params
        print(f"*** {name.upper()} PARAMS: ***")
        print(f"Patient ID: {data['patient_id'].iloc[0]}, Timepoint: {data['timepoint'].iloc[0]}")
        print(f"Fitted parameters: h={params[0]}, a={params[1]}")

        # Extract the optimal parameters
        h_optimal, a_optimal = check_params(params)

        # Calculate the fitted values
        h_values, v_fitted = calculate_fitted_values(data, [h_optimal, a_optimal], name)

        # Filter the fitted values
        h_values_filtered, v_fitted_filtered = filter_fitted_values(h_values, v_fitted)

        # Update the DataFrame with the fitted values
        data = update_dataframe(data, h_values_filtered, v_fitted_filtered, h_col, v_col, h_param_col, a_param_col, h_optimal, a_optimal)

    print("Data after fitting ellipse: \n", data)
    return data

    
if __name__=='__main__':
    ## MAIN SCRIPT TO PLOT ELLIPSE FORM
    # data = pd.read_csv('Image_Processing_Scripts/data_entries.csv')
    # side_data=pd.read_csv('Image_Processing_Scripts/included_patient_info.csv')
    # ellipse_data_filename='ellipse_data.csv'
    data = pd.read_csv('Image_Processing_Scripts/batch2_data_entries.csv')
    side_data=pd.read_csv('Image_Processing_Scripts/batch2_included_patient_info.csv')
    ellipse_data_filename='batch2_ellipse_data.csv'

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



    # Transform data points such that posterior point lies on (0, 0) and anterior lies on y=0 (x axis) 
    #    using transform_points function and rotate_points function
    #       Recall baseline coords are at end of contour anterior, posterior (last two points in list in that order)


    # Initialise data frame to add to
    transformed_df = pd.DataFrame()#columns=['patient_id', 'timepoint', 'deformed_contour_x', 'deformed_contour_y',
        #'reflected_contour_x', 'reflected_contour_y', 'side', 'h_def', 'v_def',
        #'h_ref', 'v_ref', 'h_def_tr', 'v_def_tr', 'h_ref_tr', 'v_ref_tr',
        #'h_def_rot', 'v_def_rot', 'h_ref_rot', 'v_ref_rot', 'angle',
        #'ellipse_h_def', 'ellipse_v_def', 'ellipse_h_ref', 'ellipse_v_ref'])
    
    # Loop through each row in the total_df
    for i in range (len(total_df)):
        #print(total_df.iloc[i])
        
        # get copy of slice of total_df line by line
        data = total_df.iloc[[i]].copy()
        print(data)
        data.columns=total_df.columns
        
        #Plot original data
        plt.scatter(data['h_def'].iloc[0], data['v_def'].iloc[0], color='red', s=1)
        plt.scatter(data['h_ref'].iloc[0], data['v_ref'].iloc[0], color='blue', s=1)
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
        plt.scatter(transformed_data['h_ref_tr'].iloc[0], transformed_data['v_ref_tr'].iloc[0], color='blue', s=1)
        plt.title(f"{transformed_data['patient_id'].iloc[0]} {transformed_data['timepoint'].iloc[0]}")
        # Set the aspect ratio of the plot to be equal
        plt.gca().set_aspect('equal', adjustable='box')
        plt.close()

    
        transformed_data=rotate_points(transformed_data) # Rotate function
        
        print(f"transformed data shape: {transformed_data.shape}")
        

        # plot rotated data
        plt.scatter(transformed_data['h_def_rot'].iloc[0], transformed_data['v_def_rot'].iloc[0], color='red', s=1)
        plt.scatter(transformed_data['h_def_rot'].iloc[0][-2], transformed_data['v_def_rot'].iloc[0][-2], color='magenta', s=20) # anterior point
        plt.scatter(transformed_data['h_ref_rot'].iloc[0], transformed_data['v_ref_rot'].iloc[0], color='blue', s=1)
        plt.title(f"{transformed_data['patient_id']} {transformed_data['timepoint']}")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.close()

        
        transformed_data=center_points(transformed_data) # Center function (parks data into <>_<>ef_cent columns)
        print(f"transformed data shape: {transformed_data.shape}")
        #print(f"transformed data columns: {transformed_data.columns}")
        #print(f"transformed data at 0: \n {transformed_data.iloc[0]}")  
        

        # plot data
        plt.scatter(transformed_data['h_def_cent'].iloc[0], transformed_data['v_def_cent'].iloc[0], color='red', s=1)
        plt.scatter(transformed_data['h_ref_cent'].iloc[0], transformed_data['v_ref_cent'].iloc[0], color='blue', s=1)
        plt.scatter(transformed_data['h_def_cent'].iloc[0][-2], transformed_data['v_def_cent'].iloc[0][-2], color='magenta', s=20)
        plt.scatter(transformed_data['h_def_cent'].iloc[0][-1], transformed_data['v_def_cent'].iloc[0][-1], color='green', s=20)
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

        print("transformed data.columns: \n", transformed_data.columns)

        ##  apply resampling function for the transformed_data row we are working with (row [0])
        for name in ['def', 'ref']:
            h_name = f'h_{name}_cent'
            v_name = f'v_{name}_cent'

            # Create copies of original data with '_og' suffix
            transformed_data[f'{h_name}_og'] = transformed_data[h_name].copy()
            transformed_data[f'{v_name}_og'] = transformed_data[v_name].copy()

            h_val = transformed_data.loc[0, h_name]
            v_val = transformed_data.loc[0, v_name]
            resampled_x, resampled_y = resample(h_val, v_val)
            transformed_data.at[0, h_name] = resampled_x
            transformed_data.at[0, v_name] = resampled_y
        
        

        ellipse_data = fit_ellipse(transformed_data)
        print(f"transformed_data_shape post ellipse: {ellipse_data.shape}")
        #print(f"ellipse data: \n {ellipse_data}")
        print(f"ellipse_data columns: {ellipse_data.columns}")

        # rename transformed data for name in ['def', 'ref'] from 'h_<name>_cent' to 'h_name>_cent_resampled
            # and 'h_<name>_cent_og' to 'h_<name>_cent'
        for name in ['def', 'ref']:
            h_name = f'h_{name}_cent'
            v_name = f'v_{name}_cent'
            h_name_og = f'h_{name}_cent_og'
            v_name_og = f'v_{name}_cent_og'
            h_name_resampled = f'h_{name}_cent_resampled'
            v_name_resampled = f'v_{name}_cent_resampled'

            transformed_data[h_name_resampled] = transformed_data[h_name]
            transformed_data[v_name_resampled] = transformed_data[v_name]
            transformed_data[h_name] = transformed_data[h_name_og]
            transformed_data[v_name] = transformed_data[v_name_og]
          
        

        # PLOT FITTED ELLIPSE
        set_publication_style()

        plt.figure(figsize=(8, 6), dpi=150)

        plt.scatter(ellipse_data['h_def_cent'].iloc[0], ellipse_data['v_def_cent'].iloc[0], label='translated and rotated data points', s=10, alpha=0.7, color='red')
        plt.plot(ellipse_data['ellipse_h_def'].iloc[0], ellipse_data['ellipse_v_def'].iloc[0], label='Fitted curve', color='red')
        plt.scatter(transformed_data['h_ref_cent'].iloc[0], transformed_data['v_ref_cent'].iloc[0], label='translated and rotated data points', s=10, alpha=0.7, color='blue')
        plt.plot(ellipse_data['ellipse_h_ref'].iloc[0], ellipse_data['ellipse_v_ref'].iloc[0], label='Fitted curve', color='blue')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"{transformed_data['patient_id'].iloc[0]} {transformed_data['timepoint'].iloc[0]}")
        # Set y-axis maximum limit to 60
        plt.ylim(top=60)
        plt.savefig(f"Image_Processing_Scripts/ellipse_plots/{transformed_data['patient_id'].iloc[0]}_{transformed_data['timepoint'].iloc[0]}_ellipse.png")
        plt.close()


        # Create a clean version with only points and curves
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.scatter(ellipse_data['h_def_cent'].iloc[0], ellipse_data['v_def_cent'].iloc[0], s=10, alpha=0.7, color='red')
        ax.plot(ellipse_data['ellipse_h_def'].iloc[0], ellipse_data['ellipse_v_def'].iloc[0], color='red')
        ax.scatter(transformed_data['h_ref_cent'].iloc[0], transformed_data['v_ref_cent'].iloc[0], s=10, alpha=0.7, color='blue')
        ax.plot(ellipse_data['ellipse_h_ref'].iloc[0], ellipse_data['ellipse_v_ref'].iloc[0], color='blue')

        # Set y-axis limit to match the original plot
        ax.set_ylim(top=60)

        # Make sure the aspect ratio is the same as the original
        ax.set_aspect('equal', adjustable='box')

        # Remove axes, ticks, labels, etc.
        ax.axis('off')

        # Set tight layout to crop the image around the content
        plt.tight_layout(pad=0)

               

        # Save the clean version
        plt.savefig(f"Image_Processing_Scripts/ellipse_plots/{transformed_data['patient_id'].iloc[0]}_{transformed_data['timepoint'].iloc[0]}_ellipse_clean.png", 
                    bbox_inches='tight', pad_inches=0, transparent=False)
        plt.savefig(f"../Thesis/phd-thesis-template-2.4/Chapter5/Figs/{transformed_data['patient_id'].iloc[0]}_{transformed_data['timepoint'].iloc[0]}_ellipse_clean.pdf",bbox_inches='tight', pad_inches=0, transparent=False, dpi=150)
        plt.close()
        
            
        # Store fitted ellipse data in DataFrame
        #print(transformed_data.columns)
        new_row = ellipse_data.iloc[0]
        
        transformed_df = pd.concat([transformed_df, new_row], axis=1, ignore_index=True)
        print("transformed_df shape: ", transformed_df.shape)
        
        
        

    print('*****')
    transformed_df = transformed_df.T
    print(transformed_df.columns)
    # remove 'h_def_tr', 'v_def_tr', 'h_ref_tr', 'v_ref_tr' columns
    transformed_df = transformed_df.drop(columns=['h_def', 'v_def', 'h_ref', 'v_ref']) #'h_def_tr', 'v_def_tr', 'h_ref_tr', 'v_ref_tr'
    print(transformed_df.columns)
    print(transformed_df.T.head)
    #print(transformed_df.head)
    # Save to .csv
    transformed_df.to_csv(f'Image_Processing_Scripts/{ellipse_data_filename}', index=False)
    # Also pickle the data - preserve long arrays
    transformed_df.to_pickle(f'Image_Processing_Scripts/{ellipse_data_filename.replace(".csv", ".pkl")}')




# Plot change in area over time for each patient (x axis: time, y axis: area)

# Reverse transform data points, save to df / .csv

# Plot on image.
