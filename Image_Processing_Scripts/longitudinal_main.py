import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline


def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold', 
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

def get_dictionary(data, patient_ids, timepoints, subset_name='h_param_def'):
    patient_dict = {}
    for patient_id in patient_ids:
        patient_dict[patient_id] = {}
        for timepoint in timepoints:
            subset = patient_dict[patient_id][timepoint] = data[(data['patient_id'] == patient_id) & (data['timepoint'] == timepoint)] # subset = patient_dict[patient_id][timepoint] = 
            if len(patient_dict[patient_id][timepoint]) > 0:
                patient_dict[patient_id][timepoint] = subset[subset_name].iloc[0]
            else:
                patient_dict[patient_id][timepoint] = None
    return patient_dict

def convert_dict_to_long_df(data_dict, timepoints, value_name='h_param_def'):
    # Convert from wide to long format
    df = pd.DataFrame(data_dict)
    df = df.T
    df_reset = df.reset_index()
    df_reset.columns = ['patient_id'] + timepoints
    long_df = pd.melt(df_reset, id_vars=['patient_id'], var_name='timepoint', value_name=value_name)
    # Sort by patient_id
    long_df['timepoint'] = pd.Categorical(long_df['timepoint'], categories=timepoints, ordered=True)
    long_df = long_df.sort_values(by=['patient_id', 'timepoint']).reset_index(drop=True)
    return long_df

def create_hex_color_map_from_cmap(cmap_name, n):
    cmap = plt.get_cmap(cmap_name)
    # Convert the colormap to a list of hex colors
    colors = cmap(np.linspace(0, 1, n))
    hex_colors = [rgb2hex(color) for color in colors]
    return hex_colors

def create_hex_color_map_custom(base_colors, n):
    # Create a custom colormap with the base colors
    cmap = LinearSegmentedColormap.from_list('custom', base_colors, N=n)
    # Convert the colormap to a list of hex colors
    colors = cmap(np.linspace(0, 1, n))
    hex_colors = [rgb2hex(color) for color in colors]
    return hex_colors

def get_color_old(unique_label, color_map):
    pid = unique_label.split(' ')[0]
    return color_map.get(int(pid), 'gray')  # Default to 'gray' if pid is not found

def get_color(unique_label, color_map):
    """
    Get color for a patient ID from color_map. If patient ID is not in 
    color_map, dynamically assign a new color and add it to color_map.
    
    Args:
        unique_label: Label containing patient ID (e.g., 'pid ...')
        color_map: Dictionary mapping patient IDs to colors
        
    Returns:
        Color for the patient ID
    """
    pid = unique_label.split(' ')[0]
    
    # First check if this pid is already in the color map (as string)
    if pid in color_map:
        return color_map[pid]
    
    # For backward compatibility, check if int(pid) is in color_map
    try:
        int_pid = int(pid)
        if int_pid in color_map:
            # Also store the string version for future lookups
            color_map[pid] = color_map[int_pid]
            return color_map[int_pid]
    except ValueError:
        # If pid can't be converted to int, that's fine
        pass
    
    # If we get here, this patient ID needs a new color
    # Generate a new distinct color - using standard matplotlib colors
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Use the length of color_map to determine the new color's index
    color_idx = len(color_map) % 10  # Cycle through 10 colors
    
    # Get a color from tab10 colormap
    new_color = plt.cm.tab10(color_idx)
    
    # Or use these predefined colors for more variety
    predefined_colors = [
        'red', 'blue', 'green', 'orange', 'purple', 
        'brown', 'pink', 'olive', 'cyan', 'magenta',
        'gold', 'limegreen', 'darkviolet', 'deepskyblue', 'crimson',
        'darkgreen', 'darkblue', 'darkorange', 'hotpink', 'teal'
    ]
    
    if len(color_map) < len(predefined_colors):
        new_color = predefined_colors[len(color_map)]
    
    # Add the new color to the color_map for future use
    color_map[pid] = new_color
    
    # Print for debugging
    print(f"Assigned new color to patient {pid}: {new_color}")
    
    return new_color

def plot_longitudinal_data(long_df, name):
    # color map
    global color_map
    # Create a unique color map for patient_id
    unique_patient_ids = long_df['patient_id'].unique()
    n=len(unique_patient_ids) # number of unique patient id's for color map
    #colors = plt.get_cmap('tab10').colors  # Use 'tab10' colormap for a set of distinct colors
    # colours to use:
    base_colors = ['red', 'cyan', 'yellow', 'magenta', 'brown', 'lightblue', 'orange']
    """
    def create_hex_color_map_from_cmap(cmap_name,n):
        cmap=plt.get_cmap(cmap_name)
        # Convert the colormap to a list of hex colors
        colors=cmap(np.linspace(0,1,n))
        hex_colors = [rgb2hex(color) for color in colors]
        return hex_colors
    
    def create_hex_color_map_custom(base_colors,n):
        # Create a custom colormap with the base colors
        cmap = LinearSegmentedColormap.from_list('tab20', base_colors, N=n)
        # Convert the colormap to a list of hex colors
        colors=cmap(np.linspace(0,1,n))
        hex_colors = [rgb2hex(color) for color in colors]
        return hex_colors
    """
    
    hex_color_map=create_hex_color_map_from_cmap('tab20',n)
    hex_color_map=create_hex_color_map_custom(base_colors,n)
    print(hex_color_map)
    

    #cmap=plt.get_cmap('tab10')
    #colors = cmap(np.linspace(0,1,n)) # n is the number of colours
    color_map = {pid: hex_color_map[i] for i, pid in enumerate(unique_patient_ids)}
    """
    # Function to determine color based on the beginning of 'unique_label'
    def get_color(unique_label):
        pid = unique_label.split(' ')[0]
        return color_map.get(int(pid), 'gray')  # Default to 'gray' if pid is not found
    """
    # PLOT GROUPED BY PATIENT
    plt.figure(figsize=(12, 8))
    #plt.bar(long_df['timepoint'], long_df['h_param_def'] )
    long_df['unique_label']=long_df['patient_id'].astype(str) + ' ' + long_df['timepoint'].astype(str)
    #bars=plt.bar(long_df.index, long_df[name], color=[get_color(label, color_map) for label in long_df['unique_label']])
    plt.xticks(long_df.index, long_df['unique_label'], rotation=90, fontsize=8)
    plt.xlabel('Timepoint')
    plt.ylabel(name)
    plt.title(f"{name} by patient")
    # Create legend
    handles = [plt.Line2D([0], [0], color=color_map[pid], lw=4) for pid in unique_patient_ids]
    labels = [str(pid) for pid in unique_patient_ids]
    plt.legend(handles, labels, title="Patient ID")
    plt.tight_layout()  # Adjust layout to fit labels
    # Show the plot
    if name == 'h_param_def' or name == 'h_param_ref':
        plt.ylim([0, 1.0])
    #plt.ylim([0, 1.0])
    plt.close()

  
    ## GROUPED BY TIMEPOINT
    # Create a pivot table for plotting
    pivot_df = long_df.pivot(index='timepoint', columns='patient_id', values=name)
    #print(pivot_df)
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_df.plot(kind='bar', ax=ax, color=[color_map[pid] for pid in pivot_df.columns]) # use same color map as before
    #ax = pivot_df.plot(kind='bar', figsize=(12, 8), colormap='tab10')
    plt.title(f"{name} grouped by timepoint")
    plt.xlabel('Timepoint')
    plt.ylabel(name)
    plt.legend(title='Patient ID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if name == 'h_param_def' or name == 'h_param_ref':
        plt.ylim([0, 1.0])
    plt.close()

    return 0


def map_timepoint_to_string(numeric_timepoint):
    """
    Convert a numeric timepoint to the closest string timepoint.
    
    Args:
        numeric_timepoint: Numeric value of the timepoint
        
    Returns:
        String representation of the closest timepoint
    """
    if pd.isna(numeric_timepoint):
        return None
    
    # Find the closest reference timepoint
    closest_idx = np.argmin([abs(numeric_timepoint - tp) for tp in timepoint_values])
    return timepoints[closest_idx]



# Main

# Load the ellipse data
data=pd.read_csv('Image_Processing_Scripts/ellipse_data.csv')
data['patient_id'] = data['patient_id'].astype(str)
print(data.columns)
area_data=pd.read_csv('Image_Processing_Scripts/area_data.csv')
area_data['patient_id'] = area_data['patient_id'].astype(str)
batch2_area_data=pd.read_csv('Image_Processing_Scripts/batch2_area_data.csv')
batch2_area_data['patient_id'] = batch2_area_data['patient_id'].astype(str)
print(area_data.columns)


# array of timepoints
timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
timepoint_values = [50, 336, 504, 2160, 4320, 8640, 17280]

# Convert numeric timepoints to strings
batch2_area_data['timepoint'] = batch2_area_data['timepoint'].apply(map_timepoint_to_string)

# Combine both data frames
combined_area_data=pd.concat([area_data,batch2_area_data],ignore_index=True)
area_data=combined_area_data

#reorder data to be all numbers together with timepoints in order
# Create an order for timepoint
area_data['timepoint_order'] = area_data['timepoint'].apply(lambda x: timepoints.index(x) if x in timepoints else 999)
# Sort the dataframe by patient_id, then by the position of timepoint in our list
area_data = area_data.sort_values(by=['patient_id', 'timepoint_order'])
area_data = area_data.drop('timepoint_order', axis=1) # remove sorting column
# save to csv for plotting
area_data[['patient_id', 'timepoint']].to_csv('patient_timepoint_matrix.csv', index=False)



# get possible patient IDs
patient_ids = area_data['patient_id'].unique()
print(patient_ids)
n = len(patient_ids) # number of colours for plotting






# Create global color map for plots
color_map={}

# h param plots
h_param_def_dict = get_dictionary(data, patient_ids, timepoints, subset_name='h_param_def')
h_param_ref_dict = get_dictionary(data, patient_ids, timepoints, subset_name='h_param_ref')

def_df = convert_dict_to_long_df(h_param_def_dict, timepoints, value_name='h_param_def')
ref_df = convert_dict_to_long_df(h_param_ref_dict, timepoints, value_name='h_param_ref')

plot_longitudinal_data(def_df, name='h_param_def')
plot_longitudinal_data(ref_df, name='h_param_ref')

# a param plots (note a_param_ref should be the same as a_param_def)
a_param_def_dict = get_dictionary(data, patient_ids, timepoints, subset_name='a_param_def')
#a_param_ref_dict = get_dictionary(data, patient_ids, timepoints, subset_name='a_param_ref')

defa_df = convert_dict_to_long_df(a_param_def_dict, timepoints, value_name='a_param_def')
#refa_df = convert_dict_to_long_df(a_param_ref_dict, timepoints, value_name='a_param_ref')

plot_longitudinal_data(defa_df, name='a_param_def')
#plot_longitudinal_data(refa_df, name='a_param_ref')

# Area plots [area calculated prior to rotation from orienting ellipse_data.csv '<>_<>ef_tr' columns in area_main.py]
area_diff_dict = get_dictionary(area_data, patient_ids, timepoints, subset_name='area_diff')
area_diff_df = convert_dict_to_long_df(area_diff_dict, timepoints, value_name='area_diff')
plot_longitudinal_data(area_diff_df, name='area_diff')

# Line plots superimposed

print(area_diff_df)
# convert timepoints to a numerical array 0 to len(timepoints)
timepoints_num = np.arange(len(timepoints))



set_publication_style()
# Set figure size before plotting
plt.figure(figsize=(12, 8))
# Set default color before plotting
default_color = 'gray'

# recall patient_ids = data['patient_id'].unique() have already been collected
# Add patient id x all timepoints to plot as scatter, create cubic spline between for each patient with more than 2 timepoints
for patient_id in patient_ids:
    patient_subset = area_diff_df[area_diff_df['patient_id'] == patient_id]
    print(patient_subset['area_diff'])
    # convert patient_subset['area_diff'] to a numpy array for plotting
    area_diff_subset = np.array(patient_subset['area_diff'])
    valid_indices = ~np.isnan(area_diff_subset)

    # skip patients with no valid measurements
    if not np.any(valid_indices):
        print(f'Patient {patient_id} has no valid area_diff measurements')
        continue

    # Find the earliest valid measurements
    earliest_valid_index=np.where(valid_indices)[0][0]
    first_area=area_diff_subset[earliest_valid_index]

    # normalise data to start at 0
    area_diff_subset = area_diff_subset - first_area
    area_diff_subset_valid = area_diff_subset[valid_indices]
    timepoints_num_valid = timepoints_num[valid_indices]

    """
    if not np.isnan(area_diff_subset[0]):
        first_area = area_diff_subset[0]
        area_diff_subset = area_diff_subset - first_area
    elif np.isnan(area_diff_subset[0]):
        if not np.isnan(area_diff_subset[1]):
            first_area = area_diff_subset[1]
            area_diff_subset = area_diff_subset - first_area
    else:
        print('First two area_diff values are NaN')
        continue
    area_diff_subset_valid = area_diff_subset[valid_indices]
    timepoints_num_valid = timepoints_num[valid_indices]
    """

    # Create a smooth line using spline interpolation
    # 1. interpolate
    interpolator = interp1d(timepoints_num_valid, area_diff_subset_valid, kind='linear')
    x_fine = np.linspace(timepoints_num_valid.min(), timepoints_num_valid.max(), 100)
    y_fine = interpolator(x_fine)
    print(f"interpolator: {interpolator}")
    #plt.scatter(x_fine, y_fine, label=patient_id, color='gray')
    # 2. Create cubic spline
    cs=CubicSpline(timepoints_num_valid, area_diff_subset_valid, bc_type='natural')
    #cs=CubicSpline(x_fine, y_fine, bc_type='natural')
    x_smooth = x_fine
    y_smooth = cs(x_smooth)
    print(f"cs: {cs}")
    """
    # 2. create univariate spline
    spline = UnivariateSpline(x_fine, y_fine)
    x_smooth = np.linspace(timepoints_num_valid.min(), timepoints_num_valid.max(), 100)
    y_smooth = spline(x_smooth)
    """

    color = color_map.get(patient_id, default_color)
    print(f"Color for patient {patient_id}: {color}")
    #if patient_id == 20174:
    plt.plot(x_smooth, y_smooth, label=patient_id, color=color)
    plt.scatter(timepoints_num_valid, area_diff_subset_valid, color=color, s=20, alpha=0.5)

#plt.figure(figsize=(12, 8))
#print(f"color_map: {color_map}")
plt.xlim([0, len(timepoints)-1])
#plt.legend(title='Patient ID')
plt.xlabel('Time')

# instead of 0-6, use timepoints
plt.xticks(timepoints_num, timepoints)
plt.ylabel('Area Change [mm$^2$]')
plt.title('Area Change Over Time')
#position legend outside of plot
plt.tight_layout()
plt.legend(title='Patient ID', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('Image_Processing_Scripts/plots/area_change_longitudinal.png', bbox_inches='tight')
plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/area_change_longitudinal.pdf', bbox_inches='tight', dpi=300)
plt.close()
#plt.show()

