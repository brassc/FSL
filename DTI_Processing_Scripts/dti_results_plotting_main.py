import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from matplotlib.lines import Line2D
import matplotlib as mpl
import statsmodels.formula.api as smf
import seaborn as sns
import sys
import os
import statsmodels.stats.multitest as smm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Image_Processing_Scripts.longitudinal_main import map_timepoint_to_string
from Image_Processing_Scripts.set_publication_style import set_publication_style



import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import PackageNotInstalledError

# Activate pandas to R conversion
pandas2ri.activate()

# First, install required R packages if not already installed
utils = importr('utils')
utils.chooseCRANmirror(ind=1)  # Choose the first CRAN mirror

# Check and install required packages
packnames = ('lme4', 'emmeans', 'pbkrtest', 'lmerTest')
names_to_install = [x for x in packnames if not rpy2.robjects.packages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# Import necessary R packages
base = importr('base')
lme4 = importr('lme4')
emmeans = importr('emmeans')


def create_hex_color_map_from_cmap(cmap_name, n):
    """Create a list of hex colors from a colormap."""
    cmap = plt.get_cmap(cmap_name)
    # Convert the colormap to a list of hex colors
    colors = cmap(np.linspace(0, 1, n))
    hex_colors = [rgb2hex(color) for color in colors]
    return hex_colors

def create_hex_color_map_custom(base_colors, n):
    """Create a list of hex colors from a list of base colors."""
    # Create a custom colormap with the base colors
    cmap = LinearSegmentedColormap.from_list('custom', base_colors, N=n)
    # Convert the colormap to a list of hex colors
    colors = cmap(np.linspace(0, 1, n))
    hex_colors = [rgb2hex(color) for color in colors]
    return hex_colors

def get_color(patient_id, color_map):
    """
    Get color for a patient ID from color_map. If patient ID is not in 
    color_map, dynamically assign a new color and add it to color_map.
    """
    pid = str(patient_id)
    
    # First check if this pid is already in the color map (as string)
    if pid in color_map:
        return color_map[pid]
    
    # If we get here, this patient ID needs a new color
    # Predefined colors for variety
    predefined_colors = [
        'red', 'blue', 'green', 'orange', 'purple', 
        'brown', 'pink', 'olive', 'cyan', 'magenta',
        'gold', 'limegreen', 'darkviolet', 'deepskyblue', 'crimson',
        'darkgreen', 'darkblue', 'darkorange', 'hotpink', 'teal'
    ]
    
    if len(color_map) < len(predefined_colors):
        new_color = predefined_colors[len(color_map)]
    else:
        # Use tab10 colormap for additional colors
        color_idx = len(color_map) % 10
        new_color = plt.cm.tab10(color_idx)
    
    # Add the new color to the color_map for future use
    color_map[pid] = new_color
    
    # Print for debugging
    print(f"Assigned new color to patient {pid}: {new_color}")
    
    return new_color

def plot_all_rings_combined(df, parameter, num_bins=5, save_path=None):
    """
    Plot all rings data on a single figure with days since injury on x-axis.
    Each patient has a unique color, and different ring types use different markers/line styles.
    
    Args:
        df: DataFrame with the data
        save_path: Path to save the figure (optional)
    """
    set_publication_style()
    
    # Get unique patient IDs
    patient_ids = df['patient_id'].unique()
    
    # Create color map for patients
    patient_color_map = {}
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 10))
    fig2, ax2 = plt.subplots(figsize=(15, 10))  # Timepoint category plot
    
    # Define markers for different ring types to distinguish them
    # Define markers for different ring types to distinguish them (without parameter prefixes)
    
    if num_bins == 5:
        ring_markers = {
            'anterior_ring_1': 'o',
            'anterior_ring_2': 's',
            'anterior_ring_3': '^',
            'anterior_ring_4': 'D',
            'anterior_ring_5': 'p',
            'posterior_ring_1': 'o',
            'posterior_ring_2': 's',
            'posterior_ring_3': '^',
            'posterior_ring_4': 'D',
            'posterior_ring_5': 'p',
        }
    elif num_bins == 10:
        ring_markers = {
            'anterior_ring_1': 'o',
            'anterior_ring_2': 's',
            'anterior_ring_3': '^',
            'anterior_ring_4': 'D',
            'anterior_ring_5': 'p',
            'anterior_ring_6': 'X',
            'anterior_ring_7': 'H',
            'anterior_ring_8': 'P',
            'anterior_ring_9': 'v',
            'anterior_ring_10': '<',
            'posterior_ring_1': 'o',
            'posterior_ring_2': 's',
            'posterior_ring_3': '^',
            'posterior_ring_4': 'D',
            'posterior_ring_5': 'p',
            'posterior_ring_6': 'X',
            'posterior_ring_7': 'H',
            'posterior_ring_8': 'P',
            'posterior_ring_9': 'v',
            'posterior_ring_10': '<',
        }
    
    
    # Define line styles for anterior vs posterior
    line_styles = {
        'anterior': '-',
        'posterior': '--',
    }
    
    # Define alpha values for baseline vs current
    alpha_values = {
        'baseline': 0.4,
        'current': 0.9,
    }
    
    # For legend
    legend_handles = []
    legend_labels = []
    patient_added_to_legend = set()
    ring_type_added_to_legend = set()

    # fig2 plotting
    # Define timepoint order 
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    
    # Create a mapping for timepoints to numerical positions
    timepoint_spacing = 3.0  # Adjust this value to control the gutter width
    # timepoint_positions = {tp: i * timepoint_spacing for i, tp in enumerate(timepoint_order)}

    # We'll use a simple numeric sequence as the base positions
    base_positions = list(range(len(timepoint_order)))
    
    # Apply spacing to create the actual positions
    spaced_positions = [pos * timepoint_spacing for pos in base_positions]
    
    # Create mapping from timepoint names to spaced positions
    timepoint_positions = {tp: spaced_positions[i] for i, tp in enumerate(timepoint_order)}

    # Create a numerical index for each patient ID to determine offset
    patient_indices = {pid: i for i, pid in enumerate(sorted(patient_ids))}
    total_patients = len(patient_ids)

    # Calculate the width to allocate for each timepoint
    timepoint_width = 0.8  # Width allocated for each timepoint
    
    # All possible column names for rings
    ring_cols = []
    for region in ['anterior', 'posterior']:
        for ring_num in range(1, num_bins + 1):
            ring_cols.append(f'{parameter}_{region}_ring_{ring_num}')
            ring_cols.append(f'{parameter}_baseline_{region}_ring_{ring_num}')
    
    #print(f"Ring columns: {ring_cols}")
    
    # For the timepoint plot, first collect all data points per timepoint and patient
    # to determine better offset calculations
    timepoint_data_collection = {}
    
    # First pass: collect data by timepoint and patient
    for pid in patient_ids:
        patient_data = df[df['patient_id'] == pid].copy()
        if patient_data.empty or 'timepoint' not in patient_data.columns:
            continue
            
        for tp in timepoint_order:
            if tp not in timepoint_data_collection:
                timepoint_data_collection[tp] = {}
                
            tp_data = patient_data[patient_data['timepoint'] == tp]
            if not tp_data.empty:
                timepoint_data_collection[tp][pid] = tp_data
    
    # For each timepoint, determine which patients have data and assign positions
    timepoint_patient_positions = {}
    for tp in timepoint_order:
        if tp in timepoint_data_collection:
            # Get patients that have data for this timepoint
            patients_with_data = list(timepoint_data_collection[tp].keys())
            
            # If there are patients with data
            if patients_with_data:
                # Sort patients for consistent ordering
                patients_with_data.sort()
                
                # Create evenly distributed positions for this timepoint
                timepoint_width = 0.8  # Width allocated for each timepoint
                
                # Assign positions
                if len(patients_with_data) > 1:
                    for i, pid in enumerate(patients_with_data):
                        # Calculate offset to center the distribution
                        offset = (i - (len(patients_with_data) - 1) / 2) * (timepoint_width / len(patients_with_data))
                        
                        if tp not in timepoint_patient_positions:
                            timepoint_patient_positions[tp] = {}
                        timepoint_patient_positions[tp][pid] = offset
                else:
                    # Just one patient, no offset needed
                    if tp not in timepoint_patient_positions:
                        timepoint_patient_positions[tp] = {}
                    timepoint_patient_positions[tp][patients_with_data[0]] = 0




    # Loop through patients
    for pid in patient_ids:
        patient_data = df[df['patient_id'] == pid].copy()
        # Skip patients with no data
        if patient_data.empty:
            continue
            
        # Sort by days since injury
        patient_data = patient_data.sort_values('Days_since_injury')
        
        # Get color for this patient
        patient_color = get_color(pid, patient_color_map)

        # Calculate patient-specific offset within timepoint
        if total_patients > 1:
            # Scale to fit within the timepoint width
            patient_offset = (patient_indices[pid] - (total_patients - 1) / 2) * (timepoint_width / total_patients)
        else:
            patient_offset = 0
        
        # Loop through all ring columns
        for col in ring_cols:
            if col not in patient_data.columns:
                continue
                
            # Skip columns with all NaN values
            if patient_data[col].isna().all():
                continue
                
            # Determine if this is baseline or current data
            is_baseline = 'baseline' in col
            
            # Determine region (anterior or posterior)
            region = 'anterior' if 'anterior' in col else 'posterior'
            
            # Extract ring number
            ring_num = int(col.split('_')[-1])
            
            # Determine marker and line style
            ring_key = f"{region}_ring_{ring_num}"
            marker = ring_markers[ring_key]
            line_style = line_styles[region]
            alpha = alpha_values['baseline' if is_baseline else 'current']
            
            # Filter out NaN values
            valid_data = patient_data[~patient_data[col].isna()]
            if len(valid_data) == 0:
                continue
                
            # Plot scatter points
            scatter = ax.scatter(
                valid_data['Days_since_injury'], 
                valid_data[col],
                color=patient_color,
                marker=marker,
                s=30 if is_baseline else 50,
                alpha=alpha,
                edgecolors='black' if not is_baseline else None,
                linewidths=0.5 if not is_baseline else 0
            )

            # PLOT 2: Timepoints (discrete)
            # Only plot if timepoint data is available

            # PLOT 2: Timepoints (discrete) with improved patient-specific offset
            if 'timepoint' in valid_data.columns:
                for _, row in valid_data.iterrows():
                    timepoint = row['timepoint']
                    if pd.notna(timepoint) and timepoint in timepoint_positions:
                        # Use the pre-calculated patient position for this timepoint if available
                        if (timepoint in timepoint_patient_positions and 
                            pid in timepoint_patient_positions[timepoint]):
                            patient_offset = timepoint_patient_positions[timepoint][pid]
                        else:
                            patient_offset = 0  # Fallback if position not calculated
                            
                        # Base position + patient-specific offset
                        x_pos = timepoint_positions[timepoint] + patient_offset
                        
                        ax2.scatter(
                            x_pos,
                            row[col],
                            color=patient_color,
                            marker=marker,
                            s=30 if is_baseline else 50,
                            alpha=alpha,
                            edgecolors='black' if not is_baseline else None,
                            linewidths=0.5 if not is_baseline else 0
                        )
            # if 'timepoint' in valid_data.columns:
            #     # Convert timepoints to numeric positions
            #     valid_data['timepoint_pos'] = valid_data['timepoint'].map(timepoint_positions)
                
            #     # Only use data points with valid timepoints
            #     timepoint_data = valid_data.dropna(subset=['timepoint_pos'])
                
            #     if not timepoint_data.empty:
            #         ax2.scatter(
            #             timepoint_data['timepoint_pos'],
            #             timepoint_data[col],
            #             color=patient_color,
            #             marker=marker,
            #             s=30 if is_baseline else 50,
            #             alpha=alpha,
            #             edgecolors='black' if not is_baseline else None,
            #             linewidths=0.5 if not is_baseline else 0
            #         )

            
            # Add to legend (only add each patient and ring type once)
            if pid not in patient_added_to_legend:
                legend_handles.append(Line2D([0], [0], color=patient_color, marker='o', linestyle='-', 
                                           markersize=6, label=f'Patient {pid}'))
                legend_labels.append(f'Patient {pid}')
                patient_added_to_legend.add(pid)
            
            # Create a unique identifier for this ring type
            ring_type_id = f"{region}_ring_{ring_num}"
            if ring_type_id not in ring_type_added_to_legend:
                legend_handles.append(Line2D([0], [0], color='black', marker=marker, linestyle=line_style, 
                                           markersize=6))
                legend_labels.append(f"{region.capitalize()} Ring {ring_num}")
                ring_type_added_to_legend.add(ring_type_id)
            
            # If we have enough points (>=2), plot a smooth curve
            if not is_baseline and len(valid_data) >= 3:
                x_data = valid_data['Days_since_injury'].values
                y_data = valid_data[col].values
                
                try:
                    # Create smooth x values for the curve
                    x_smooth = np.linspace(min(x_data), max(x_data), 100)
                    
                    # Use cubic spline for smoothing
                    cs = CubicSpline(x_data, y_data, bc_type='natural')
                    y_smooth = cs(x_smooth)
                    
                    # Plot the smooth curve
                    # ax.plot(x_smooth, y_smooth, color=patient_color, alpha=0.7, 
                    #        linestyle=line_style, linewidth=1.0)
                except Exception as e:
                    print(f"Could not create smooth curve for patient {pid}, {col}: {e}")
    
    # Add any missing ring types to the legend when num_bins=10
    if num_bins == 10:
        for region in ['anterior', 'posterior']:
            for ring_num in range(1, 11):
                ring_type_id = f"{region}_ring_{ring_num}"
                if ring_type_id not in ring_type_added_to_legend:
                    marker = ring_markers[ring_type_id]
                    line_style = line_styles[region]
                    legend_handles.append(Line2D([0], [0], color='black', marker=marker, linestyle=line_style, 
                                            markersize=6))
                    legend_labels.append(f"{region.capitalize()} Ring {ring_num}")
                    ring_type_added_to_legend.add(ring_type_id)


    # Add baseline vs current markers to legend
    legend_handles.append(plt.Line2D([0], [0], color='black', marker='o', linestyle='None', 
                               markersize=6, alpha=alpha_values['current'], 
                               markeredgecolor='black', markeredgewidth=0.5))
    legend_labels.append('Current')
    
    legend_handles.append(plt.Line2D([0], [0], color='black', marker='o', linestyle='None', 
                               markersize=6, alpha=alpha_values['baseline']))
    legend_labels.append('Baseline')
    
    # # Add anterior vs posterior line styles to legend
    # legend_handles.append(Line2D([0], [0], color='black', linestyle=line_styles['anterior']))
    # legend_labels.append('Anterior')
    
    # legend_handles.append(Line2D([0], [0], color='black', linestyle=line_styles['posterior']))
    # legend_labels.append('Posterior')
    
    # Set labels and title
    ax.set_xlabel('Days Since Injury')
    ax.set_ylabel('Fractional Anisotropy (FA)')
    ax.set_title('Fractional Anisotropy: All Rings Over Time')
    
    # Set x-axis to start at 0 and find the maximum value for upper limit
    max_days = 0
    for pid in patient_ids:
        patient_data = df[df['patient_id'] == pid]
        if not patient_data.empty and 'Days_since_injury' in patient_data.columns:
            patient_max = patient_data['Days_since_injury'].max()
            if not pd.isna(patient_max) and patient_max > max_days:
                max_days = patient_max
    
    # Add a small margin on the right (5%)
    margin = 0.05 * max_days
    ax.set_xlim(0, max_days + margin)


    # Set labels and title for Plot 2 (Timepoint category)
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel(f'{parameter.upper()}')
    ax2.set_title(f'{parameter.upper()}: All Rings Over Time (Discrete Timepoints)')
    
    # Set x-ticks for Plot 2 to be the timepoint names
    ax2.set_xticks(spaced_positions)
    ax2.set_xticklabels(timepoint_order, rotation=45)
    # Set x-axis limits for Plot 2 to include some padding
    # min_pos = min(timepoint_positions.values()) - 0.5 * timepoint_spacing
    # max_pos = max(timepoint_positions.values()) + 0.5 * timepoint_spacing
    ax2.set_xlim(-0.5, spaced_positions[-1] + 0.5)

        # Add vertical lines at each timepoint to further separate them visually
    for pos in spaced_positions:
        ax2.axvline(x=pos - 0.4, color='lightgray', linestyle='-', alpha=0.3)
        ax2.axvline(x=pos + 0.4, color='lightgray', linestyle='-', alpha=0.3)
    
    # Add legend to both plots
    for ax in [ax, ax2]:
        ax.legend(handles=legend_handles, labels=legend_labels, loc='center left', 
                 bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    
    # Adjust layout for both plots
    for figure in [fig, fig2]:
        figure.tight_layout()
        plt.figure(figure.number)
        plt.subplots_adjust(right=0.8)
    
    # Save figures if path is provided
    if save_path:
        # Create file paths by inserting a suffix before the extension
        base, ext = os.path.splitext(save_path)
        days_path = f"{base}_days{ext}"
        timepoint_path = f"{base}_timepoint{ext}"
        
        fig.savefig(days_path, bbox_inches='tight')
        fig2.savefig(timepoint_path, bbox_inches='tight')
        print(f"Figures saved to {days_path} and {timepoint_path}")
    
    return (fig, ax), (fig2, ax2)


def process_timepoint_data(input_file_location):
    """
    Process patient timepoint data by standardizing timepoint values and sorting results.
    
    Args:
        df: Input pandas DataFrame containing patient data
        
    Returns:
        Processed pandas DataFrame with standardized timepoints and proper sorting
    """
    # Check if harmonised data file supplied
    if 'harmonised' in input_file_location:
        print('Harmonised data file supplied')
    else:
        print('Harmonised data file not supplied, please check')
        input('Press Enter to continue or Ctrl+C to exit')

        #return None

    df=pd.read_csv(input_file_location)
    df['patient_id'] = df['patient_id'].astype(str)
    timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    string_mask = df['timepoint'].isin(timepoints)
    numeric_mask = ~string_mask & df['timepoint'].apply(lambda x: pd.notnull(pd.to_numeric(x, errors='coerce')))
    # Convert numeric values to their appropriate string representations
    for idx in df[numeric_mask].index:
        try:
            numeric_value = float(df.loc[idx, 'timepoint'])
            df.loc[idx, 'timepoint'] = map_timepoint_to_string(numeric_value)
        except (ValueError, TypeError):
            continue
    #df['timepoint'] = df['timepoint'].apply(map_timepoint_to_string) # convert to string category
    #print(f"df:\n{df}")
    # Save original timepoint values before recategorization (if needed for reference)
    df['original_timepoint'] = df['timepoint']
    # Now apply the timepoint recategorization based on Days_since_injury
    # Define the ranges and labels for recategorization
    ranges = [(0, 2), (2, 8), (8, 42), (42, 179), (179, 278), (278, 540), (540, 500000)]
    labels = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    # Only recategorize rows where Days_since_injury is not null
    mask = df['Days_since_injury'].notnull()
    if mask.any():
        df.loc[mask, 'timepoint'] = pd.cut(
            df.loc[mask, 'Days_since_injury'],
            bins=[0, 2, 8, 42, 179, 278, 540, 500000],
            labels=labels
        )
    # sort data according to patient id then timepoint
    def get_sort_key(patient_id):
        try:
            return (0, int(patient_id)) # Numeric IDs first, sorted numerically
        except ValueError:
            return (1, patient_id) # Alphanumeric IDs second, sorted alphabetically
    # Create a sort key column
    df['sort_key'] = df['patient_id'].apply(get_sort_key)
    df['timepoint_order'] = df['timepoint'].apply(lambda x: timepoints.index(x) if x in timepoints else 999)
    # Sort the dataframe by patient_id, then by the position of timepoint in our list
    df = df.sort_values(by=['sort_key', 'timepoint_order'])
    df = df.drop(['sort_key', 'timepoint_order'], axis=1) # remove sorting column
    
    return df


# Set publication style for matplotlib
set_publication_style()

if __name__ == '__main__':
    print('running dti_results_plotting_main.py')
    
    # Load the (harmonised) data 

    # not harmonised data test
    data_5x4vox_not_harmonised_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_notharmonised.csv'
    data_5x4vox_not_harmonised=process_timepoint_data(input_file_location=data_5x4vox_not_harmonised_filename)
    plot_all_rings_combined(df=data_5x4vox_not_harmonised, parameter='fa', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_not_harmonised.png')

    sys.exit()
    data_5x4vox_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_harmonised.csv'
    data_5x4vox=process_timepoint_data(input_file_location=data_5x4vox_filename)
    # Now data_5x4vox has been recategorized based on Days_since_injury, exactly the same as the deformation analysis
    plot_all_rings_combined(df=data_5x4vox, parameter='fa', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_filtered.png')
    #plot_all_rings_combined(df=data_5x4vox, parameter='md', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_filtered_md.png')

    # Load the (harmonised) data
    #data_10x4vox=pd.read_csv('DTI_Processing_Scripts/merged_data_10x4vox_NEW_filtered_harmonised.csv')
    #data_10x4vox=process_timepoint_data(df=data_10x4vox)
    # Now data_10x4vox has been recategorized based on Days_since_injury, exactly the same as the deformation analysis
    #plot_all_rings_combined(df=data_10x4vox, parameter='fa', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_10x4vox_filtered.png')
    #plot_all_rings_combined(df=data_10x4vox, parameter='md', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_10x4vox_filtered_md.png')



    ## Plot the FA data. anterior, posterior, baseline_anterior, baseline_posterior on patient by patient basis across time



