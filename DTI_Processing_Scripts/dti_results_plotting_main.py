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
import re
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


def plot_metric_difference(df, parameter, region, save_path=None, plot_type='box', group_by='region'):
    """
    Plot the difference between baseline and current values for a specific parameter and region across ROIs.
    
    Args:
        df: DataFrame with the data
        parameter: The metric to plot (e.g., 'fa', 'md')
        region: Region to analyze ('anterior', 'posterior', or 'both')
        save_path: Path to save the figure (optional)
        plot_type: Type of plot ('box', 'violin', 'scatter', or 'strip')
        group_by: How to group the data ('region' or 'timepoint')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re
    import seaborn as sns
    
    # Set publication style (assuming this function exists in your environment)
    try:
        set_publication_style()
    except NameError:
        plt.style.use('seaborn-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.figsize': (12, 8)
        })
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique timepoints
    timepoints = df['timepoint'].unique()  
    
    # Define the desired order for timepoints
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Filter to only include timepoints present in your data
    present_timepoints = [tp for tp in timepoint_order if tp in timepoints]
        
    # Create color map for timepoints based on the ordered list
    cmap = plt.cm.get_cmap('viridis', len(present_timepoints))
    timepoint_colors = {tp: cmap(i) for i, tp in enumerate(present_timepoints)}
    
    # Determine regions to analyze
    regions_to_analyze = []
    if region == 'both':
        regions_to_analyze = ['anterior', 'posterior']
    else:
        regions_to_analyze = [region]
    
    # Create an empty DataFrame to store the difference data
    diff_df = pd.DataFrame()


    
    # Process each region
    for curr_region in regions_to_analyze:
        # Find all columns matching the parameter and current region
        parameter = parameter.lower()
        current_pattern = f"{parameter}_{curr_region}_ring_(\\d+)"
        baseline_pattern = f"{parameter}_baseline_{curr_region}_ring_(\\d+)"
        
        current_cols = [col for col in df.columns if re.match(current_pattern, col.lower())]
        baseline_cols = [col for col in df.columns if re.match(baseline_pattern, col.lower())]
        
        if not current_cols or not baseline_cols:
            continue
        
        # Extract ring numbers and create mapping between current and baseline columns
        ring_mapping = {}
        ring_numbers = []
        
        for col in current_cols:
            match = re.search(r'_ring_(\d+)$', col)
            if match:
                ring_num = int(match.group(1))
                ring_numbers.append(ring_num)
                baseline_col = f"{parameter}_baseline_{curr_region}_ring_{ring_num}"
                if baseline_col in baseline_cols:
                    ring_mapping[col] = baseline_col
        
        # Sort unique ring numbers
        unique_ring_numbers = sorted(set(ring_numbers))
        
        # Calculate differences for each timepoint and ring
        for tp in timepoints:
            tp_data = df[df['timepoint'] == tp]
            
            for current_col, baseline_col in ring_mapping.items():
                # Extract ring number
                match = re.search(r'_ring_(\d+)$', current_col)
                if not match:
                    continue
                    
                ring_num = int(match.group(1))
                
                # Get the difference (baseline - current)
                current_values = tp_data[current_col]
                baseline_values = tp_data[baseline_col]
                
                # Filter for rows where both values are not NaN
                valid_indices = current_values.notna() & baseline_values.notna()
                
                if valid_indices.sum() > 0:
                    # Calculate difference
                    diff_values = baseline_values[valid_indices] - current_values[valid_indices]
                    
                    # Create a temporary DataFrame for this difference
                    temp_df = pd.DataFrame({
                        'difference': diff_values,
                        'ring': [f"Ring {ring_num}"] * len(diff_values),
                        'timepoint': [tp] * len(diff_values),
                        'region': [curr_region.capitalize()] * len(diff_values)
                    })
                    
                    # Append to the main difference DataFrame
                    diff_df = pd.concat([diff_df, temp_df])
    
    # Check if we have data to plot
    if len(diff_df) == 0:
        ax.text(0.5, 0.5, f"No matching data found for {parameter.upper()} in the specified region(s)", 
                ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # After creating diff_df but before plotting
    diff_df['timepoint'] = pd.Categorical(
        diff_df['timepoint'], 
        categories=present_timepoints,
        ordered=True
    )
    
    # Create the appropriate plot based on plot_type and group_by
    if plot_type == 'box':
        # Create a box plot
        if group_by == 'region' and len(regions_to_analyze) > 1:
            # Group by region (anterior vs posterior)
            sns.boxplot(
                data=diff_df,
                x='ring', y='difference', hue='region',
                palette='Set1',
                ax=ax
            )
            
            # Add points on top of the boxes
            sns.stripplot(
                data=diff_df,
                x='ring', y='difference', hue='region',
                palette='Set1',
                dodge=True,
                alpha=0.7,
                size=4,
                edgecolor='black',
                linewidth=0.5,
                ax=ax,
                legend=False
            )
        else:
            # Group by timepoint
            sns.boxplot(
                data=diff_df,
                x='ring', y='difference', hue='timepoint',
                palette=timepoint_colors,
                ax=ax
            )
            
            # Add points on top of the boxes
            sns.stripplot(
                data=diff_df,
                x='ring', y='difference', hue='timepoint',
                palette=timepoint_colors,
                dodge=True,
                alpha=0.7,
                size=4,
                edgecolor='black',
                linewidth=0.5,
                ax=ax,
                legend=False
            )
    
    elif plot_type == 'violin':
        # Create a violin plot
        if group_by == 'region' and len(regions_to_analyze) > 1:
            sns.violinplot(
                data=diff_df,
                x='ring', y='difference', hue='region',
                palette='Set1',
                split=True, inner='quartile',
                ax=ax
            )
        else:
            sns.violinplot(
                data=diff_df,
                x='ring', y='difference', hue='timepoint',
                palette=timepoint_colors,
                split=True, inner='quartile',
                ax=ax
            )
    
    elif plot_type == 'scatter':
        # Create a scatter plot with jitter
        if group_by == 'region' and len(regions_to_analyze) > 1:
            # Use region for color
            for region_name in diff_df['region'].unique():
                region_data = diff_df[diff_df['region'] == region_name]
                
                # Get unique rings
                rings = region_data['ring'].unique()
                
                for i, ring in enumerate(sorted(rings)):
                    ring_data = region_data[region_data['ring'] == ring]
                    
                    # Create jitter
                    x = np.full(len(ring_data), i)
                    x = x + np.random.normal(0, 0.05, size=len(x))
                    
                    # Color based on region
                    color = 'C0' if region_name == 'Anterior' else 'C1'
                    
                    # Plot with slight jitter on x-axis
                    ax.scatter(
                        x, ring_data['difference'], 
                        color=color,
                        s=50, alpha=0.7, edgecolors='black', linewidths=0.5,
                        label=region_name if i == 0 else ""  # Only add to legend once
                    )
        else:
            # Use timepoint for color
            for tp in timepoints:
                tp_data = diff_df[diff_df['timepoint'] == tp]
                
                if len(tp_data) > 0:
                    # Add jitter to the categorical x positions
                    rings = tp_data['ring'].unique()
                    for i, ring in enumerate(sorted(rings)):
                        ring_data = tp_data[tp_data['ring'] == ring]
                        
                        # Create jitter
                        x = np.full(len(ring_data), i)
                        x = x + np.random.normal(0, 0.05, size=len(x))
                        
                        # Plot with slight jitter on x-axis
                        ax.scatter(
                            x, ring_data['difference'], 
                            color=timepoint_colors[tp],
                            s=50, alpha=0.7, edgecolors='black', linewidths=0.5,
                            label=tp if i == 0 else ""  # Only add to legend once
                        )
        
        # Set x-ticks
        ax.set_xticks(range(len(diff_df['ring'].unique())))
        ax.set_xticklabels(sorted(diff_df['ring'].unique()))
        
    elif plot_type == 'strip':
        # Create a strip plot
        if group_by == 'region' and len(regions_to_analyze) > 1:
            sns.stripplot(
                data=diff_df,
                x='ring', y='difference', hue='region',
                palette='Set1',
                dodge=True,
                alpha=0.7,
                size=8,
                edgecolor='black',
                linewidth=0.5,
                ax=ax
            )
        else:
            sns.boxplot(
                data=diff_df,
                x='ring', y='difference',
                color='lightgray',
                ax=ax,
                showfliers=False
            )
            sns.stripplot(
                data=diff_df,
                x='ring', y='difference', hue='timepoint',
                palette={tp: timepoint_colors[tp] for tp in timepoint_order if tp in diff_df['timepoint'].unique()},
                dodge=True,
                alpha=0.7,
                size=8,
                edgecolor='black',
                linewidth=0.5,
                ax=ax
            )
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    if group_by == 'region' and len(regions_to_analyze) > 1:
        ax.legend(by_label.values(), by_label.keys(), title='Region')
    else:
        ax.legend(by_label.values(), by_label.keys(), title='Timepoint')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Set axis labels and title
    ax.set_xlabel('Ring Number')
    ax.set_ylabel(f'{parameter.upper()} Difference (Control Side - Craniectomy Side)')
    
    # Set appropriate y-limits based on parameter
    if parameter.lower() == 'fa':
        # FA values typically range from 0 to 1, differences will be smaller
        ax.set_ylim(-0.3, 0.3)
    elif parameter.lower() == 'md':
        # MD values are typically very small (e.g., 0.0001 to 0.003)
        ax.set_ylim(-0.001, 0.001)
    
    # Format region name for title
    if region == 'both':
        region_title = "Anterior and Posterior Regions"
    else:
        region_title = f"{region.capitalize()} Region"
    
    # Format group name for title
    if group_by == 'region' and len(regions_to_analyze) > 1:
        group_title = "by Region"
    else:
        group_title = "by Timepoint"
        
    ax.set_title(f'{parameter.upper()} Differences (Control Side - Craniectomy Side) for {region_title} {group_title}')
    
    if 'wm' in save_path:
        ax.set_title(f'{parameter.upper()} Differences (Control Side - Craniectomy Side) in White Matter for {region_title} {group_title}')


    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

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
    if 'fa' in parameter:
        ax.set_ylim(0, 0.5)
    else:
        ax.set_ylim(0, 0.0035)
    


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
    if 'fa' in parameter:
        ax.set_ylim(0, 0.5)
    else:
        ax.set_ylim(0, 0.0035)
    

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


def plot_metric_roi(df, parameter, region_type, save_path=None, plot_type='scatter'):
    """
    Plot the metric for a specific region type across all patients.
    
    Args:
        df: DataFrame with the data
        parameter: The metric to plot (e.g., 'fa', 'md')
        region_type: Type of region ('anterior', 'posterior', 'baseline_anterior', 'baseline_posterior')
        save_path: Path to save the figure (optional)
        plot_type: Type of plot ('scatter', 'box', or 'violin')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re
    import seaborn as sns
    
    # Set publication style
    set_publication_style()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique timepoints
    timepoints = df['timepoint'].unique()  
    
    # Define the desired order for timepoints
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Filter to only include timepoints present in your data
    present_timepoints = [tp for tp in timepoint_order if tp in timepoints]
        
    # Create color map for timepoints based on the ordered list
    cmap = plt.cm.get_cmap('viridis', len(present_timepoints))
    timepoint_colors = {tp: cmap(i) for i, tp in enumerate(present_timepoints)}
    
    # Find all columns matching the parameter and region type
    parameter = parameter.lower()
    pattern = f"{parameter}_{region_type}_ring_(\\d+)"
    matching_cols = [col for col in df.columns if re.match(pattern, col.lower())]
    
    if not matching_cols:
        ax.text(0.5, 0.5, f"No data found for {parameter.upper()} in {region_type} rings", 
               ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Extract ring numbers and sort columns by ring number
    ring_numbers = []
    for col in matching_cols:
        match = re.search(r'_ring_(\d+)$', col)
        if match:
            ring_numbers.append(int(match.group(1)))
        else:
            ring_numbers.append(0)
            
    # Sort columns by ring number
    sorted_indices = np.argsort(ring_numbers)
    sorted_cols = [matching_cols[i] for i in sorted_indices]
    sorted_ring_numbers = [ring_numbers[i] for i in sorted_indices]
    
    # Prepare data for plotting
    plot_df = pd.DataFrame()
    
    for i, col in enumerate(sorted_cols):
        ring_num = sorted_ring_numbers[i]
        
        for tp in timepoints:
            tp_data = df[df['timepoint'] == tp]
            values = tp_data[col].dropna()
            
            if len(values) > 0:
                # Create a temporary DataFrame for this ring/timepoint combination
                temp_df = pd.DataFrame({
                    'value': values,
                    'ring': [f"Ring {ring_num}"] * len(values),
                    'timepoint': [tp] * len(values)
                })
                
                # Append to the main plotting DataFrame
                plot_df = pd.concat([plot_df, temp_df])
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f"No data found for {parameter.upper()} in {region_type} rings", 
               ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Create appropriate plot based on plot_type
    if plot_type == 'scatter':
        # Create a scatter plot with jitter
        for tp in timepoints:
            tp_data = plot_df[plot_df['timepoint'] == tp]
            
            if len(tp_data) > 0:
                # Add jitter to the categorical x positions
                rings = tp_data['ring'].unique()
                for i, ring in enumerate(sorted(rings)):
                    ring_data = tp_data[tp_data['ring'] == ring]
                    
                    # Create jitter
                    x = np.full(len(ring_data), i)
                    x = x + np.random.normal(0, 0.05, size=len(x))
                    
                    # Plot with slight jitter on x-axis
                    ax.scatter(
                        x, ring_data['value'], 
                        color=timepoint_colors[tp],
                        s=50, alpha=0.7, edgecolors='black', linewidths=0.5,
                        label=tp if i == 0 else ""  # Only add to legend once
                    )
        
        # Set x-ticks
        ax.set_xticks(range(len(sorted_ring_numbers)))
        ax.set_xticklabels([f"Ring {ring}" for ring in sorted_ring_numbers])
        
    elif plot_type == 'box':
        # Create a box plot
        sns.boxplot(
            data=plot_df,
            x='ring', y='value', hue='timepoint',
            palette=timepoint_colors,
            ax=ax
        )

        # Add strip plot on top (this automatically adds dots over the boxes)
        sns.stripplot(
            data=plot_df,
            x='ring', y='value', hue='timepoint',
            palette=timepoint_colors,
            dodge=True,  # This makes the points align with their respective boxes
            alpha=0.7,
            size=4,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
            legend=False  # Don't add a second legend
        )
    elif plot_type == 'strip':
        # one box plot per thingy
        sns.boxplot(
            data=plot_df,
            x='ring', y='value',
            color='lightgray',  # Plain gray boxes
            ax=ax,
            showfliers=False
        )
        # Add strip plot on top (this automatically adds dots over the boxes)
        sns.stripplot(
            data=plot_df,
            x='ring', y='value', hue='timepoint',
            palette=timepoint_colors,
            dodge=True,  # This makes the points align with their respective boxes
            alpha=0.7,
            size=4,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
            legend=True  # Don't add a second legend
        )
        
    elif plot_type == 'violin':
        # Create a violin plot
        sns.violinplot(
            data=plot_df,
            x='ring', y='value', hue='timepoint',
            palette=timepoint_colors,
            split=True, inner='quartile',
            ax=ax
        )
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Timepoint')
    
    # Set axis labels and title
    ax.set_xlabel('Ring Number')
    ax.set_ylabel(f'{parameter.upper()} Value')
    if 'fa' in parameter:
        ax.set_ylim(0, 0.5)
    elif 'md' in parameter:
        ax.set_ylim(0, 0.0035)
    
    # Format region type for title
    region_title = region_type.replace('_', ' ').title()
    if region_title.startswith('Baseline'):
        region_title = region_title.replace('Baseline', 'Control Side')
    elif region_title.startswith('Anterior'):
        region_title = region_title.replace('Anterior', 'Craniectomy Side Anterior')
    elif region_title.startswith('Posterior'):
        region_title = region_title.replace('Posterior', 'Craniectomy Side Posterior')
    ax.set_title(f'{parameter.upper()} Values for {region_title} Rings by Timepoint')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
    
    


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
    # data_5x4vox_not_harmonised_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_notharmonised.csv'
    # data_5x4vox_not_harmonised=process_timepoint_data(input_file_location=data_5x4vox_not_harmonised_filename)
    # plot_all_rings_combined(df=data_5x4vox_not_harmonised, parameter='fa', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_not_harmonised.png')
    # plot_all_rings_combined(df=data_5x4vox_not_harmonised, parameter='md', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_not_harmonised_md.png')

    
    data_5x4vox_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_harmonised.csv'
    data_5x4vox=process_timepoint_data(input_file_location=data_5x4vox_filename)
    # Now data_5x4vox has been recategorized based on Days_since_injury, exactly the same as the deformation analysis
    # plot_all_rings_combined(df=data_5x4vox, parameter='fa', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_filtered.png')
    # plot_all_rings_combined(df=data_5x4vox, parameter='md', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox_filtered_md.png')


    # # Scatter plots for 5x4 vox data
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_anterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_anterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_posterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_posterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_anterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_anterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_posterior_scatter.png')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_posterior_scatter.png')

    # # Box plots for 5x4 vox data
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_posterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_posterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_posterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_posterior_box.png', plot_type='box')

    # # Strip plots for 5x4 vox data
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_5x4vox_baseline_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=data_5x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox_baseline_posterior_strip.png', plot_type='strip')

    wm_data_5x4vox_filename='DTI_Processing_Scripts/merged_data_5x4vox_NEW_filtered_wm_harmonised.csv'
    wm_data_5x4vox=process_timepoint_data(input_file_location=wm_data_5x4vox_filename)

    # # Strip plots for 5x4 vox data
    # plot_metric_roi(df=wm_data_5x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_baseline_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_baseline_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_5x4vox_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_5x4vox_baseline_anterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_5x4vox_posterior_strip.png', plot_type='strip')
    # plot_metric_roi(df=wm_data_5x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_5x4vox_baseline_posterior_strip.png', plot_type='strip')

    plot_metric_difference(df=wm_data_5x4vox, parameter='fa', region='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_anterior_comparison_box.png', plot_type='strip')
    plot_metric_difference(df=wm_data_5x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_both_regions_comparison_box.png', plot_type='strip', group_by='timepoint')
    # plot_metric_difference

    wm_data_10x4vox_filename='DTI_Processing_Scripts/merged_data_10x4vox_NEW_filtered_wm_harmonised.csv'
    wm_data_10x4vox=process_timepoint_data(input_file_location=wm_data_10x4vox_filename)
    # plot_metric_difference(df=wm_data_5x4vox, parameter='fa', region='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_5x4vox_anterior_comparison_box.png', plot_type='strip')
    plot_metric_difference(df=wm_data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_10x4vox_both_regions_comparison_box.png', plot_type='strip', group_by='timepoint')
    plot_metric_difference(df=wm_data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_10x4vox_both_regions_comparison_box.png', plot_type='strip', group_by='timepoint')
    plot_metric_difference(df=wm_data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_wm_10x4vox_both_regions_comparison_boxplot.png', plot_type='box', group_by='timepoint')
    plot_metric_difference(df=wm_data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_wm_10x4vox_both_regions_comparison_boxplot.png', plot_type='box', group_by='timepoint')


    sys.exit()



    # plot_metric_roi(df=data_5x4vox, parameter='md', save_path='DTI_Processing_Scripts/test_results/roi_md_5x4vox.png')



    # Load the (harmonised) data
    data_10x4vox_filename='DTI_Processing_Scripts/merged_data_10x4vox_NEW_filtered_harmonised.csv'
    data_10x4vox=process_timepoint_data(input_file_location=data_10x4vox_filename)
    
    #Scatter plots for 10x4vox
    # plot_metric_roi(df=data_10x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_anterior_scatter.png')
    # plot_metric_roi(df=data_10x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_baseline_anterior_scatter.png')
    # plot_metric_roi(df=data_10x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_posterior_scatter.png')
    # plot_metric_roi(df=data_10x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_baseline_posterior_scatter.png')
    # plot_metric_roi(df=data_10x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_anterior_scatter.png')
    # plot_metric_roi(df=data_10x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_baseline_anterior_scatter.png')
    # plot_metric_roi(df=data_10x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_posterior_scatter.png')
    # plot_metric_roi(df=data_10x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_baseline_posterior_scatter.png')

    # Box plots for 10x4 vox
    # plot_metric_roi(df=data_10x4vox, parameter='fa', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_10x4vox, parameter='fa', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_baseline_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_10x4vox, parameter='fa', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_posterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_10x4vox, parameter='fa', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_baseline_posterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_10x4vox, parameter='md', region_type='anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_10x4vox, parameter='md', region_type='baseline_anterior', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_baseline_anterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_10x4vox, parameter='md', region_type='posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_posterior_box.png', plot_type='box')
    # plot_metric_roi(df=data_10x4vox, parameter='md', region_type='baseline_posterior', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_baseline_posterior_box.png', plot_type='box')


    # plot_metric_difference(df=data_10x4vox, parameter='fa', region='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_anterior_comparison_box.png', plot_type='box')
    # plot_metric_difference(df=data_10x4vox, parameter='fa', region='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_anterior_comparison_strip.png', plot_type='strip')
    # plot_metric_difference(df=data_10x4vox, parameter='fa', region='anterior', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_anterior_comparison_scatter.png', plot_type='scatter')

    # plot_metric_difference(df=data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_both_regions_comparison_box.png', plot_type='box')
    # plot_metric_difference(df=data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_both_regions_comparison_strip.png', plot_type='strip')
    # plot_metric_difference(df=data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_both_regions_comparison_scatter.png', plot_type='scatter')

    # plot_metric_difference(df=data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_both_regions_comparison_box.png', plot_type='box')
    # plot_metric_difference(df=data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_both_regions_comparison_strip.png', plot_type='strip')
    # plot_metric_difference(df=data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_both_regions_comparison_scatter.png', plot_type='scatter')

    # group by timepoint
    # plot_metric_difference(df=data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_both_regions_comparison_box_by_timepoint.png', plot_type='box', group_by='timepoint')
    plot_metric_difference(df=data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_both_regions_comparison_strip_by_timepoint.png', plot_type='strip', group_by='timepoint')
    # plot_metric_difference(df=data_10x4vox, parameter='fa', region='both', save_path='DTI_Processing_Scripts/test_results/roi_fa_10x4vox_both_regions_comparison_scatter_by_timepoint.png', plot_type='scatter', group_by='timepoint')

    # plot_metric_difference(df=data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_both_regions_comparison_box_by_timepoint.png', plot_type='box', group_by='timepoint')
    plot_metric_difference(df=data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_both_regions_comparison_strip_by_timepoint.png', plot_type='strip', group_by='timepoint')
    # plot_metric_difference(df=data_10x4vox, parameter='md', region='both', save_path='DTI_Processing_Scripts/test_results/roi_md_10x4vox_both_regions_comparison_scatter_by_timepoint.png', plot_type='scatter', group_by='timepoint')





    ## Plot the FA data. anterior, posterior, baseline_anterior, baseline_posterior on patient by patient basis across time
    print("\n\nScript complete!")


