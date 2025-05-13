import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
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

def plot_all_rings_combined(df, parameter='fa', save_path=None):
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
    
    # Define markers for different ring types to distinguish them
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
    
    # All possible column names for rings
    ring_cols = []
    for region in ['anterior', 'posterior']:
        for ring_num in range(1, 6):
            ring_cols.append(f'{parameter}_{region}_ring_{ring_num}')
            ring_cols.append(f'{parameter}_baseline_{region}_ring_{ring_num}')
    
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
            base_col = col.replace('baseline_', '')
            marker = ring_markers[base_col]
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
            
            # Add to legend (only add each patient and ring type once)
            if pid not in patient_added_to_legend:
                legend_handles.append(plt.Line2D([0], [0], color=patient_color, marker='o', linestyle='-', 
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
                    ax.plot(x_smooth, y_smooth, color=patient_color, alpha=0.7, 
                           linestyle=line_style, linewidth=1.0)
                except Exception as e:
                    print(f"Could not create smooth curve for patient {pid}, {col}: {e}")
    
    # Add baseline vs current markers to legend
    legend_handles.append(plt.Line2D([0], [0], color='black', marker='o', linestyle='None', 
                               markersize=6, alpha=alpha_values['current'], 
                               markeredgecolor='black', markeredgewidth=0.5))
    legend_labels.append('Current')
    
    legend_handles.append(plt.Line2D([0], [0], color='black', marker='o', linestyle='None', 
                               markersize=6, alpha=alpha_values['baseline']))
    legend_labels.append('Baseline')
    
    # Add anterior vs posterior line styles to legend
    legend_handles.append(plt.Line2D([0], [0], color='black', linestyle=line_styles['anterior']))
    legend_labels.append('Anterior')
    
    legend_handles.append(plt.Line2D([0], [0], color='black', linestyle=line_styles['posterior']))
    legend_labels.append('Posterior')
    
    # Set labels and title
    ax.set_xlabel('Days Since Injury')
    ax.set_ylabel('Fractional Anisotropy (FA)')
    ax.set_title('Fractional Anisotropy: All Rings Over Time')

    import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
import matplotlib as mpl
from matplotlib.lines import Line2D

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
        'legend.fontsize': 9,
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

def plot_all_rings_combined(df, save_path=None):
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
    
    # Define markers for different ring types to distinguish them
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
    
    # All possible column names for rings
    ring_cols = []
    for region in ['anterior', 'posterior']:
        for ring_num in range(1, 6):
            ring_cols.append(f'{region}_ring_{ring_num}')
            ring_cols.append(f'baseline_{region}_ring_{ring_num}')
    
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
            base_col = col.replace('baseline_', '')
            marker = ring_markers[base_col]
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
                    ax.plot(x_smooth, y_smooth, color=patient_color, alpha=0.7, 
                           linestyle=line_style, linewidth=1.0)
                except Exception as e:
                    print(f"Could not create smooth curve for patient {pid}, {col}: {e}")
    
    # Add baseline vs current markers to legend
    legend_handles.append(plt.Line2D([0], [0], color='black', marker='o', linestyle='None', 
                               markersize=6, alpha=alpha_values['current'], 
                               markeredgecolor='black', markeredgewidth=0.5))
    legend_labels.append('Current')
    
    legend_handles.append(plt.Line2D([0], [0], color='black', marker='o', linestyle='None', 
                               markersize=6, alpha=alpha_values['baseline']))
    legend_labels.append('Baseline')
    
    # Add anterior vs posterior line styles to legend
    legend_handles.append(Line2D([0], [0], color='black', linestyle=line_styles['anterior']))
    legend_labels.append('Anterior')
    
    legend_handles.append(Line2D([0], [0], color='black', linestyle=line_styles['posterior']))
    legend_labels.append('Posterior')
    
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
    
    # Add legend
    ax.legend(handles=legend_handles, labels=legend_labels, loc='center left', 
             bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax

# Set publication style for matplotlib
set_publication_style()

if __name__ == '__main__':
    print('running dti_results_plotting_main.py')
    
    # Load the data 
    data_5x4vox=pd.read_csv('DTI_Processing_Scripts/merged_data_5x4vox_NEW.csv')
    data_5x4vox['patient_id'] = data_5x4vox['patient_id'].astype(str)
    timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    string_mask = data_5x4vox['timepoint'].isin(timepoints)
    numeric_mask = ~string_mask & data_5x4vox['timepoint'].apply(lambda x: pd.notnull(pd.to_numeric(x, errors='coerce')))

    # Convert numeric values to their appropriate string representations
    for idx in data_5x4vox[numeric_mask].index:
        try:
            numeric_value = float(data_5x4vox.loc[idx, 'timepoint'])
            data_5x4vox.loc[idx, 'timepoint'] = map_timepoint_to_string(numeric_value)
        except (ValueError, TypeError):
            continue
    
    #data_5x4vox['timepoint'] = data_5x4vox['timepoint'].apply(map_timepoint_to_string) # convert to string category
    #print(f"data_5x4vox:\n{data_5x4vox}")


     
    # Save original timepoint values before recategorization (if needed for reference)
    data_5x4vox['original_timepoint'] = data_5x4vox['timepoint']
    
    # Now apply the timepoint recategorization based on Days_since_injury
    # Define the ranges and labels for recategorization
    ranges = [(0, 2), (2, 8), (8, 42), (42, 179), (179, 278), (278, 540), (540, 500000)]
    labels = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    
    # Only recategorize rows where Days_since_injury is not null
    mask = data_5x4vox['Days_since_injury'].notnull()
    if mask.any():
        data_5x4vox.loc[mask, 'timepoint'] = pd.cut(
            data_5x4vox.loc[mask, 'Days_since_injury'], 
            bins=[0, 2, 8, 42, 179, 278, 540, 500000], 
            labels=labels
        )

    # sort data according to patient id then timepoint

    def get_sort_key(patient_id):
        try:
            return (0, int(patient_id))  # Numeric IDs first, sorted numerically
        except ValueError:
            return (1, patient_id)       # Alphanumeric IDs second, sorted alphabetically

    # Create a sort key column
    data_5x4vox['sort_key'] = data_5x4vox['patient_id'].apply(get_sort_key)
    data_5x4vox['timepoint_order'] = data_5x4vox['timepoint'].apply(lambda x: timepoints.index(x) if x in timepoints else 999)
    
    # Sort the dataframe by patient_id, then by the position of timepoint in our list
    data_5x4vox = data_5x4vox.sort_values(by=['sort_key', 'timepoint_order'])
    data_5x4vox = data_5x4vox.drop(['sort_key', 'timepoint_order'], axis=1)  # remove sorting column
    
    print(f"data_5x4vox columns:\n{data_5x4vox.columns}")
    sys.exit()
    # Now data_5x4vox has been recategorized based on Days_since_injury, exactly the same as the deformation analysis

    plot_all_rings_combined(data_5x4vox, parameter='fa', save_path='DTI_Processing_Scripts/test_results/all_rings_combined_5x4vox.png')




    ## Plot the FA data. anterior, posterior, baseline_anterior, baseline_posterior on patient by patient basis across time



