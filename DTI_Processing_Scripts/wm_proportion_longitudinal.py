#!/usr/bin/env python3
"""
WM Proportion Longitudinal Analysis
Visualizes white matter proportion changes over time for rings 5, 6, 7
Similar to area_change_longitudinal.pdf from Image_Processing_Scripts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from scipy.interpolate import interp1d, CubicSpline
import sys
import os

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

def create_hex_color_map_custom(base_colors, n):
    """Create a custom hex color map from base colors."""
    cmap = LinearSegmentedColormap.from_list('custom', base_colors, N=n)
    colors = cmap(np.linspace(0, 1, n))
    hex_colors = [rgb2hex(color) for color in colors]
    return hex_colors

def map_timepoint_to_string(numeric_timepoint):
    """
    Convert a numeric timepoint to the closest string timepoint.

    Args:
        numeric_timepoint: Numeric value of the timepoint (hours or days)

    Returns:
        String representation of the closest timepoint
    """
    timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Define ranges in hours
    timepoint_ranges = [
        (0, 48),        # ultra-fast: 0-48 hours (0-2 days)
        (48, 192),      # fast: 48-192 hours (2-8 days)
        (192, 1008),    # acute: 192-1008 hours (8-42 days)
        (1008, 4296),   # 3mo: 42-179 days (1008-4296 hours)
        (4296, 6672),   # 6mo: 179-278 days (4296-6672 hours)
        (6672, 12960),  # 12mo: 278-540 days (6672-12960 hours)
        (12960, 500000) # 24mo: 540+ days (12960+ hours)
    ]

    for i, (min_val, max_val) in enumerate(timepoint_ranges):
        if min_val <= numeric_timepoint < max_val:
            return timepoints[i]

    return timepoints[-1]  # Default to last timepoint

def process_timepoint_data(df):
    """
    Process patient timepoint data by standardizing timepoint values.

    Args:
        df: Input pandas DataFrame containing patient data

    Returns:
        Processed pandas DataFrame with standardized timepoints
    """
    df['patient_id'] = df['patient_id'].astype(str)
    timepoints = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']

    # Check if timepoint is already a string
    string_mask = df['timepoint'].isin(timepoints)
    numeric_mask = ~string_mask & df['timepoint'].apply(lambda x: pd.notnull(pd.to_numeric(x, errors='coerce')))

    # Convert numeric values to their appropriate string representations
    for idx in df[numeric_mask].index:
        try:
            numeric_value = float(df.loc[idx, 'timepoint'])
            df.loc[idx, 'timepoint'] = map_timepoint_to_string(numeric_value)
        except (ValueError, TypeError):
            continue

    # Sort data by patient_id then timepoint
    def get_sort_key(patient_id):
        try:
            return (0, int(patient_id))  # Numeric IDs first
        except ValueError:
            return (1, patient_id)  # Alphanumeric IDs second

    df['sort_key'] = df['patient_id'].apply(get_sort_key)
    df['timepoint_order'] = df['timepoint'].apply(lambda x: timepoints.index(x) if x in timepoints else 999)
    df = df.sort_values(by=['sort_key', 'timepoint_order'])
    df = df.drop(['sort_key', 'timepoint_order'], axis=1)
    df = df.reset_index(drop=True)

    return df

def plot_wm_proportion_longitudinal(df, region, ring, output_path):
    """
    Plot WM proportion over time for a specific region and ring.

    Args:
        df: DataFrame with WM proportion data
        region: Region name (ant, post, baseline_ant, baseline_post)
        ring: Ring number (5, 6, or 7)
        output_path: Path to save the figure
    """
    # Column name for this region/ring
    prop_col = f'WM_prop_{region}_ring_{ring}'

    if prop_col not in df.columns:
        print(f"Warning: Column {prop_col} not found in data")
        return

    # Define timepoint order
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    timepoints_num = np.arange(len(timepoint_order))

    # Create a categorical column for sorting
    df['timepoint_cat'] = pd.Categorical(
        df['timepoint'],
        categories=timepoint_order,
        ordered=True
    )

    # Sort by patient and timepoint
    df_sorted = df.sort_values(['patient_id', 'timepoint_cat']).reset_index(drop=True)

    # Get unique patients
    patient_ids = df_sorted['patient_id'].unique()
    n_patients = len(patient_ids)

    # Create color map
    base_colors = ['red', 'cyan', 'yellow', 'magenta', 'brown', 'lightblue', 'orange', 'green', 'purple', 'pink']
    hex_colors = create_hex_color_map_custom(base_colors, n_patients)
    color_map = {pid: hex_colors[i] for i, pid in enumerate(patient_ids)}
    default_color = 'gray'

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot each patient
    for patient_id in patient_ids:
        patient_subset = df_sorted[df_sorted['patient_id'] == patient_id]

        # Extract WM proportions
        wm_prop_array = np.array(patient_subset[prop_col])

        # Create timepoint indices
        timepoint_indices = []
        for tp in patient_subset['timepoint']:
            if tp in timepoint_order:
                timepoint_indices.append(timepoint_order.index(tp))
            else:
                timepoint_indices.append(np.nan)

        timepoint_indices = np.array(timepoint_indices)

        # Find valid (non-NaN) measurements
        valid_indices = ~np.isnan(wm_prop_array) & ~np.isnan(timepoint_indices)

        if not np.any(valid_indices):
            print(f'Patient {patient_id} has no valid WM proportion measurements for {region} ring {ring}')
            continue

        wm_prop_valid = wm_prop_array[valid_indices]
        timepoints_valid = timepoint_indices[valid_indices]

        # Need at least 2 points for interpolation
        if len(wm_prop_valid) < 2:
            # Just plot the single point
            color = color_map.get(patient_id, default_color)
            plt.scatter(timepoints_valid, wm_prop_valid, color=color, s=30, alpha=0.7, label=patient_id)
            continue

        # Create smooth line using cubic spline
        try:
            cs = CubicSpline(timepoints_valid, wm_prop_valid, bc_type='natural')
            x_smooth = np.linspace(timepoints_valid.min(), timepoints_valid.max(), 100)
            y_smooth = cs(x_smooth)

            color = color_map.get(patient_id, default_color)
            plt.plot(x_smooth, y_smooth, label=patient_id, color=color, linewidth=2)
            plt.scatter(timepoints_valid, wm_prop_valid, color=color, s=30, alpha=0.7)
        except Exception as e:
            print(f"Could not interpolate for patient {patient_id}: {e}")
            # Fall back to scatter plot only
            color = color_map.get(patient_id, default_color)
            plt.scatter(timepoints_valid, wm_prop_valid, color=color, s=30, alpha=0.7, label=patient_id)

    # Format plot
    plt.xlim([0, len(timepoint_order) - 1])
    plt.xticks(timepoints_num, timepoint_order, rotation=45, ha='right')
    plt.xlabel('Time')
    plt.ylabel('WM Proportion')

    # Create title
    region_name = region.replace('_', ' ').title()
    plt.title(f'White Matter Proportion Over Time\n{region_name} Ring {ring}')

    # Position legend outside plot
    plt.legend(title='Patient ID', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Redact patient IDs in legend (skip 10)
    leg = plt.gca().get_legend()
    for i, text in enumerate(leg.get_texts(), 1):
        # Skip number 10: if i >= 10, add 1 to the displayed number
        display_num = i + 1 if i >= 10 else i
        text.set_text(str(display_num))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_wm_proportion_combined_rings(df, region, output_path):
    """
    Plot average WM proportion across rings 5, 6, 7 over time for a specific region.

    Args:
        df: DataFrame with WM proportion data
        region: Region name (ant, post, baseline_ant, baseline_post)
        output_path: Path to save the figure
    """
    # Calculate average proportion across rings 5, 6, 7
    prop_cols = [f'WM_prop_{region}_ring_{ring}' for ring in [5, 6, 7]]

    # Check if all columns exist
    missing_cols = [col for col in prop_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        return

    # Calculate mean across rings
    df['WM_prop_mean'] = df[prop_cols].mean(axis=1)

    # Define timepoint order
    timepoint_order = ['ultra-fast', 'fast', 'acute', '3mo', '6mo', '12mo', '24mo']
    timepoints_num = np.arange(len(timepoint_order))

    # Create a categorical column for sorting
    df['timepoint_cat'] = pd.Categorical(
        df['timepoint'],
        categories=timepoint_order,
        ordered=True
    )

    # Sort by patient and timepoint
    df_sorted = df.sort_values(['patient_id', 'timepoint_cat']).reset_index(drop=True)

    # Get unique patients
    patient_ids = df_sorted['patient_id'].unique()
    n_patients = len(patient_ids)

    # Create color map
    base_colors = ['red', 'cyan', 'yellow', 'magenta', 'brown', 'lightblue', 'orange', 'green', 'purple', 'pink']
    hex_colors = create_hex_color_map_custom(base_colors, n_patients)
    color_map = {pid: hex_colors[i] for i, pid in enumerate(patient_ids)}
    default_color = 'gray'

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot each patient
    for patient_id in patient_ids:
        patient_subset = df_sorted[df_sorted['patient_id'] == patient_id]

        # Extract WM proportions
        wm_prop_array = np.array(patient_subset['WM_prop_mean'])

        # Create timepoint indices
        timepoint_indices = []
        for tp in patient_subset['timepoint']:
            if tp in timepoint_order:
                timepoint_indices.append(timepoint_order.index(tp))
            else:
                timepoint_indices.append(np.nan)

        timepoint_indices = np.array(timepoint_indices)

        # Find valid (non-NaN) measurements
        valid_indices = ~np.isnan(wm_prop_array) & ~np.isnan(timepoint_indices)

        if not np.any(valid_indices):
            print(f'Patient {patient_id} has no valid WM proportion measurements for {region}')
            continue

        wm_prop_valid = wm_prop_array[valid_indices]
        timepoints_valid = timepoint_indices[valid_indices]

        # Need at least 2 points for interpolation
        if len(wm_prop_valid) < 2:
            color = color_map.get(patient_id, default_color)
            plt.scatter(timepoints_valid, wm_prop_valid, color=color, s=30, alpha=0.7, label=patient_id)
            continue

        # Create smooth line using cubic spline
        try:
            cs = CubicSpline(timepoints_valid, wm_prop_valid, bc_type='natural')
            x_smooth = np.linspace(timepoints_valid.min(), timepoints_valid.max(), 100)
            y_smooth = cs(x_smooth)

            color = color_map.get(patient_id, default_color)
            plt.plot(x_smooth, y_smooth, label=patient_id, color=color, linewidth=2)
            plt.scatter(timepoints_valid, wm_prop_valid, color=color, s=30, alpha=0.7)
        except Exception as e:
            print(f"Could not interpolate for patient {patient_id}: {e}")
            color = color_map.get(patient_id, default_color)
            plt.scatter(timepoints_valid, wm_prop_valid, color=color, s=30, alpha=0.7, label=patient_id)

    # Format plot
    plt.xlim([0, len(timepoint_order) - 1])
    plt.xticks(timepoints_num, timepoint_order, rotation=45, ha='right')
    plt.xlabel('Time')
    plt.ylabel('Mean WM Proportion (Rings 5-7)')

    # Create title
    region_name = region.replace('_', ' ').title()
    plt.title(f'White Matter Proportion Over Time\n{region_name} (Average Rings 5-7)')

    # Position legend outside plot
    plt.legend(title='Patient ID', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Redact patient IDs in legend (skip 10)
    leg = plt.gca().get_legend()
    for i, text in enumerate(leg.get_texts(), 1):
        # Skip number 10: if i >= 10, add 1 to the displayed number
        display_num = i + 1 if i >= 10 else i
        text.set_text(str(display_num))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    """Main analysis function."""

    # Set publication style
    set_publication_style()

    # Create output directory
    output_dir = 'DTI_Processing_Scripts/wm_proportion_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Define input file (modify as needed)
    input_file = 'DTI_Processing_Scripts/results/all_wm_proportion_analysis_10x4vox_NEW_filtered.csv'

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please ensure WM proportion data has been extracted first.")
        sys.exit(1)

    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # Process timepoints
    df = process_timepoint_data(df)

    print(f"Loaded {len(df)} patient-timepoint records")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Timepoints: {df['timepoint'].unique()}")

    # Plot for each region and ring combination
    regions = ['ant', 'post', 'baseline_ant', 'baseline_post']
    rings = [5, 6, 7]

    print("\nGenerating individual ring plots...")
    for region in regions:
        for ring in rings:
            output_path = os.path.join(output_dir, f'wm_proportion_{region}_ring_{ring}.png')
            print(f"Plotting {region} ring {ring}...")
            plot_wm_proportion_longitudinal(df, region, ring, output_path)

    # Plot combined (average across rings 5, 6, 7)
    print("\nGenerating combined ring plots (average of rings 5-7)...")
    for region in regions:
        output_path = os.path.join(output_dir, f'wm_proportion_{region}_combined_567.png')
        print(f"Plotting {region} combined rings 5-7...")
        plot_wm_proportion_combined_rings(df, region, output_path)

    print(f"\nAll plots saved to {output_dir}/")
    print("Analysis complete!")

if __name__ == '__main__':
    main()
