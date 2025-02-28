import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point, LineString, LinearRing
import shapely.ops
import os
from scipy.optimize import minimize
from scipy.spatial import distance
import seaborn as sns
from scipy import stats
import sys

def convert_to_numpy_array(s):
    """Convert string representation of array to numpy array."""
    if isinstance(s, str):
        s = s.strip('[]')
        def convert_value(value):
            try:
                return int(value)
            except ValueError:
                return float(value)
        return np.array([convert_value(value) for value in s.split()])
    return s  # Return as is if already a numpy array

def point_to_ellipse_distance(point, h_param, a_param, x_vals, y_vals):
    """
    Calculate the minimum distance from a point to the ellipse.
    
    Uses a discrete approximation by finding the closest point on the ellipse.
    
    Parameters:
    point (tuple): (x, y) coordinates of the point
    h_param (float): Height parameter of the ellipse
    a_param (float): Width parameter of the ellipse
    x_vals (array): x-coordinates of the ellipse points
    y_vals (array): y-coordinates of the ellipse points
    
    Returns:
    float: Minimum distance from the point to the ellipse
    """
    x0, y0 = point
    
    # Create array of points on the ellipse 
    ellipse_points = np.column_stack((x_vals, y_vals))
    
    # Calculate distances from the point to all points on the ellipse
    distances = np.sqrt(np.sum((ellipse_points - np.array([x0, y0]))**2, axis=1))
    
    # Return the minimum distance
    return np.min(distances)

def calculate_rmse_ellipse_fit(contour_x, contour_y, ellipse_x, ellipse_y, h_param, a_param):
    """
    Calculate Root Mean Square Error between contour points and ellipse.
    
    Parameters:
    contour_x, contour_y (array): Coordinates of contour points
    ellipse_x, ellipse_y (array): Coordinates of ellipse points
    h_param (float): Height parameter of ellipse
    a_param (float): Width parameter of ellipse
    
    Returns:
    float: RMSE value
    """
    distances = []
    
    # For each contour point, find the minimum distance to the ellipse
    for i in range(len(contour_x)):
        point = (contour_x[i], contour_y[i])
        dist = point_to_ellipse_distance(point, h_param, a_param, ellipse_x, ellipse_y)
        distances.append(dist)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.array(distances)**2))
    return rmse

def calculate_mae_ellipse_fit(contour_x, contour_y, ellipse_x, ellipse_y, h_param, a_param):
    """
    Calculate Mean Absolute Error between contour points and ellipse.
    
    Parameters:
    contour_x, contour_y (array): Coordinates of contour points
    ellipse_x, ellipse_y (array): Coordinates of ellipse points
    h_param (float): Height parameter of ellipse
    a_param (float): Width parameter of ellipse
    
    Returns:
    float: MAE value
    """
    distances = []
    
    # For each contour point, find the minimum distance to the ellipse
    for i in range(len(contour_x)):
        point = (contour_x[i], contour_y[i])
        dist = point_to_ellipse_distance(point, h_param, a_param, ellipse_x, ellipse_y)
        distances.append(dist)
    
    # Calculate MAE
    mae = np.mean(np.abs(distances))
    return mae

def calculate_max_error_ellipse_fit(contour_x, contour_y, ellipse_x, ellipse_y, h_param, a_param):
    """
    Calculate maximum error between contour points and ellipse.
    
    Parameters:
    contour_x, contour_y (array): Coordinates of contour points
    ellipse_x, ellipse_y (array): Coordinates of ellipse points
    h_param (float): Height parameter of ellipse
    a_param (float): Width parameter of ellipse
    
    Returns:
    float: Maximum error value
    """
    distances = []
    
    # For each contour point, find the minimum distance to the ellipse
    for i in range(len(contour_x)):
        point = (contour_x[i], contour_y[i])
        dist = point_to_ellipse_distance(point, h_param, a_param, ellipse_x, ellipse_y)
        distances.append(dist)
    
    # Calculate maximum error
    max_error = np.max(distances)
    return max_error

def calculate_r2_ellipse_fit(contour_x, contour_y, ellipse_x, ellipse_y, h_param, a_param):
    """
    Calculate R-squared (coefficient of determination) for ellipse fit.
    
    Parameters:
    contour_x, contour_y (array): Coordinates of contour points
    ellipse_x, ellipse_y (array): Coordinates of ellipse points
    h_param (float): Height parameter of ellipse
    a_param (float): Width parameter of ellipse
    
    Returns:
    float: R-squared value
    """
    # Calculate distances to ellipse
    distances = []
    for i in range(len(contour_x)):
        point = (contour_x[i], contour_y[i])
        dist = point_to_ellipse_distance(point, h_param, a_param, ellipse_x, ellipse_y)
        distances.append(dist)
    
    # Calculate mean y-coordinate of contour
    y_mean = np.mean(contour_y)
    
    # Calculate total sum of squares
    ss_total = np.sum((contour_y - y_mean) ** 2)
    
    # Calculate residual sum of squares
    ss_residual = np.sum(np.array(distances) ** 2)
    
    # Calculate R-squared
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    return r_squared

def calculate_area_ratio(contour_x, contour_y, ellipse_x, ellipse_y):
    """
    Calculate ratio of contour area to ellipse area.
    
    Parameters:
    contour_x, contour_y (array): Coordinates of contour points
    ellipse_x, ellipse_y (array): Coordinates of ellipse points
    
    Returns:
    float: Area ratio
    """
    # Create polygons
    contour_polygon = ShapelyPolygon(list(zip(contour_x, contour_y)))
    ellipse_polygon = ShapelyPolygon(list(zip(ellipse_x, ellipse_y)))
    
    # Calculate areas
    contour_area = contour_polygon.area
    ellipse_area = ellipse_polygon.area
    
    # Calculate ratio (ensure non-zero denominator)
    ratio = contour_area / ellipse_area if ellipse_area > 0 else 0
    
    return ratio

def calculate_overlap_metrics(contour_x, contour_y, ellipse_x, ellipse_y):
    """
    Calculate area-based overlap metrics for the fit.
    
    Parameters:
    contour_x, contour_y (array): Coordinates of contour points
    ellipse_x, ellipse_y (array): Coordinates of ellipse points
    
    Returns:
    dict: Dictionary of overlap metrics
    """
    try:
        # Create valid polygons by ensuring they're properly formed
        # 1. Buffer by a tiny amount to fix topology issues
        # 2. Convert to LinearRing and check if valid
        
        # For contour polygon
        contour_points = list(zip(contour_x, contour_y))
        contour_polygon = make_valid_polygon(contour_points)
        
        # For ellipse polygon
        ellipse_points = list(zip(ellipse_x, ellipse_y))
        ellipse_polygon = make_valid_polygon(ellipse_points)
        
        # If either polygon is invalid, simplify calculation
        if contour_polygon is None or ellipse_polygon is None:
            # Estimate area using simpler methods
            contour_area = simple_polygon_area(contour_points)
            ellipse_area = simple_polygon_area(ellipse_points)
            
            # Calculate relative difference in areas
            if contour_area > 0:
                # Use a more appropriate area difference formula
                rel_diff = abs(ellipse_area - contour_area) / max(ellipse_area, contour_area)
                area_diff_pct = 100 * rel_diff
            else:
                area_diff_pct = 100.0
            
            return {
                'contour_area': contour_area,
                'ellipse_area': ellipse_area,
                'intersection_area': 0,
                'iou': 0,
                'dice': 0,
                'area_diff_pct': min(area_diff_pct, 100.0)  # Cap at 100%
            }
        
        # Calculate areas
        contour_area = contour_polygon.area
        ellipse_area = ellipse_polygon.area
        
        # Calculate intersection and union with error handling
        try:
            intersection = contour_polygon.intersection(ellipse_polygon)
            intersection_area = intersection.area
        except Exception:
            intersection_area = 0
            
        try:
            union = contour_polygon.union(ellipse_polygon)
            union_area = union.area
        except Exception:
            union_area = contour_area + ellipse_area - intersection_area
        
        # Calculate IoU (Intersection over Union) - Jaccard Index
        iou = intersection_area / union_area if union_area > 0 else 0
        
        # Calculate Dice coefficient
        dice = (2 * intersection_area) / (contour_area + ellipse_area) if (contour_area + ellipse_area) > 0 else 0
        
        # Calculate area difference percentage - improved formula
        # Using the larger area as denominator ensures result is between 0-100%
        if max(contour_area, ellipse_area) > 0:
            rel_diff = abs(ellipse_area - contour_area) / max(ellipse_area, contour_area)
            area_diff_pct = 100 * rel_diff
        else:
            area_diff_pct = 100.0
        
        return {
            'contour_area': contour_area,
            'ellipse_area': ellipse_area,
            'intersection_area': intersection_area,
            'iou': iou,
            'dice': dice,
            'area_diff_pct': min(area_diff_pct, 100.0)  # Cap at 100%
        }
    except Exception as e:
        print(f"Error calculating overlap metrics: {e}")
        # Fallback to simple area difference calculation
        try:
            # Estimate area using simpler methods
            contour_area = simple_polygon_area(list(zip(contour_x, contour_y)))
            ellipse_area = simple_polygon_area(list(zip(ellipse_x, ellipse_y)))
            
            # Calculate area difference percentage - improved formula
            if max(contour_area, ellipse_area) > 0:
                rel_diff = abs(ellipse_area - contour_area) / max(ellipse_area, contour_area)
                area_diff_pct = 100 * rel_diff
            else:
                area_diff_pct = 100.0
            
            return {
                'contour_area': contour_area,
                'ellipse_area': ellipse_area,
                'intersection_area': 0,
                'iou': 0,
                'dice': 0,
                'area_diff_pct': min(area_diff_pct, 100.0)  # Cap at 100%
            }
        except:
            return {
                'contour_area': 0,
                'ellipse_area': 0,
                'intersection_area': 0,
                'iou': 0,
                'dice': 0,
                'area_diff_pct': 100.0
            }

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
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def visualize_fit_analysis(contour_x, contour_y, ellipse_x, ellipse_y, metrics, patient_id, timepoint, side, name, output_dir='ellipse_fit_analysis'):
    """
    Create visualization of the ellipse fit with metrics.
    
    Parameters:
    contour_x, contour_y (array): Coordinates of contour points
    ellipse_x, ellipse_y (array): Coordinates of ellipse points
    metrics (dict): Dictionary of fit metrics
    patient_id (str): Patient ID
    timepoint (str): Timepoint
    side (str): Side (L/R)
    name (str): Name ('def' or 'ref')
    output_dir (str): Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication style
    set_publication_style()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors
    def_color = '#FF5555'  # Bright red
    ref_color = '#5555FF'  # Bright blue
    def_dark = '#8B0000'   # Dark red
    ref_dark = '#00008B'   # Dark blue
    
    # Determine color scheme based on analysis type
    contour_color = def_color if name == 'def' else ref_color
    ellipse_color = def_dark if name == 'def' else ref_dark
    
    # Plot contour points
    ax.scatter(contour_x, contour_y, color=contour_color, s=10, alpha=0.7, 
               edgecolor=ellipse_color, linewidth=0.5, 
               label=f'{"Deformed" if name == "def" else "Reference"} Contour')
    
    # Plot ellipse curve
    ax.plot(ellipse_x, ellipse_y, color=ellipse_color, linewidth=2, 
            label=f'{"Deformed" if name == "def" else "Reference"} Ellipse Fit')
    
    # Format metrics text
    metrics_text = (
        f"Metrics:\n"
        f"RMSE: {metrics['rmse']:.3f}\n"
        f"R²: {metrics['r2']:.3f}\n"
        f"Area Similarity: {metrics['area_diff_pct']:.1f}%\n"
        f"h: {metrics['h_param']:.2f}"
    )
    
    # Position metrics text in top right corner
    bbox_props = dict(boxstyle="round,pad=0.5", 
                      facecolor='white', 
                      edgecolor=ellipse_color, 
                      alpha=0.8)
    
    ax.text(0.97, 0.97, metrics_text, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top', 
            horizontalalignment='right', 
            bbox=bbox_props,
            color=ellipse_color)
    
    # Add title
    ax.set_title(f"Ellipse Fit Analysis - Patient {patient_id}, Timepoint {timepoint}, "
                 f"{'Deformed' if name == 'def' else 'Reference'} Configuration")
    
    # Add legend with nice formatting - positioned outside plot area
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), frameon=True, framealpha=0.9, edgecolor='gray')
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Set y-axis limit to match original plots
    ax.set_ylim(top=60)
    
    # Add labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Adjust figure layout
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    
    # Save figure
    filename = f"{patient_id}_{timepoint}_{name}_fit_analysis.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path



def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',  # This makes titles bold
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

def create_summary_visualisations(metrics_df, output_dir):
    """
    Generate comprehensive visualizations and summary statistics for contour metrics.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame containing contour metrics with columns for def and ref metrics
    output_dir : str
        Directory to save output visualizations and summary files
    
    Returns:
    --------
    dict
        Dictionary containing summary statistics and test results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication style
    set_publication_style()
    
    # Color palette
    # Colors
    def_color = '#FF5555'  # Bright red
    ref_color = '#5555FF'  # Bright blue
    def_dark = '#8B0000'   # Dark red
    ref_dark = '#00008B'   # Dark blue

    
    # Prepare long-format data for plotting
    metrics_to_plot = ['rmse', 'r2']
    long_data = []
    
    for metric in metrics_to_plot:
        def_data = metrics_df[f'def_{metric}']
        ref_data = metrics_df[f'ref_{metric}']
        
        def_rows = pd.DataFrame({
            'metric_value': def_data,
            'configuration': 'Deformed',
            'metric_name': metric.upper()
        })
        
        ref_rows = pd.DataFrame({
            'metric_value': ref_data,
            'configuration': 'Reference',
            'metric_name': metric.upper()
        })
        
        long_data.append(def_rows)
        long_data.append(ref_rows)
    
    long_df = pd.concat(long_data, ignore_index=True)
    
    # 1. Box plot of RMSE and R²# 1. Create plots with ONLY the data points - no boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # RMSE Plot (left)
    rmse_data = long_df[long_df['metric_name'] == 'RMSE']

    # Set up x-axis
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Deformed', 'Reference'])

    # Get data for each configuration
    deformed_rmse = rmse_data[rmse_data['configuration'] == 'Deformed']['metric_value'].values
    reference_rmse = rmse_data[rmse_data['configuration'] == 'Reference']['metric_value'].values

    # Add only scatter points for Deformed
    ax1.scatter(
        [0] * len(deformed_rmse),
        deformed_rmse,
        color=def_color, s=10, alpha=0.7,
        edgecolor=def_color,  # Match fill color to remove grey rings
        linewidth=0.5
    )

    # Add only scatter points for Reference
    ax1.scatter(
        [1] * len(reference_rmse),
        reference_rmse,
        color=ref_color, s=10, alpha=0.7,
        edgecolor=ref_color,  # Match fill color to remove grey rings
        linewidth=0.5
    )

    # Optional: Add subtle statistical indicators manually
    # Calculate statistics
    deformed_median = np.median(deformed_rmse)
    reference_median = np.median(reference_rmse)
    deformed_q1 = np.percentile(deformed_rmse, 25)
    deformed_q3 = np.percentile(deformed_rmse, 75)
    reference_q1 = np.percentile(reference_rmse, 25)
    reference_q3 = np.percentile(reference_rmse, 75)

    # Add thin horizontal lines for median
    ax1.axhline(y=deformed_median, xmin=0.2, xmax=0.3, color=def_color, linestyle='-', linewidth=1)
    ax1.axhline(y=reference_median, xmin=0.7, xmax=0.8, color=ref_color, linestyle='-', linewidth=1)

    # Set titles and labels
    ax1.set_title('RMSE by Configuration')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Root Mean Square Error')

    # R² Plot (right)
    r2_data = long_df[long_df['metric_name'] == 'R2']

    # Set up x-axis
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Deformed', 'Reference'])

    # Get data for each configuration
    deformed_r2 = r2_data[r2_data['configuration'] == 'Deformed']['metric_value'].values
    reference_r2 = r2_data[r2_data['configuration'] == 'Reference']['metric_value'].values

    # Add only scatter points for Deformed
    ax2.scatter(
        [0] * len(deformed_r2),
        deformed_r2,
        color=def_color, s=10, alpha=0.7,
        edgecolor=def_color,  # Match fill color to remove grey rings
        linewidth=0.5
        )

    # Add only scatter points for Reference
    ax2.scatter(
        [1] * len(reference_r2),
        reference_r2,
        color=ref_color, s=10, alpha=0.7,
        edgecolor=ref_color,  # Match fill color to remove grey rings
        linewidth=0.5
    )

    # Optional: Add subtle statistical indicators manually
    # Calculate statistics
    deformed_median_r2 = np.median(deformed_r2)
    reference_median_r2 = np.median(reference_r2)

    # Add thin horizontal lines for median
    ax2.axhline(y=deformed_median_r2, xmin=0.2, xmax=0.3, color=def_color, linestyle='-', linewidth=1)
    ax2.axhline(y=reference_median_r2, xmin=0.7, xmax=0.8, color=ref_color, linestyle='-', linewidth=1)

    # Set titles and labels
    ax2.set_title('R² by Configuration')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('R² (Coefficient of Determination)')

    
    # Final adjustments
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_distribution.png'), dpi=300)
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/scatter_distribution.pdf', dpi=300)
    plt.close()
    
    # 2. Scatter plot of RMSE vs R²
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(metrics_df['def_r2'], metrics_df['def_rmse'], 
                          color=def_color, 
                          alpha=0.7, 
                          label='Deformed',
                          edgecolors='black', 
                          linewidth=0.5)
    plt.scatter(metrics_df['ref_r2'], metrics_df['ref_rmse'], 
                color=ref_color, 
                alpha=0.7, 
                label='Reference',
                edgecolors='black', 
                linewidth=0.5)
    
    plt.xlabel('Root Mean Square Error (RMSE)')
    plt.ylabel('R² (Coefficient of Determination)')
    plt.title('RMSE vs R² by Configuration')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_rmse_r2.png'))
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/scatter_rmse_r2.pdf', dpi=300)
    plt.close()
    
    # 3. Statistical tests
    results = {}
    
    # RMSE t-test
    rmse_ttest = stats.ttest_ind(metrics_df['def_rmse'], metrics_df['ref_rmse'])
    
    # R² t-test
    r2_ttest = stats.ttest_ind(metrics_df['def_r2'], metrics_df['ref_r2'])
    
    # Store results
    results['rmse_ttest'] = {
        't_statistic': rmse_ttest.statistic,
        'p_value': rmse_ttest.pvalue
    }
    results['r2_ttest'] = {
        't_statistic': r2_ttest.statistic,
        'p_value': r2_ttest.pvalue
    }
    
    # Save statistical test results
    test_results_file = os.path.join(output_dir, 'statistical_tests.txt')
    with open(test_results_file, 'w') as f:
        f.write("Statistical Test Results\n")
        f.write("======================\n\n")
        f.write("RMSE T-Test:\n")
        f.write(f"t-statistic: {rmse_ttest.statistic:.4f}\n")
        f.write(f"p-value: {rmse_ttest.pvalue:.4f}\n\n")
        f.write("R² T-Test:\n")
        f.write(f"t-statistic: {r2_ttest.statistic:.4f}\n")
        f.write(f"p-value: {r2_ttest.pvalue:.4f}\n")
    
    # Print summary to console
    print("Summary statistics and visualizations saved in:", output_dir)
    print("\nRMSE T-Test:")
    print(f"t-statistic: {rmse_ttest.statistic:.4f}, p-value: {rmse_ttest.pvalue:.4f}")
    print("\nR² T-Test:")
    print(f"t-statistic: {r2_ttest.statistic:.4f}, p-value: {r2_ttest.pvalue:.4f}")
    
    return results
    
def main():
    """Main function to run the ellipse fit analysis."""
    # File paths
    input_filename1 = 'batch2_ellipse_data.pkl'
    input_filename2 = 'ellipse_data.pkl'
    output_filename = 'combined_ellipse_fit_metrics.csv'
    output_dir = 'Image_Processing_Scripts/ellipse_fit_analysis'

    # flags
    plot_ellipse_flag = False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize combined dataframe
    combined_data = pd.DataFrame()
    
    # Load and combine data from both pickle files
    print("Loading and combining data from pickle files...")
    
    # Try to load first dataset
    try:
        data1 = pd.read_pickle(f'Image_Processing_Scripts/{input_filename1}')
        print(f"Successfully loaded data from Image_Processing_Scripts/{input_filename1}")
    except FileNotFoundError:
        try:
            data1 = pd.read_pickle(input_filename1)
            print(f"Successfully loaded data from {input_filename1}")
        except FileNotFoundError:
            print(f"Could not find {input_filename1} in either location.")
            data1 = None
    
    # Try to load second dataset
    try:
        data2 = pd.read_pickle(f'Image_Processing_Scripts/{input_filename2}')
        print(f"Successfully loaded data from Image_Processing_Scripts/{input_filename2}")
    except FileNotFoundError:
        try:
            data2 = pd.read_pickle(input_filename2)
            print(f"Successfully loaded data from {input_filename2}")
        except FileNotFoundError:
            print(f"Could not find {input_filename2} in either location.")
            data2 = None
    
    # Combine datasets if available
    if data1 is not None and data2 is not None:
        # Check if columns match
        data1_cols = set(data1.columns)
        data2_cols = set(data2.columns)
        
        if data1_cols == data2_cols:
            # Straightforward concatenation if columns match
            combined_data = pd.concat([data1, data2], ignore_index=True)
            print("Successfully combined both datasets.")
        else:
            # Handle different column sets
            print("Datasets have different columns:")
            print(f"First dataset columns: {data1_cols}")
            print(f"Second dataset columns: {data2_cols}")
            
            # Find common columns
            common_cols = data1_cols.intersection(data2_cols)
            required_cols = {'patient_id', 'timepoint', 'side', 
                            'h_def_cent', 'v_def_cent', 'ellipse_h_def', 'ellipse_v_def',
                            'h_ref_cent', 'v_ref_cent', 'ellipse_h_ref', 'ellipse_v_ref',
                            'h_param_def', 'a_param_def', 'h_param_ref', 'a_param_ref'}
            
            if required_cols.issubset(common_cols):
                # If all required columns are present, proceed with those
                print(f"Combining datasets using {len(common_cols)} common columns.")
                combined_data = pd.concat([data1[list(common_cols)], data2[list(common_cols)]], ignore_index=True)
            else:
                # Missing required columns
                missing = required_cols - common_cols
                print(f"Cannot combine datasets: missing required columns: {missing}")
                # Use the first dataset if available
                if data1 is not None:
                    combined_data = data1
                    print("Proceeding with first dataset only.")
    elif data1 is not None:
        combined_data = data1
        print("Only first dataset available. Proceeding with it.")
    elif data2 is not None:
        combined_data = data2
        print("Only second dataset available. Proceeding with it.")
    else:
        print("No data files found. Exiting.")
        return
    
    # Check if we have data to analyze
    if combined_data.empty:
        print("No data to analyze. Exiting.")
        return
        
    # Remove duplicate rows if any
    initial_rows = len(combined_data)
    combined_data = combined_data.drop_duplicates(subset=['patient_id', 'timepoint', 'side'])
    if initial_rows > len(combined_data):
        print(f"Removed {initial_rows - len(combined_data)} duplicate rows.")
    
    data = combined_data  # Use the combined data for analysis
    print(f"Final dataset has {len(data)} rows.")
    print("Columns in data:", data.columns)
    
    # Create DataFrame for metrics
    metrics_df = pd.DataFrame(columns=[
        'patient_id', 'timepoint', 'side',
        'def_rmse', 'def_mae', 'def_max_error', 'def_r2', 
        'def_area_ratio', 'def_iou', 'def_dice', 'def_area_diff_pct',
        'ref_rmse', 'ref_mae', 'ref_max_error', 'ref_r2', 
        'ref_area_ratio', 'ref_iou', 'ref_dice', 'ref_area_diff_pct'
    ])
    
    # Process each row
    for i in range(len(data)):
        print(f"Processing row {i+1}/{len(data)}: Patient {data['patient_id'].iloc[i]} at timepoint {data['timepoint'].iloc[i]}")
        
        patient_id = data['patient_id'].iloc[i]
        timepoint = data['timepoint'].iloc[i]
        side = data['side'].iloc[i]
        
        # Initialize row for metrics DataFrame
        metrics_row = {
            'patient_id': patient_id,
            'timepoint': timepoint,
            'side': side
        }
        
        # Process deformed contour fit
        for name in ['def', 'ref']:
            print(f"  Analyzing {name} fit...")
            
            # Get contour and ellipse data
            try:
                h_cent = data[f'h_{name}_cent'].iloc[i]
                v_cent = data[f'v_{name}_cent'].iloc[i]
                ellipse_h = data[f'ellipse_h_{name}'].iloc[i]
                ellipse_v = data[f'ellipse_v_{name}'].iloc[i]
                h_param = data[f'h_param_{name}'].iloc[i]
                a_param = data[f'a_param_{name}'].iloc[i]
                
                # Calculate distance-based metrics
                rmse = calculate_rmse_ellipse_fit(h_cent, v_cent, ellipse_h, ellipse_v, h_param, a_param)
                mae = calculate_mae_ellipse_fit(h_cent, v_cent, ellipse_h, ellipse_v, h_param, a_param)
                max_error = calculate_max_error_ellipse_fit(h_cent, v_cent, ellipse_h, ellipse_v, h_param, a_param)
                r2 = calculate_r2_ellipse_fit(h_cent, v_cent, ellipse_h, ellipse_v, h_param, a_param)
                
                # Calculate area-based metrics
                area_ratio = calculate_area_ratio(h_cent, v_cent, ellipse_h, ellipse_v)
                
                # Calculate overlap metrics
                overlap_metrics = calculate_overlap_metrics(h_cent, v_cent, ellipse_h, ellipse_v)
                
                # Store metrics in row
                metrics_row[f'{name}_rmse'] = rmse
                metrics_row[f'{name}_mae'] = mae
                metrics_row[f'{name}_max_error'] = max_error
                metrics_row[f'{name}_r2'] = r2
                metrics_row[f'{name}_area_ratio'] = area_ratio
                metrics_row[f'{name}_iou'] = overlap_metrics['iou']
                metrics_row[f'{name}_dice'] = overlap_metrics['dice']
                metrics_row[f'{name}_area_diff_pct'] = overlap_metrics['area_diff_pct']
                
                # Visualize the fit
                visualization_metrics = {
                    'rmse': rmse,
                    'mae': mae,
                    'max_error': max_error,
                    'r2': r2,
                    'iou': overlap_metrics['iou'],
                    'dice': overlap_metrics['dice'],
                    'area_diff_pct': overlap_metrics['area_diff_pct'],
                    'h_param': h_param,
                    'a_param': a_param
                }
                if plot_ellipse_flag == True:
                    vis_path = visualize_fit_analysis(
                        h_cent, v_cent, ellipse_h, ellipse_v, 
                        visualization_metrics, patient_id, timepoint, side, name, 
                        output_dir=output_dir
                    )
                
                print(f"    Saved visualization to {vis_path}")
                print(f"    Metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, Max Error={max_error:.3f}, R²={r2:.3f}")
                print(f"    Area Metrics: IoU={overlap_metrics['iou']:.3f}, Dice={overlap_metrics['dice']:.3f}")
            
            except Exception as e:
                print(f"    Error analyzing {name} fit: {e}")
                # Set default values for metrics
                metrics_row[f'{name}_rmse'] = np.nan
                metrics_row[f'{name}_mae'] = np.nan
                metrics_row[f'{name}_max_error'] = np.nan
                metrics_row[f'{name}_r2'] = np.nan
                metrics_row[f'{name}_area_ratio'] = np.nan
                metrics_row[f'{name}_iou'] = 0
                metrics_row[f'{name}_dice'] = 0
                metrics_row[f'{name}_area_diff_pct'] = np.nan
        
        # Create combined visualization with both contours
        if plot_ellipse_flag == True:
            combined_path = create_combined_plot(data, i, metrics_row, output_dir)
            if combined_path:
                print(f"    Saved combined visualization to {combined_path}")
        
        # Add metrics row to DataFrame
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)
        
    # Save metrics to CSV
    output_path = os.path.join('Image_Processing_Scripts', output_filename)
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")
    
    # Calculate and display summary statistics
    print("\nSummary Statistics:")
    for name in ['def', 'ref']:
        print(f"\n{name.upper()} FIT METRICS:")
        print(f"  Average RMSE: {metrics_df[f'{name}_rmse'].mean():.3f}")
        print(f"  Average MAE: {metrics_df[f'{name}_mae'].mean():.3f}")
        print(f"  Average R²: {metrics_df[f'{name}_r2'].mean():.3f}")
        print(f"  Average IoU: {metrics_df[f'{name}_iou'].mean():.3f}")
        print(f"  Average Dice: {metrics_df[f'{name}_dice'].mean():.3f}")
    
    # Create summary visualizations
    create_summary_visualisations(metrics_df, output_dir)

def make_valid_polygon(points):
    """
    Create a valid Shapely polygon from a list of points.
    
    Parameters:
    points (list): List of (x, y) coordinate tuples
    
    Returns:
    ShapelyPolygon or None: Valid polygon or None if creation fails
    """
    if len(points) < 3:
        return None
    
    try:
        # Create a LinearRing to ensure the points form a valid ring
        ring = LinearRing(points)
        
        # Check if ring is valid
        if not ring.is_valid:
            # Try to fix with buffer
            buffered = ShapelyPolygon(points).buffer(0)
            if buffered.is_valid:
                return buffered
            else:
                return None
        else:
            # Create a polygon from the valid ring
            polygon = ShapelyPolygon(ring)
            
            # Final validity check
            if not polygon.is_valid:
                # Last resort: simplify slightly
                polygon = polygon.simplify(0.01)
                if polygon.is_valid:
                    return polygon
                else:
                    return None
            return polygon
    except Exception:
        return None

def simple_polygon_area(points):
    """
    Calculate the area of a polygon using the shoelace formula.
    
    Parameters:
    points (list): List of (x, y) coordinate tuples
    
    Returns:
    float: Area of the polygon
    """
    n = len(points)
    area = 0.0
    
    if n < 3:
        return area
        
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    area = abs(area) / 2.0
    return area

def create_combined_plot(data, i, metrics_row, output_dir):
    """
    Create combined visualization showing both deformed and reference contours with their fits.
    
    Parameters:
    data (DataFrame): The complete dataset
    i (int): Current row index in the dataset
    metrics_row (dict): Dictionary with calculated metrics
    output_dir (str): Directory to save visualizations
    
    Returns:
    str: Path to the saved file
    """
    try:
        # Extract patient info
        patient_id = data['patient_id'].iloc[i]
        timepoint = data['timepoint'].iloc[i]
        side = data['side'].iloc[i]
        
        # Set publication style
        set_publication_style()
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Colors
        def_color = '#FF5555'  # Bright red
        ref_color = '#5555FF'  # Bright blue
        def_dark = '#8B0000'   # Dark red
        ref_dark = '#00008B'   # Dark blue
        
        # Draw deformed contour and fit
        h_def_cont = data['h_def_cent'].iloc[i]
        v_def_cont = data['v_def_cent'].iloc[i]
        ellipse_h_def = data['ellipse_h_def'].iloc[i]
        ellipse_v_def = data['ellipse_v_def'].iloc[i]
        
        ax.scatter(h_def_cont, v_def_cont, color=def_color, s=10, alpha=0.7, 
                  edgecolor=def_dark, linewidth=0.5, label='Deformed Contour')
        ax.plot(ellipse_h_def, ellipse_v_def, color=def_dark, linewidth=2, 
               label='Deformed Ellipse Fit')
        
        # Draw reference contour and fit
        h_ref_cont = data['h_ref_cent'].iloc[i]
        v_ref_cont = data['v_ref_cent'].iloc[i]
        ellipse_h_ref = data['ellipse_h_ref'].iloc[i]
        ellipse_v_ref = data['ellipse_v_ref'].iloc[i]
        
        ax.scatter(h_ref_cont, v_ref_cont, color=ref_color, s=10, alpha=0.7, 
                  edgecolor=ref_dark, linewidth=0.5, label='Reference Contour')
        ax.plot(ellipse_h_ref, ellipse_v_ref, color=ref_dark, linewidth=2, 
               label='Reference Ellipse Fit')
        
        # Calculate area similarity from area difference
        def_area_similarity = metrics_row.get('def_area_similarity_pct', 100 - metrics_row.get('def_area_diff_pct', 0))
        ref_area_similarity = metrics_row.get('ref_area_similarity_pct', 100 - metrics_row.get('ref_area_diff_pct', 0))
        
        # Format metrics text for each type
        def_metrics = (
            f"Deformed Metrics:\n"
            f"RMSE: {metrics_row['def_rmse']:.3f}\n"
            f"R²: {metrics_row['def_r2']:.3f}\n"
            #f"Area Similarity: {def_area_similarity:.1f}%\n"
            f"Area Similarity: {metrics_row['def_area_diff_pct']:.1f}%\n"
            f"h: {data['h_param_def'].iloc[i]:.2f}"
        )
        
        ref_metrics = (
            f"Reference Metrics:\n"
            f"RMSE: {metrics_row['ref_rmse']:.3f}\n"
            f"R²: {metrics_row['ref_r2']:.3f}\n"
            #f"Area Similarity: {ref_area_similarity:.1f}%\n"
            f"Area Similarity: {metrics_row['ref_area_diff_pct']:.1f}%\n"
            f"h: {data['h_param_ref'].iloc[i]:.2f}"
        )
        
        # Position metrics text in left/right corners
        bbox_props_def = dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor=def_dark, alpha=0.8)
        bbox_props_ref = dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor=ref_dark, alpha=0.8)
        
        ax.text(0.03, 0.97, def_metrics, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left', bbox=bbox_props_def,
                color=def_dark)
        
        ax.text(0.97, 0.97, ref_metrics, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=bbox_props_ref,
                color=ref_dark)
        
        # Add title
        ax.set_title(f"Ellipse Fit Analysis - Patient {patient_id}, Timepoint {timepoint}")
        
        # Add legend with nice formatting - positioned outside plot area
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), frameon=True, framealpha=0.9, edgecolor='gray')
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Set y-axis limit to match original plots
        ax.set_ylim(top=60)
        
        # Add labels
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Adjust figure size to accommodate legend
        fig.tight_layout()
        fig.subplots_adjust(right=0.85)
        
        # Save figure
        filename = f"{patient_id}_{timepoint}_combined_fit_analysis.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    except Exception as e:
        print(f"Error creating combined plot: {e}")
        return None
    """
    Create summary visualizations of fit metrics.
    
    Parameters:
    metrics_df (DataFrame): DataFrame with metrics data
    output_dir (str): Directory to save visualizations
    """
    # Helper function to safely filter data
    def safe_data(column):
        return metrics_df[column].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Create boxplots for RMSE
    plt.figure(figsize=(10, 6))
    def_rmse = safe_data('def_rmse')
    ref_rmse = safe_data('ref_rmse')
    boxplot_data = [def_rmse, ref_rmse]
    plt.boxplot(boxplot_data, labels=['Deformed', 'Reference'])
    plt.title('RMSE Distribution for Ellipse Fits')
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    # Add counts to the plot
    plt.annotate(f"n={len(def_rmse)}", xy=(1, plt.ylim()[1]*0.95), ha='center')
    plt.annotate(f"n={len(ref_rmse)}", xy=(2, plt.ylim()[1]*0.95), ha='center')
    plt.savefig(os.path.join(output_dir, 'rmse_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create boxplots for R²
    plt.figure(figsize=(10, 6))
    def_r2 = safe_data('def_r2')
    ref_r2 = safe_data('ref_r2')
    # Filter out extreme R² values for better visualization
    def_r2 = def_r2[(def_r2 > -3) & (def_r2 < 1.5)]
    ref_r2 = ref_r2[(ref_r2 > -3) & (ref_r2 < 1.5)]
    boxplot_data = [def_r2, ref_r2]
    plt.boxplot(boxplot_data, labels=['Deformed', 'Reference'])
    plt.title('R² Distribution for Ellipse Fits')
    plt.ylabel('R²')
    plt.grid(True, linestyle='--', alpha=0.7)
    # Add counts to the plot
    plt.annotate(f"n={len(def_r2)}", xy=(1, plt.ylim()[1]*0.95), ha='center')
    plt.annotate(f"n={len(ref_r2)}", xy=(2, plt.ylim()[1]*0.95), ha='center')
    plt.savefig(os.path.join(output_dir, 'r2_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot of RMSE vs Area Difference %
    plt.figure(figsize=(10, 6))
    # Filter data for plot
    def_data = metrics_df[['def_rmse', 'def_area_diff_pct']].replace([np.inf, -np.inf], np.nan).dropna()
    ref_data = metrics_df[['ref_rmse', 'ref_area_diff_pct']].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Plot only if data is available
    if not def_data.empty:
        plt.scatter(def_data['def_rmse'], def_data['def_area_diff_pct'], 
                   color='red', alpha=0.7, label=f'Deformed (n={len(def_data)})')
    if not ref_data.empty:
        plt.scatter(ref_data['ref_rmse'], ref_data['ref_area_diff_pct'], 
                   color='blue', alpha=0.7, label=f'Reference (n={len(ref_data)})')
    
    plt.xlabel('RMSE')
    plt.ylabel('Area Difference (%)')
    plt.title('RMSE vs Area Difference for Ellipse Fits')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # Cap y-axis if extreme values present
    if plt.ylim()[1] > 100:
        plt.ylim(top=100)
    plt.savefig(os.path.join(output_dir, 'rmse_vs_area_diff.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create histogram of area difference percentages
    plt.figure(figsize=(12, 6))
    def_area_diff = safe_data('def_area_diff_pct')
    ref_area_diff = safe_data('ref_area_diff_pct')
    
    # Cap extreme values for better visualization
    def_area_diff = def_area_diff[def_area_diff <= 100]
    ref_area_diff = ref_area_diff[ref_area_diff <= 100]
    
    # Ensure we have data before plotting
    if len(def_area_diff) > 0:
        plt.hist(def_area_diff, bins=min(20, len(def_area_diff)), 
                alpha=0.7, color='red', label=f'Deformed (n={len(def_area_diff)})')
    if len(ref_area_diff) > 0:
        plt.hist(ref_area_diff, bins=min(20, len(ref_area_diff)), 
                alpha=0.7, color='blue', label=f'Reference (n={len(ref_area_diff)})')
    
    plt.xlabel('Area Difference (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Area Differences Between Contours and Fitted Ellipses')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'area_diff_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot of R² vs Max Error
    plt.figure(figsize=(10, 6))
    # Filter data
    def_error_data = metrics_df[['def_r2', 'def_max_error']].replace([np.inf, -np.inf], np.nan).dropna()
    ref_error_data = metrics_df[['ref_r2', 'ref_max_error']].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Filter out extreme R² values
    def_error_data = def_error_data[(def_error_data['def_r2'] > -3) & (def_error_data['def_r2'] < 1.5)]
    ref_error_data = ref_error_data[(ref_error_data['ref_r2'] > -3) & (ref_error_data['ref_r2'] < 1.5)]
    
    if not def_error_data.empty:
        plt.scatter(def_error_data['def_r2'], def_error_data['def_max_error'], 
                   color='red', alpha=0.7, label=f'Deformed (n={len(def_error_data)})')
    if not ref_error_data.empty:
        plt.scatter(ref_error_data['ref_r2'], ref_error_data['ref_max_error'], 
                   color='blue', alpha=0.7, label=f'Reference (n={len(ref_error_data)})')
    
    plt.xlabel('R²')
    plt.ylabel('Maximum Error')
    plt.title('R² vs Maximum Error for Ellipse Fits')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'r2_vs_max_error.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()