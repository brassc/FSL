from matplotlib.lines import Line2D
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



def calculate_dice_from_masks(contour_x, contour_y, ellipse_x, ellipse_y): 
    # Create binary masks 
    min_x = min(np.min(contour_x), np.min(ellipse_x)) 
    max_x = max(np.max(contour_x), np.max(ellipse_x)) 
    min_y = min(np.min(contour_y), np.min(ellipse_y)) 
    max_y = max(np.max(contour_y), np.max(ellipse_y))

    # Create grid 
    grid_size = 100 
    x_grid = np.linspace(min_x, max_x, grid_size) 
    y_grid = np.linspace(min_y, max_y, grid_size) 
    XX, YY = np.meshgrid(x_grid, y_grid)

    # Create masks \
    contour_points = list(zip(contour_x, contour_y)) 
    ellipse_points = list(zip(ellipse_x, ellipse_y)) 
    contour_path = Path(contour_points) 
    ellipse_path = Path(ellipse_points)

    contour_mask = contour_path.contains_points(np.vstack([XX.flatten(), YY.flatten()]).T).reshape(XX.shape) 
    ellipse_mask = ellipse_path.contains_points(np.vstack([XX.flatten(), YY.flatten()]).T).reshape(XX.shape)

    # Calculate Dice 
    intersection = np.logical_and(contour_mask, ellipse_mask).sum() 
    dice = (2.0 * intersection) / (contour_mask.sum() + ellipse_mask.sum())

    return dice


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
    contour_x, contour_y = resample(ellipse_x, contour_x, contour_y)
    distances = []
    
    # For each contour point, find the minimum distance to the ellipse
    for i in range(len(contour_x)):
        point = (contour_x[i], contour_y[i])
        dist = point_to_ellipse_distance(point, h_param, a_param, ellipse_x, ellipse_y)
        distances.append(dist)
    
    # Calculate RMSE

    rmse = np.sqrt(np.mean(np.array(distances)**2))
    print(f"RMSE raw values: {distances[:5]}, calculated RMSE: {rmse}")
    return rmse


def resample(ellipse_x, contour_x, contour_y):
    # Check for empty or single-point contours
    if len(contour_x) <= 1:
        print("Warning: Contour has insufficient points for resampling")
        return contour_x, contour_y
        
    num_ellipse_points = len(ellipse_x)
    print(f"Ellipse points: {num_ellipse_points}, Contour points: {len(contour_x)}")

    # Sort the contour points by x-coordinate
    # Sort (contour_x, contour_y) by contour_x
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
    
    print(f"Total contour length: {total_length}, Segments: {len(segment_lengths)}")
    
    # Resample at equal intervals along the contour
    for i in range(num_ellipse_points):
        # Position along the contour (normalized)
        target_dist = (i / (num_ellipse_points - 1 if num_ellipse_points > 1 else 1)) * total_length
        
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
    
    print(f"Generated {len(resampled_x)} resampled points")
    
    # Ensure we have multiple points
    if len(resampled_x) <= 1:
        print("Warning: Resampling produced too few points, returning original contour")
        return contour_x, contour_y
    
    # #plot with thickness of 10
    # plt.scatter(contour_x, contour_y, label='Original Contour', linewidth=10)
    # plt.plot(resampled_x, resampled_y, label='Resampled Contour')
    # plt.legend()
    # plt.show()
        
    return np.array(resampled_x), np.array(resampled_y)

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
    # resample
    contour_x, contour_y = resample(ellipse_x, contour_x, contour_y)

    distances = []
    # For each contour point, find the minimum distance to the ellipse
    for i in range(len(contour_x)):
        point = (contour_x[i], contour_y[i])
        dist = point_to_ellipse_distance(point, h_param, a_param, ellipse_x, ellipse_y)
        distances.append(dist)
    
    # Calculate MAE
    mae = np.mean(np.abs(distances))
    print(f"MAE raw values: {distances[:5]}, calculated MAE: {mae}")
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
        # Calculate Dice coefficient using masks approach as primary method
        dice = calculate_dice_from_masks(contour_x, contour_y, ellipse_x, ellipse_y)
        
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
                'dice': dice,  # Use the mask-based Dice calculation
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
        
        # We're using the mask-based Dice calculation instead of the polygon-based one
        # so we don't calculate dice here anymore
        
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
            'dice': dice,  # Use the mask-based Dice calculation
            'area_diff_pct': min(area_diff_pct, 100.0)  # Cap at 100%
        }
    except Exception as e:
        print(f"Error calculating overlap metrics: {e}")
        # Fallback to simple area difference calculation
        try:
            # Always try to calculate the mask-based Dice first
            try:
                dice = calculate_dice_from_masks(contour_x, contour_y, ellipse_x, ellipse_y)
            except Exception as dice_error:
                print(f"Error calculating mask-based Dice: {dice_error}")
                dice = 0
            
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
                'dice': dice,
                'area_diff_pct': min(area_diff_pct, 100.0)  # Cap at 100%
            }
        except Exception as e:
            print(f"Error in fallback calculation: {e}")
            # Try one last time to at least get the dice from masks
            try:
                dice = calculate_dice_from_masks(contour_x, contour_y, ellipse_x, ellipse_y)
            except:
                dice = 0
                
            return {
                'contour_area': 0,
                'ellipse_area': 0,
                'intersection_area': 0,
                'iou': 0,
                'dice': dice,
                'area_diff_pct': 100.0
            }


def calculate_dice_from_masks(contour_x, contour_y, ellipse_x, ellipse_y):
    """
    Calculate Dice coefficient using binary masks.
    
    Parameters:
    contour_x, contour_y (array): Coordinates of contour points
    ellipse_x, ellipse_y (array): Coordinates of ellipse points
    
    Returns:
    float: Dice coefficient
    """
    import numpy as np
    from matplotlib.path import Path
    from scipy.interpolate import interp1d
    
    # Create binary masks with extended boundaries
    min_x = min(np.min(contour_x), np.min(ellipse_x))
    max_x = max(np.max(contour_x), np.max(ellipse_x))
    min_y = min(np.min(contour_y), np.min(ellipse_y))
    max_y = max(np.max(contour_y), np.max(ellipse_y))
    
    # Add padding to ensure we capture the full shapes
    padding = 0.05  # 5% padding
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    min_x -= x_range * padding
    max_x += x_range * padding
    min_y -= y_range * padding
    max_y += y_range * padding
    
    # Resample the contour points to create a continuous curve
    # First, sort the contour points by x-coordinate
    contour_indices = np.argsort(contour_x)
    contour_x_sorted = np.array(contour_x)[contour_indices]
    contour_y_sorted = np.array(contour_y)[contour_indices]
    
    # Create a high-resolution grid for x
    grid_size = 500  # High resolution for both curves
    x_grid = np.linspace(min_x, max_x, grid_size)
    y_grid = np.linspace(min_y, max_y, grid_size)
    XX, YY = np.meshgrid(x_grid, y_grid)
    
    # Generate filled masks for both shapes
    # For the contour (point cloud), use linear interpolation
    try:
        # Use linear interpolation to draw straight lines between points
        contour_interp = interp1d(contour_x_sorted, contour_y_sorted, 
                                 kind='linear', 
                                 bounds_error=False, 
                                 fill_value=(contour_y_sorted[0], contour_y_sorted[-1]))
        
        # Generate y-values for the contour curve
        contour_curve_y = contour_interp(x_grid)
        
        # Create a filled contour mask
        contour_mask = np.zeros_like(XX, dtype=bool)
        for i, x_val in enumerate(x_grid):
            y_val = contour_curve_y[i]
            if not np.isnan(y_val):
                # Fill all points below the curve (assuming semi-ellipse)
                contour_mask[:, i] = YY[:, i] <= y_val
    except:
        # If interpolation fails, fall back to the original path method
        contour_points = list(zip(contour_x, contour_y))
        # Close the path if it's not already closed
        if contour_points[0] != contour_points[-1]:
            contour_points.append(contour_points[0])
        contour_path = Path(contour_points)
        points = np.vstack([XX.flatten(), YY.flatten()]).T
        contour_mask = contour_path.contains_points(points).reshape(XX.shape)
    
    # For the ellipse, use the same linear approach
    try:
        # Sort ellipse points by x-coordinate
        ellipse_indices = np.argsort(ellipse_x)
        ellipse_x_sorted = np.array(ellipse_x)[ellipse_indices]
        ellipse_y_sorted = np.array(ellipse_y)[ellipse_indices]
        
        # Linear interpolation for ellipse curve
        ellipse_interp = interp1d(ellipse_x_sorted, ellipse_y_sorted, 
                                 kind='linear', 
                                 bounds_error=False, 
                                 fill_value=(ellipse_y_sorted[0], ellipse_y_sorted[-1]))
        
        # Generate y-values for the ellipse curve
        ellipse_curve_y = ellipse_interp(x_grid)
        
        # Create a filled ellipse mask
        ellipse_mask = np.zeros_like(XX, dtype=bool)
        for i, x_val in enumerate(x_grid):
            y_val = ellipse_curve_y[i]
            if not np.isnan(y_val):
                # Fill all points below the curve (assuming semi-ellipse)
                ellipse_mask[:, i] = YY[:, i] <= y_val
    except:
        # If interpolation fails, fall back to the original path method
        ellipse_points = list(zip(ellipse_x, ellipse_y))
        # Close the path if it's not already closed
        if ellipse_points[0] != ellipse_points[-1]:
            ellipse_points.append(ellipse_points[0])
        ellipse_path = Path(ellipse_points)
        points = np.vstack([XX.flatten(), YY.flatten()]).T
        ellipse_mask = ellipse_path.contains_points(points).reshape(XX.shape)
    
    # Calculate Dice coefficient
    intersection = np.logical_and(contour_mask, ellipse_mask).sum()
    total = contour_mask.sum() + ellipse_mask.sum()
    
    if total > 0:
        dice = (2.0 * intersection) / total
    else:
        dice = 0.0
    
    return dice


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
        dice_from_masks = calculate_dice_from_masks(contour_x, contour_y, ellipse_x, ellipse_y)
        
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
        f"MAE: {metrics['mae']:.3f}\n"
        f"Dice: {metrics['dice']:.3f}\n"
        #f"R²: {metrics['r2']:.3f}\n"
        #f"Area Similarity: {metrics['area_diff_pct']:.1f}%\n"
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
    metrics_to_plot = ['rmse', 'mae']
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
    deformed_median = np.median(deformed_rmse)
    reference_median = np.median(reference_rmse)
    combined_median = np.median(np.concatenate([deformed_rmse, reference_rmse]))
    print(f"Combined RMSE median: {combined_median}")

    # Add thin horizontal lines for median
    #ax1.axhline(y=deformed_median, xmin=0.15, xmax=0.35, color=def_color, linestyle='-', linewidth=1)
    #ax1.axhline(y=reference_median, xmin=0.7, xmax=0.8, color=ref_color, linestyle='-', linewidth=1)

    # add median text
    # Instead of horizontal lines, add text labels for medians
    # Create a boxed annotation for RMSE plot
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    median_text = (f"Deformed median: {deformed_median:.2f}\n"
                f"Reference median: {reference_median:.2f}\n"
                f"Combined median: {combined_median:.2f}")

    # Position the text box in the upper left corner
    ax1.text(0.025, 0.95, median_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)


    # Set titles and labels
    ax1.set_ylim(0, 10)
    ax1.set_title('RMSE by Configuration')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Root Mean Square Error')

    # MAE plot (right)
    mae_data = long_df[long_df['metric_name'] == 'MAE']

    # Set up x-axis
    ax2.set_ylim(0, 10)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Deformed', 'Reference'])

    # Get data for each configuration
    deformed_mae = mae_data[mae_data['configuration'] == 'Deformed']['metric_value'].values
    reference_mae = mae_data[mae_data['configuration'] == 'Reference']['metric_value'].values

    # Add only scatter points for Deformed
    ax2.scatter(
        [0] * len(deformed_mae),
        deformed_mae,
        color=def_color, s=10, alpha=0.7,
        edgecolor=def_color,  # Match fill color to remove grey rings
        linewidth=0.5
        )

    # Add only scatter points for Reference
    ax2.scatter(
        [1] * len(reference_mae),
        reference_mae,
        color=ref_color, s=10, alpha=0.7,
        edgecolor=ref_color,  # Match fill color to remove grey rings
        linewidth=0.5
    )

    # Optional: Add subtle statistical indicators manually
    # Calculate statistics
    deformed_median_mae = np.median(deformed_mae)
    reference_median_mae = np.median(reference_mae)

    # Add thin horizontal lines for median
    ax2.axhline(y=deformed_median_mae, xmin=0.2, xmax=0.3, color=def_color, linestyle='-', linewidth=1)
    ax2.axhline(y=reference_median_mae, xmin=0.7, xmax=0.8, color=ref_color, linestyle='-', linewidth=1)

    # Set titles and labels
    ax2.set_title('MAE by Configuration')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Mean Absolute Error')

    
    # Final adjustments
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_distribution_resampled.png'), dpi=300)
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/scatter_distribution_resampled.pdf', dpi=300)
    plt.close()

    # Create separate figures for RMSE and MAE plots

    # RMSE Plot (first figure)
    fig1 = plt.figure(figsize=(8, 7))
    ax1 = fig1.add_subplot(111)

    # RMSE Plot content
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
        edgecolor=def_color, # Match fill color to remove grey rings
        linewidth=0.5
    )
    # Add only scatter points for Reference
    ax1.scatter(
        [1] * len(reference_rmse),
        reference_rmse,
        color=ref_color, s=10, alpha=0.7,
        edgecolor=ref_color, # Match fill color to remove grey rings
        linewidth=0.5
    )
    # Calculate statistics
    deformed_median = np.median(deformed_rmse)
    reference_median = np.median(reference_rmse)
    combined_median = np.median(np.concatenate([deformed_rmse, reference_rmse]))

    deformed_q1 = np.percentile(deformed_rmse, 25)
    deformed_q3 = np.percentile(deformed_rmse, 75)
    reference_q1 = np.percentile(reference_rmse, 25)
    reference_q3 = np.percentile(reference_rmse, 75)
    # Add thin horizontal lines for median
    #ax1.axhline(y=deformed_median, xmin=0.2, xmax=0.3, color=def_color, linestyle='-', linewidth=1)
    #ax1.axhline(y=reference_median, xmin=0.7, xmax=0.8, color=ref_color, linestyle='-', linewidth=1)
    # Set titles and labels
    ax1.set_ylim(0, 10)
    ax1.set_title('RMSE by Configuration')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Root Mean Square Error')


    # Add thin horizontal lines for median
    #ax1.axhline(y=deformed_median, xmin=0.15, xmax=0.35, color=def_color, linestyle='-', linewidth=1)
    #ax1.axhline(y=reference_median, xmin=0.7, xmax=0.8, color=ref_color, linestyle='-', linewidth=1)

    # add median text
    # Instead of horizontal lines, add text labels for medians
    # Create a boxed annotation for RMSE plot
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    median_text = (f"Deformed median: {deformed_median:.2f}\n"
                f"Reference median: {reference_median:.2f}\n"
                f"Combined median: {combined_median:.2f}")

    # Position the text box in the upper left corner
    plt.text(0.025, 0.95, median_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)


    # Save RMSE figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_scatter.png'), dpi=300)
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/rmse_scatter.pdf', dpi=300)
    plt.close(fig1)

    # MAE Plot (second figure)
    fig2 = plt.figure(figsize=(8, 7))
    ax2 = fig2.add_subplot(111)

    # MAE plot content
    mae_data = long_df[long_df['metric_name'] == 'MAE']
    # Set up x-axis
    ax2.set_ylim(0, 10)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Deformed', 'Reference'])
    # Get data for each configuration
    deformed_mae = mae_data[mae_data['configuration'] == 'Deformed']['metric_value'].values
    reference_mae = mae_data[mae_data['configuration'] == 'Reference']['metric_value'].values
    
    # Add only scatter points for Deformed
    ax2.scatter(
        [0] * len(deformed_mae),
        deformed_mae,
        color=def_color, s=10, alpha=0.7,
        edgecolor=def_color, # Match fill color to remove grey rings
        linewidth=0.5
    )
    # Add only scatter points for Reference
    ax2.scatter(
        [1] * len(reference_mae),
        reference_mae,
        color=ref_color, s=10, alpha=0.7,
        edgecolor=ref_color, # Match fill color to remove grey rings
        linewidth=0.5
    )
    # Calculate statistics
    deformed_median_mae = np.median(deformed_mae)
    reference_median_mae = np.median(reference_mae)
    combined_median_mae = np.median(np.concatenate([deformed_mae, reference_mae]))
    
    # Create a boxed annotation for RMSE plot
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    median_text = (f"Deformed median: {deformed_median_mae:.2f}\n"
                f"Reference median: {reference_median_mae:.2f}\n"
                f"Combined median: {combined_median_mae:.2f}")

    # Position the text box in the upper left corner
    plt.text(0.025, 0.95, median_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    
    # Add thin horizontal lines for median
    #ax2.axhline(y=deformed_median_mae, xmin=0.2, xmax=0.3, color=def_color, linestyle='-', linewidth=1)
    #ax2.axhline(y=reference_median_mae, xmin=0.7, xmax=0.8, color=ref_color, linestyle='-', linewidth=1)
    # Set titles and labels
    ax2.set_title('MAE by Configuration')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Mean Absolute Error')

    # Save MAE figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_scatter.png'), dpi=300)
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/mae_scatter.pdf', dpi=300)
    plt.close(fig2)




    # Calculate mean and std for Dice coefficient
    def_dice_mean = metrics_df['def_dice'].mean()
    def_dice_std = metrics_df['def_dice'].std()
    ref_dice_mean = metrics_df['ref_dice'].mean()
    ref_dice_std = metrics_df['ref_dice'].std()

    # Calculate mean and std for MAE
    def_mae_mean = metrics_df['def_mae'].mean()
    def_mae_std = metrics_df['def_mae'].std()
    ref_mae_mean = metrics_df['ref_mae'].mean()
    ref_mae_std = metrics_df['ref_mae'].std()

    def_label = (f"$\\mathbf{{Deformed:}}$\n"
                f"Mean Dice: {def_dice_mean:.3f} ± {def_dice_std:.3f}\n"
                f"Mean MAE: {def_mae_mean:.2f} ± {def_mae_std:.2f} [mm]")

    ref_label = (f"$\\mathbf{{Reference:}}$\n"
                f"Mean Dice: {ref_dice_mean:.3f} ± {ref_dice_std:.3f}\n"
                f"Mean MAE: {ref_mae_mean:.2f} ± {ref_mae_std:.2f} [mm]")
    
    # 2. Scatter plot of RMSE vs R²
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(metrics_df['def_dice'], metrics_df['def_mae'], 
                          color=def_color, 
                          alpha=0.7, 
                        #   label='Deformed',
                          label=def_label,
                          edgecolors='black', 
                          linewidth=0.5)
    plt.scatter(metrics_df['ref_dice'], metrics_df['ref_mae'], 
                color=ref_color, 
                alpha=0.7, 
                # label='Reference',
                label=ref_label,
                edgecolors='black', 
                linewidth=0.5)
    
    plt.ylabel('Mean Absolute Error (MAE) [mm]')
    plt.xlabel('Dice Coefficient')
    plt.title('MAE vs Dice by Configuration')
    plt.ylim(0,8)
    plt.xlim(0,1)
    legend = plt.legend(loc='upper left', fontsize=12, framealpha=0.9, 
          fancybox=True)

    # for text in legend.get_texts():
    #     text.set_verticalalignment('top')
    #     # Add some padding to the top of text to align with marker center
    #     # text.set_position((text.get_position()[0], text.get_position()[1] + 0.005))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_mae_dice.png'))
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/scatter_mae_dice.pdf', dpi=300)
    plt.close()


    ## VIOLIN PLOTS - NEW
    # RMSE Violin Plot
    fig_violin = plt.figure(figsize=(8, 7))
    ax_violin = fig_violin.add_subplot(111)
    
    # Prepare data for violin plot
    rmse_violin_data = [deformed_rmse, reference_rmse]
    
    # Create violin plot
    parts = ax_violin.violinplot(rmse_violin_data, positions=[0, 1], showmeans=False, showmedians=False, showextrema=False)
    
    # Customize violin colors
    parts['bodies'][0].set_facecolor(def_color)
    parts['bodies'][0].set_alpha(0.3)  # More transparent for better point visibility
    parts['bodies'][1].set_facecolor(ref_color)
    parts['bodies'][1].set_alpha(0.3)

    # Clip ONLY the deformed (left) violin to y=10
    deformed_violin = parts['bodies'][0]
    vertices = deformed_violin.get_paths()[0].vertices
    vertices[:, 1] = np.clip(vertices[:, 1], 0, 8.55)  # Clip to your axis range

    # Customize violin colors (continue with your existing code...)
    parts['bodies'][0].set_facecolor(def_color)
    # Customize median lines (your existing code continues...)
    # parts['cmedians'].set_colors([def_dark, ref_dark])


    
    # Customize median lines
    # parts['cmedians'].set_colors([def_dark, ref_dark])
    # parts['cmedians'].set_linewidth(2)

    # Add quartile lines for RMSE
    deformed_q1_rmse = np.percentile(deformed_rmse, 25)
    deformed_q3_rmse = np.percentile(deformed_rmse, 75)
    reference_q1_rmse = np.percentile(reference_rmse, 25)
    reference_q3_rmse = np.percentile(reference_rmse, 75)
    # ax_violin.hlines([deformed_q1_rmse, deformed_q3_rmse], -0.1, 0.1, colors=def_dark, linestyles='--', alpha=0.8, linewidth=1)
    # ax_violin.hlines([reference_q1_rmse, reference_q3_rmse], 0.9, 1.1, colors=ref_dark, linestyles='--', alpha=0.8, linewidth=1)

    # Smart jitter for deformed RMSE - inline density-aware jitter
    np.random.seed(42)
    if len(deformed_rmse) < 2:
        deformed_x_jitter = np.full_like(deformed_rmse, 0)
    else:
        from scipy.stats import gaussian_kde
        kde_def = gaussian_kde(deformed_rmse)
        density_def = kde_def(deformed_rmse)
        max_density_def = np.max(density_def)
        jitter_amount_def = (density_def / max_density_def) * 0.15
        direction_def = np.random.choice([-1, 1], size=len(deformed_rmse))
        jitter_def = np.random.uniform(-jitter_amount_def, jitter_amount_def) * direction_def
        deformed_x_jitter = 0 + jitter_def
    
    # Smart jitter for reference RMSE - inline density-aware jitter
    if len(reference_rmse) < 2:
        reference_x_jitter = np.full_like(reference_rmse, 1)
    else:
        kde_ref = gaussian_kde(reference_rmse)
        density_ref = kde_ref(reference_rmse)
        max_density_ref = np.max(density_ref)
        jitter_amount_ref = (density_ref / max_density_ref) * 0.15
        direction_ref = np.random.choice([-1, 1], size=len(reference_rmse))
        jitter_ref = np.random.uniform(-jitter_amount_ref, jitter_amount_ref) * direction_ref
        reference_x_jitter = 1 + jitter_ref

    # Add jittered scatter points with improved styling
    ax_violin.scatter(
        deformed_x_jitter,
        deformed_rmse,
        color=def_color, s=12, alpha=0.6,
        edgecolor='white',
        linewidth=0.3,
        zorder=3
    )

    ax_violin.scatter(
        reference_x_jitter,
        reference_rmse,
        color=ref_color, s=12, alpha=0.6,
        edgecolor='white',
        linewidth=0.3,
        zorder=3
    )

    # # Add sample size annotations
    # ax_violin.text(0, -0.5, f'n={len(deformed_rmse)}', ha='center', fontsize=8, color=def_dark)
    # ax_violin.text(1, -0.5, f'n={len(reference_rmse)}', ha='center', fontsize=8, color=ref_dark)

    # # Add statistical test
    # from scipy.stats import mannwhitneyu
    # statistic_rmse, p_value_rmse = mannwhitneyu(deformed_rmse, reference_rmse)
    # if p_value_rmse < 0.001:
    #     p_text_rmse = "p < 0.001"
    # else:
    #     p_text_rmse = f"p = {p_value_rmse:.3f}"
    # ax_violin.text(0.5, ax_violin.get_ylim()[1] * 0.85, p_text_rmse, ha='center', fontsize=9, 
    #                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Set up axes
    ax_violin.set_xlim(-0.5, 1.5)
    ax_violin.set_xticks([0, 1])
    ax_violin.set_xticklabels(['Deformed', 'Reference'])
    ax_violin.set_ylim(0, 10)
    ax_violin.set_title('RMSE by Configuration')
    ax_violin.set_xlabel('Configuration')
    ax_violin.set_ylabel('Root Mean Square Error')
    
    # Add median text box
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    median_text = (f"Deformed median: {deformed_median:.2f}\n"
                f"Reference median: {reference_median:.2f}\n"
                f"Combined median: {combined_median:.2f}")
    
    plt.text(0.025, 0.95, median_text, transform=ax_violin.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Save RMSE violin plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_violin.png'), dpi=300)
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/rmse_violin.pdf', dpi=300)
    plt.close(fig_violin)

    
    # MAE Violin Plot
    fig_mae_violin = plt.figure(figsize=(8, 7))
    ax_mae_violin = fig_mae_violin.add_subplot(111)
    
    # Prepare data for violin plot
    mae_violin_data = [deformed_mae, reference_mae]
    
    # Create violin plot
    parts_mae = ax_mae_violin.violinplot(mae_violin_data, positions=[0, 1], showmeans=False, showmedians=False, showextrema=False)
    
    # Customize violin colors
    parts_mae['bodies'][0].set_facecolor(def_color)
    parts_mae['bodies'][0].set_alpha(0.5)  # More transparent for better point visibility
    parts_mae['bodies'][1].set_facecolor(ref_color)
    parts_mae['bodies'][1].set_alpha(0.5)
    
    # Customize median lines
    # parts_mae['cmedians'].set_colors([def_dark, ref_dark])
    # parts_mae['cmedians'].set_linewidth(2)

    # Add quartile lines for MAE
    deformed_q1_mae = np.percentile(deformed_mae, 25)
    deformed_q3_mae = np.percentile(deformed_mae, 75)
    reference_q1_mae = np.percentile(reference_mae, 25)
    reference_q3_mae = np.percentile(reference_mae, 75)
    # ax_mae_violin.hlines([deformed_q1_mae, deformed_q3_mae], -0.1, 0.1, colors=def_dark, linestyles='--', alpha=0.8, linewidth=1)
    # ax_mae_violin.hlines([reference_q1_mae, reference_q3_mae], 0.9, 1.1, colors=ref_dark, linestyles='--', alpha=0.8, linewidth=1)

    # Smart jitter for deformed MAE - inline density-aware jitter
    np.random.seed(42)
    if len(deformed_mae) < 2:
        deformed_x_jitter_mae = np.full_like(deformed_mae, 0)
    else:
        from scipy.stats import gaussian_kde
        kde_def_mae = gaussian_kde(deformed_mae)
        density_def_mae = kde_def_mae(deformed_mae)
        max_density_def_mae = np.max(density_def_mae)
        jitter_amount_def_mae = (density_def_mae / max_density_def_mae) * 0.15
        direction_def_mae = np.random.choice([-1, 1], size=len(deformed_mae))
        jitter_def_mae = np.random.uniform(-jitter_amount_def_mae, jitter_amount_def_mae) * direction_def_mae
        deformed_x_jitter_mae = 0 + jitter_def_mae
    
    # Smart jitter for reference MAE - inline density-aware jitter
    if len(reference_mae) < 2:
        reference_x_jitter_mae = np.full_like(reference_mae, 1)
    else:
        kde_ref_mae = gaussian_kde(reference_mae)
        density_ref_mae = kde_ref_mae(reference_mae)
        max_density_ref_mae = np.max(density_ref_mae)
        jitter_amount_ref_mae = (density_ref_mae / max_density_ref_mae) * 0.15
        direction_ref_mae = np.random.choice([-1, 1], size=len(reference_mae))
        jitter_ref_mae = np.random.uniform(-jitter_amount_ref_mae, jitter_amount_ref_mae) * direction_ref_mae
        reference_x_jitter_mae = 1 + jitter_ref_mae

    # Add jittered scatter points with improved styling
    ax_mae_violin.scatter(
        deformed_x_jitter_mae,
        deformed_mae,
        color=def_color, s=12, alpha=0.3,
        edgecolor='white',
        linewidth=0.3,
        zorder=3
    )

    ax_mae_violin.scatter(
        reference_x_jitter_mae,
        reference_mae,
        color=ref_color, s=12, alpha=0.3,
        edgecolor='white',
        linewidth=0.3,
        zorder=3
    )

    # Add sample size annotations
    # ax_mae_violin.text(0, -0.5, f'n={len(deformed_mae)}', ha='center', fontsize=8, color=def_dark)
    # ax_mae_violin.text(1, -0.5, f'n={len(reference_mae)}', ha='center', fontsize=8, color=ref_dark)

    # # Add statistical test
    # from scipy.stats import mannwhitneyu
    # statistic_mae, p_value_mae = mannwhitneyu(deformed_mae, reference_mae)
    # if p_value_mae < 0.001:
    #     p_text_mae = "p < 0.001"
    # else:
    #     p_text_mae = f"p = {p_value_mae:.3f}"
    # ax_mae_violin.text(0.5, ax_mae_violin.get_ylim()[1] * 0.85, p_text_mae, ha='center', fontsize=9, 
    #                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Set up axes
    ax_mae_violin.set_xlim(-0.5, 1.5)
    ax_mae_violin.set_xticks([0, 1])
    ax_mae_violin.set_xticklabels(['Deformed', 'Reference'])
    ax_mae_violin.set_ylim(0, 10)
    ax_mae_violin.set_title('MAE by Configuration')
    ax_mae_violin.set_xlabel('Configuration')
    ax_mae_violin.set_ylabel('Mean Absolute Error')
    
    # Calculate MAE medians and add text box
    combined_median_mae = np.median(np.concatenate([deformed_mae, reference_mae]))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    median_text_mae = (f"Deformed median: {deformed_median_mae:.2f}\n"
                    f"Reference median: {reference_median_mae:.2f}\n"
                    f"Combined median: {combined_median_mae:.2f}")
    
    plt.text(0.025, 0.95, median_text_mae, transform=ax_mae_violin.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Save MAE violin plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_violin.png'), dpi=300)
    plt.savefig('../Thesis/phd-thesis-template-2.4/Chapter5/Figs/mae_violin.pdf', dpi=300)
    plt.close(fig_mae_violin)
    
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
    plot_ellipse_flag = True
    
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
                if plot_ellipse_flag == True:
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

    print(metrics_df.columns)
    # print all patient ids in metrics_df row 'patient_id'
    print(metrics_df['patient_id'])
        
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
            f"MAE: {metrics_row['def_mae']:.3f}\n"
            f"Dice: {metrics_row['def_dice']:.3f}\n"
            #f"R²: {metrics_row['def_r2']:.3f}\n"
            #f"Area Similarity: {def_area_similarity:.1f}%\n"
            
            #f"Area Similarity: {metrics_row['def_area_diff_pct']:.1f}%\n"
            f"h: {data['h_param_def'].iloc[i]:.2f}"
        )
        
        ref_metrics = (
            f"Reference Metrics:\n"
            f"RMSE: {metrics_row['ref_rmse']:.3f}\n"
            f"MAE: {metrics_row['ref_mae']:.3f}\n"
            f"Dice: {metrics_row['ref_dice']:.3f}\n"
            #f"R²: {metrics_row['ref_r2']:.3f}\n"
            #f"Area Similarity: {ref_area_similarity:.1f}%\n"
            #f"Area Similarity: {metrics_row['ref_area_diff_pct']:.1f}%\n"
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