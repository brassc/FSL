import pandas as pd
import nibabel as nib
import numpy as np


## THIS FUNCTION RETURNS A POLYNOMIAL AND VALUES TO PLOT FROM GIVEN Y AND X COORDS
def fit_poly(y_coords, x_coords, order=2):

    ## POLYNOMIAL FITTING

    # include weights 
    # Assign weights, with higher values for the first and last points
    weights = np.ones(len(y_coords))
    weights[0] = weights[-1] = 10  # Adjust this value to emphasize the first and last points more
    #weights[3] = weights[4] = 5
    #weights[1] = weights[-2] = 9  

    # Fit a polynomial of order specified (default 2) relating x to y
   
    coefficients = np.polyfit(y_coords, x_coords, order, w=weights)

    # Create a polynomial function using the coefficients
    poly_func = np.poly1d(coefficients)

    # Print the polynomial equation
    print("Polynomial Equation:")
    print(poly_func)

    # Generate points for the fitted polynomial curve
    poi_df_max_index=np.size(x_coords)-1
    y_values = np.linspace(y_coords[0],y_coords[poi_df_max_index], 100)
    x_values = poly_func(y_values)

    return poly_func, x_values, y_values


# THIS FUNCTION TAKES A .CSV FILE AND RETURNS POLYNOMIAL FUNCTION, X AND Y VALUES TO PLOT. 
# NOTE Y AND X AXES ARE REVERSED FROM WHAT IT TYPICAL DUE TO THE ORIENTATION OF THE POLYNOMIAL. LINE 23 coefficients = np.polyfit(y_coords, x_coords, 2)

def create_polynomial_from_csv(poi_csv, affine, order):
    poi_df=pd.read_csv(poi_csv)
    #print(poi_df)
    
    ## POINTS OF INTEREST

    transformed_points = []
    for index, row in poi_df.iterrows():
        point = np.array([row[0], row[1], row[2], 1])
        transformed_point = np.linalg.inv(affine).dot(point)
        transformed_points.append(transformed_point)

    # extract x and y coordinates
    transformed_points=np.array(transformed_points)
    x_coords = transformed_points[:, 0]
    y_coords = transformed_points[:, 1]


    ## POLYNOMIAL FITTING

    poly_func, x_values, y_values=fit_poly(y_coords, x_coords, order)
    """
    # Fit a polynomial of degree 2 relating x to y
    coefficients = np.polyfit(y_coords, x_coords, 2)

    # Create a polynomial function using the coefficients
    poly_func = np.poly1d(coefficients)

    # Print the polynomial equation
    print("Polynomial Equation:")
    print(poly_func)

    # Generate points for the fitted polynomial curve
    poi_df_max_index=np.size(x_coords)-1
    y_values = np.linspace(y_coords[0],y_coords[poi_df_max_index], 100)
    x_values = poly_func(y_values)
    """
    
    return poly_func, x_values, y_values, x_coords, y_coords




