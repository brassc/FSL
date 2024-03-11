import pandas as pd
import nibabel as nib
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from translate_rotate import move


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

"""
def func(x, a, b, c, d, e):
    return (np.sqrt(1-(a*x-d)*np.sqrt(1+(b*x-e)+c)))

"""

# Define the function that represents the upper portion of an ellipse
def func(x, h, a, c, d):
    # To ensure we only deal with the upper portion, we return NaN if the inside of the sqrt becomes negative
    with np.errstate(invalid='ignore'):
        y = h * np.sqrt(np.maximum(0, a**2 - (x - c)**2)) + d
    return y
"""
# Define a cost function that calculates the total squared error
def cost_function(params, h_coords, v_coords):
    h, a, c, d = params
    predicted = func(h_coords, h, a, c, d)
    return np.sum((v_coords - predicted) ** 2)

# Define the constraints by ensuring the function passes through the endpoints
def constraint1(params, h_coords, v_coords, tol):
    h, a, c, d = params
    return func(h_coords[0], h, a, c, d) - v_coords[0] - tol #should be 0

def constraint2(params, h_coords, v_coords, tol):
    h, a, c, d = params
    return func(h_coords[-1], h, a, c, d) - v_coords[-1] - tol # should be 0
"""


"""
def func3(x, a, b, c, d, e):
    return (np.sqrt(a*(1-(x-d)))*np.sqrt(b*(1+(x-e)))+c)

def func2(x, a, c, d):
    return (np.sqrt(1-(a*x**2-d))+c)


def constraint_func(params, x_endpoints, y_endpoints):
    a, b, c, d, e = params
    return [func(params, x_endpoints[0]) - y_endpoints[0], func(params, x_endpoints[-1]) - y_endpoints[-1]]

def residual(params, x, y):
    return func(params[:5], x) - y
"""

def update_c(initial_guess, h_coords, v_coords, weights):
    # Extract initial guess parameters
    h0, a, c, d = initial_guess
    
    # Specify target point
    target_point = (h_coords[0], v_coords[0])
    
    # Define a tolerance for convergence
    tolerance = 50
    
    # Maximum number of iterations
    max_iterations = 100
    
    # Initialize iteration counter
    iteration = 0
    
    # Loop until convergence or maximum iterations reached
    while True:
        # Perform curve fitting with updated parameters
        params, covariance = curve_fit(func, h_coords, v_coords, p0=(h0, a, c, d))#, sigma=weights)
        
        # Update parameter c
        c = c - (func(target_point[0], *params) - target_point[1]) / func(target_point[0], *params)
        
        # Check convergence
        if abs(func(target_point[0], *params) - target_point[1]) < tolerance or iteration >= max_iterations:
            break
        
        # Increment iteration counter
        iteration += 1
    
    # Update initial guess tuple with the new value of c
    updated_initial_guess = (h0, a, c, d)
    
    # Return updated initial guess tuple
    return updated_initial_guess

## THIS FUNCTION RETURNS A POLYNOMIAL APPROXIMATION OF A y=sqrt(1-x^2) FUNCTION GIVEN H_COORDS AND V_COORDS ALREADY TRANSLATED AND ROTATED. 
def approx_poly(h_coords, v_coords):
    print(h_coords)
    print(v_coords)

    # Define the weights
    weights = np.ones_like(h_coords)
    weights[0] = 5 # Increase weight for the first point
    weights[-1] = 5  # Increase weight for the last point
    print(h_coords[0])

    
    

    ## Calculate the mean of the first and last entries
    #mean_first_last = np.mean([h_coords[0], h_coords[-1]])
    #h_coords_centered=h_coords - mean_first_last


    # Decrease weights gradually towards the middle
    """middle_idx = int(len(weights) // 2)
    for n in range(middle_idx):
        weights[n+1]=weights[0]**(1/(n+1))
        weights[-1-n-1]=weights[-1]**(1/(n+1))
    #weights[:middle_idx] *= np.linspace(1000, 1, middle_idx)
    """ 
    try:
        # Perform curve fitting
        # initial_guess = (10, 80, -180, 2500, 2500)
        h = v_coords.max()
        a = h_coords.max() - h_coords.min()
        c = a / 2 # middle value
        d = v_coords.min()
        initial_guess=(h, a, c, d) 
        #updated_initial_guess = update_c(initial_guess, h_coords, v_coords, weights)
        params, covariance = curve_fit(func, h_coords, v_coords, p0=initial_guess, sigma=weights)
        ##popt, _ = curve_fit(func, h_coords, v_coords, p0=(10, 80, -180, 2500, 2500), sigma=weights)
        #params, covariance = curve_fit(func, h_coords, v_coords, p0=initial_guess, sigma=weights)
        
        ##popt2, _ = curve_fit(func2, h_coords, v_coords, p0=(0.005, 135, 80))
        """
        # Solving for 'd' using both endpoints
        # At x = x1, y = y1
        h, a, c, _ = params
        x1, y1 = h_coords[0], v_coords[0]
        d1 = y1 - h * np.sqrt(np.maximum(0, a**2 - (x1 - c)**2))

        # At x = x2, y = y2
        x2, y2 = h_coords[-1], v_coords[-1]
        d2 = y2 - h * np.sqrt(np.maximum(0, a**2 - (x2 - c)**2))

        # One approach could be to average d1 and d2 to find a reconciled 'd' value
        d_avg = (d1 + d2) / 2

        # Adjusting our parameter list
        params_adjusted = [h, a, c, d_avg]
        """
        """
        # Adjust 'd' to make sure the curve goes through the first endpoint
        h, a, c, _ = params
        d_adjusted = v_coords[0] - h * np.sqrt(np.maximum(0, a**2 - (h_coords[0] - c)**2))

        # Apply the adjustment
        params[-1] = d_adjusted
        """ 
        # Extract the optimal parameter
        h_optimal_init, a_optimal_init, c_optimal_init, d_optimal_init = params
        print(h_optimal_init)
        print(a_optimal_init)
        #print(b_optimal)
        print(c_optimal_init)
        print(d_optimal_init)
        #print(e_optimal)
        h_optimal, a_optimal, c_optimal, d_optimal = params


    
        
        """
        # Set constraints
        # Modifying optimization constraints to use local scope variables
        tol = 1E-2
        cons = [{'type': 'eq', 'fun': lambda params: constraint1(params, h_coords, v_coords, tol)},
                {'type': 'eq', 'fun': lambda params: constraint2(params, h_coords, v_coords, tol)}]

        
        # Use minimize to find the best parameters under the constraints
        # Optimize parameters under constraints
        result = minimize(lambda params: cost_function(params, h_coords, v_coords), initial_guess, constraints=cons)


        # Extract the optimal parameters
        h_optimal, a_optimal, c_optimal, d_optimal = result.x
        """
        print(h_optimal)
        print(a_optimal)
        #print(b_optimal)
        print('c is')
        print(c_optimal)
        print(d_optimal)

        new_c = h_coords[0] - (v_coords[0]-d_optimal)/(h_optimal*a_optimal)
        print('new c is')
        print(new_c)

        


            
        
        # Generate x values using the fitted function
        h_values = np.linspace(min(h_coords), max(h_coords), 100)
        #v_fitted = func(h_values, a_optimal, b_optimal, c_optimal, d_optimal, e_optimal)
        v_fitted = func(h_values, h_optimal, a_optimal, new_c, d_optimal)
        #v_fitted = func2(h_values, a_optimal, c_optimal, d_optimal)
        """
        X=h_coords.reshape(-1, 1)  # Ensuring X is a column vector
        Y=v_coords.reshape(-1, 1)  # Ensuring X is a column vector

        # Formulate and solve the least squares problem ||Ax - b ||^2
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = np.ones((len(X), 1))  # Explicitly defining b as a 2D column vector
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        x = x.squeeze()

        # Print the equation of the ellipse in standard form
        print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1],x[2],x[3],x[4]))

        # Plot the noisy data
        plt.scatter(X, Y, label='Data Points')

        # Plot the least squares ellipse
        x_coord = np.linspace(-5,5,300)
        y_coord = np.linspace(-5,5,300)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
        plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(0)
        plt.ylim(0)
        plt.show()
        """

        return a_optimal, h_values, v_fitted
    except RuntimeError:
        print("Curve fitting failed. Try adjusting parameters or check your data.")




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




