import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression



def get_mirror_line(yb_coords, xa_coords, xb_coords):
    #where b is baseline and a is expansion side
    # returns gradient, m; x intercept, c; fit data, Y
    y_coords_df = pd.DataFrame({'y': yb_coords})

    avg_x = ((xa_coords + xb_coords) / 2)
    avg_x = pd.DataFrame({'avg_x': avg_x})

    #TRIM TO ONLY INCLUDE FIRST AND LAST - LINE TO ONLY GO THROUGH FIRST AND LAST POINTS
    first_row = avg_x.iloc[[0]]
    last_row = avg_x.iloc[[-1]]
    tr_avg_x = pd.concat([first_row, last_row])

    firsty_row=y_coords_df.iloc[[0]]
    lasty_row=y_coords_df.iloc[[-1]]
    tr_y_coords_df=pd.concat([firsty_row, lasty_row])

    # Reshape your x and y data for sklearn
    x = tr_avg_x.values#.reshape(-1, 1)  # Reshaping is required for a single feature in sklearn
    Y = tr_y_coords_df['y'].values.reshape(-1,1)

    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model to your data
    model.fit(Y, x)

    # The slope (gradient m) and intercept (c) from the fitted model
    m = model.coef_[0]
    c = model.intercept_

    print('Gradient (m) is:', m)
    print('Intercept (c) is:', c)

    return m, c, Y

    
