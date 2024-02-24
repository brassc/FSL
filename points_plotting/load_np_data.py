
import numpy as np

def load_data_readout(data_readout_loc, filename):
    # load data
    file = f"{data_readout_loc}/{filename}"
    data = np.load(file)
    # access arrays
    poly_func = np.poly1d(data['poly_func']) 
    x_values = data['x_values']
    y_values = data['y_values']
    xx_coords = data['xx_coords']
    yy_coords = data['yy_coords']

    # return them
    return poly_func, x_values, y_values, xx_coords, yy_coords