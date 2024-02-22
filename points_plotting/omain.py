
import numpy as np

from load_np_data import load_data_readout

#load data
poly_func, x_values, y_values, xa_coords, ya_coords = load_data_readout('data_readout/baseline_arrays.npz')
polyb_func, xb_values, yb_values, xb_coords, yb_coords = load_data_readout('data_readout/deformed_arrays.npz')
polyr_func, xr_values, yr_values, xr_coords, yb_coords = load_data_readout('data_readout/reflected_baseline_arrays.npz')


