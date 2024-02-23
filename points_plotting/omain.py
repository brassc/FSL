#import libraries
import numpy as np
import matplotlib.pyplot as plt

#user defined function
from load_np_data import load_data_readout

#load data
poly_func, x_values, y_values, xa_coords, ya_coords = load_data_readout('data_readout/deformed_arrays.npz')
polyb_func, xb_values, yb_values, xb_coords, yb_coords = load_data_readout('data_readout/baseline_arrays.npz')
polyr_func, xr_values, yr_values, xr_coords, yb_coords = load_data_readout('data_readout/reflected_baseline_arrays.npz')

#change orientation
# new x coord is vertical new y coord is horizontal
def switch_orientation(x_values, y_values, xx_coords, yy_coords):
    v_values=x_values
    h_values=y_values
    vv_coords=xx_coords
    hh_coords=yy_coords

    return v_values, h_values, vv_coords, hh_coords

def reverse_switch_orientation(v_values, h_values, vv_coords, hh_coords):
    x_values=v_values
    y_values=h_values
    xx_coords=vv_coords
    yy_coords=hh_coords

    return x_values, y_values, xx_coords, yy_coords


# DEFORMED VALUES
vd_values, hd_values, vd_coords, hd_coords = switch_orientation(x_values, y_values, xa_coords, ya_coords)
# BASELINE VALUES
vb_values, hb_values, vb_coords, hb_coords = switch_orientation(xb_values, yb_values, xb_coords, yb_coords)
# REFLECTED VALUES
vr_values, hr_values, vr_coords, hr_coords = switch_orientation(xr_values, yr_values, xr_coords, yb_coords)


# MAKE NICE PLOT - WIP
plt.plot(hr_values, vr_values, c='blue')
plt.plot(hd_values, vd_values, c='red')
plt.show()
