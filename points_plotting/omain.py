#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#user defined function
from load_np_data import load_data_readout
from reorient import switch_orientation # there is also a reverse_switch_orientation function
from translate_rotate import move


#load data
patient_id = '19978'
patient_timepoint="acute"

data_readout_loc = f"points_plotting/data_readout/{patient_id}_{patient_timepoint}"
print(data_readout_loc)
poly_func, x_values, y_values, xa_coords, ya_coords = load_data_readout(data_readout_loc, 'deformed_arrays.npz')
polyb_func, xb_values, yb_values, xb_coords, yb_coords = load_data_readout(data_readout_loc, 'baseline_arrays.npz')
polyr_func, xr_values, yr_values, xr_coords, yb_coords = load_data_readout(data_readout_loc, 'reflected_baseline_arrays.npz')

#change orientation 
# new x coord is vertical, v new y coord is horizontal, h
# DEFORMED VALUES
vd_values, hd_values, vd_coords, hd_coords = switch_orientation(x_values, y_values, xa_coords, ya_coords)
# BASELINE VALUES
vb_values, hb_values, vb_coords, hb_coords = switch_orientation(xb_values, yb_values, xb_coords, yb_coords)
# REFLECTED VALUES
vr_values, hr_values, vr_coords, hr_coords = switch_orientation(xr_values, yr_values, xr_coords, yb_coords)


# CALCULATE AREA
aread, _ = quad(poly_func, hb_values[0], hb_values[-1])
arear, _ = quad(polyr_func, hb_values[0], hb_values[-1])
area_betw = np.abs(aread-arear)
area_betw=round(area_betw, 2)
print(area_betw)


# MAKE NICE PLOT

plt.plot(hr_values, vr_values, c='cyan')
plt.plot(hd_values, vd_values, c='red')
plt.fill_between(hd_values, vd_values, vr_values, color='orange', alpha=0.5 )
plt.text(70, 165, f'Area = {area_betw} mm^2', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
plt.scatter(hr_coords[0], vr_coords[0])
plt.text(hr_coords[0], vr_coords[0], 'hr_coords[0], vr_coords[0]')
plt.scatter(hr_coords[-1], vr_coords[-1])
plt.text(hr_coords[-1], vr_coords[-1], 'hr_coords[-1], vr_coords[-1]')
plt.show()


# TRANSLATE and ROTATE SO THAT POINTS LIE ON HORIZONTAL AXIS
tr_hr_coords, tr_vr_coords = move(hr_coords, vr_coords)
tr_hd_coords, tr_vd_coords = move(hd_coords, vd_coords)
tr_hr_values, tr_vr_values = move(hr_values, vr_values)
tr_hd_values, tr_vd_values = move(hd_values, vd_values)

# PLOT TRANSLATED AND ROTATED FUNCTION
plt.scatter(tr_hr_coords, tr_vr_coords, c='cyan', s=2)
plt.scatter(tr_hd_coords, tr_vd_coords, c='red', s=2)
plt.plot(tr_hd_values, tr_vd_values, c='red')
plt.plot(tr_hr_values, tr_vr_values, c='cyan')
plt.xlim(0)
plt.ylim(0)
plt.show()




 