import numpy as np

# new x coord is vertical new y coord is horizontal
def switch_orientation(x_values, y_values, xx_coords, yy_coords):
    v_values=np.abs(x_values)
    h_values=y_values
    vv_coords=np.abs(xx_coords)
    hh_coords=yy_coords

    return v_values, h_values, vv_coords, hh_coords

def reverse_switch_orientation(v_values, h_values, vv_coords, hh_coords):
    x_values=v_values
    y_values=h_values
    xx_coords=vv_coords
    yy_coords=hh_coords

    print('note reverse_switch_orientation function does not reverse sign changes that may have occurred.')
    print('Any sign changes will have to be checked and corrected using switch_sign function.')

    return x_values, y_values, xx_coords, yy_coords

def switch_sign(x_values, y_values, xx_coords, yy_coords):
    x_values= -1*(x_values)
    y_values=y_values
    xx_coords= -1*(xx_coords)
    yy_coords=yy_coords

    return x_values, y_values, xx_coords, yy_coords
