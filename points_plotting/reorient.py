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

