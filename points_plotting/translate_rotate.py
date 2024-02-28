import numpy as np
from numpy.polynomial.polynomial import Polynomial

def move(h, v, poly=0):
    rotation_angle = np.arctan((v[-1]-v[0])/(h[0]-h[-1]))

    # rotation matrix
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [np.sin(rotation_angle), np.cos(rotation_angle)],
                            ])
    
    group_coords=np.vstack((h, v))
    rot_coords=np.dot(rotation_matrix, group_coords)

    left_magnitude=rot_coords[0,-1]
    down_magnitude=rot_coords[1, 0]  
    tr_v_coords=rot_coords[1]-down_magnitude
    tr_h_coords=rot_coords[0]-left_magnitude

    print(f"Rotation angle: {rotation_angle}, Translation by ({left_magnitude}, {down_magnitude})")

    if poly != 0:
        coefficients_rotated = np.dot(rotation_matrix, poly.coef)
        poly_rotated = Polynomial(coefficients_rotated)
        translation_vector = np.array([left_magnitude, down_magnitude])
        poly_rt_coef = poly_rotated + translation_vector
        poly_rt=Polynomial(poly_rt_coef)
        print(f"Polynomial:\n {poly_rt}")
        return tr_h_coords, tr_v_coords, poly_rt
    else:
        return tr_h_coords, tr_v_coords, None


    