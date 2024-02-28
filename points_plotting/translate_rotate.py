import numpy as np

def move(h, v):
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

    return tr_h_coords, tr_v_coords