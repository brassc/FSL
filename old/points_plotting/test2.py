
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import eig


def fit_ellipse(x, y, bb):
    # Check if x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    
    # Build design matrix
    D = np.column_stack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    # Build scatter matrix
    S = np.dot(D.T, D)
    # Build 6x6 constraint matrix
    b8 = 8 / (bb*bb)
    b64 = b8*b8
    C = np.zeros((6, 6))
    C[(0, 0)] = 1
    C[(0, 2)] = 1
    C[(1, 1)] = -1
    C[(2, 0)] = 1
    C[(2, 2)] = 1
    C[(0, 5)] = b8
    C[(5, 0)] = b8
    C[(5, 2)] = b8
    C[(2, 5)] = b8
    C[(5, 5)] = b64
    # Solve eigensystem
    geval, gevec = eig(S, C)
    geval = np.real_if_close(geval)
    print("geval:", geval)
    print("gevec:", gevec)
    dminindex = np.argmax(np.real(geval))
    print("dminindex:", dminindex)
    # Extract eigenvector corresponding to negative eigenvalue
    A = np.real(gevec[:, dminindex])
    print('A in fit_ellipse is')
    print(A)
    # Get ellipse parameters
    par = getpar(A)
    return par



def getpar(A):
    a, b, c, d, e, f = A
    # Compute ellipse parameters
    num = 2 * (a*f*f + c*d*d + e*b*b - 2*b*d*e - a*c*f)
    den = (b*b - a*c) * np.sqrt((a - c)*(a - c) + 4*b*b) - (a + c)*(a + c)
    
    if den <= 0 or np.isclose(den, 0):
        # Handle case where denominator is non-positive
        A = 0
        B = 0
    else:
        A = np.sqrt(num / den)
        B = np.sqrt(num / den)
        
    cx = (2*c*d - 2*b*e) / (b*b - a*c)
    cy = (2*a*e - 2*b*d) / (b*b - a*c)
    # Angle
    theta = 0.5 * np.arctan((2*b) / (a - c))
    # Return parameters
    return {'A': A, 'B': B, 'center': (cx, cy), 'angle': theta}

# Example usage
x = np.array([95.46967004, 84.46234386, 73.90504033, 63.45198863, 53.10513761, 42.94686109,
 32.89706958, 23.14972321, 13.60400694,  6.53311059,  0.        ])
x = x/10
y = np.array([ 0.,          9.60073879, 14.59072159, 18.55264902, 21.46727314, 22.52237299,
 22.50764317, 19.51042042, 14.52486903,  8.78589364,  0.        ])
y=y/10
bb = 6
result = fit_ellipse(x, y, bb)
print(result)





"""
alpha = 5
beta = 3
N = 500
DIM = 2

np.random.seed(2)

# Generate random points on the unit circle by sampling uniform angles
theta = np.random.uniform(0, 2*np.pi, (N,1))
eps_noise = 0.2 * np.random.normal(size=[N,1])
circle = np.hstack([np.cos(theta), np.sin(theta)])

# Stretch and rotate circle to an ellipse with random linear tranformation
B = np.random.randint(-3, 3, (DIM, DIM))
noisy_ellipse = circle.dot(B) + eps_noise

# Extract x coords and y coords of the ellipse as column vectors
X = noisy_ellipse[:,0:1]
Y = noisy_ellipse[:,1:]

print(X.shape)
print(X[0])

X = np.array([95.46967004, 84.46234386, 73.90504033, 63.45198863, 53.10513761, 42.94686109,
 32.89706958, 23.14972321, 13.60400694,  6.53311059,  0.        ])
X=X.reshape(len(X), 1)
print(X.shape)

Y=np.array([[ 0.,          9.60073879, 14.59072159, 18.55264902, 21.46727314, 22.52237299,
 22.50764317, 19.51042042, 14.52486903,  8.78589364,  0.        ]]).reshape(11,1)
print(Y.shape)
"""
"""
y_mean = np.mean(Y)
upper_half_filter = Y >= y_mean
X = X[upper_half_filter].reshape(-1, 1)
Y = Y[upper_half_filter].reshape(-1, 1)
"""
"""
# Formulate and solve the least squares problem ||Ax - b ||^2
A = np.hstack([X**2, X * Y, Y**2, X, Y])
b = np.ones_like(X)
x = np.linalg.lstsq(A, b)[0].squeeze()



# Print the equation of the ellipse in standard form
print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1],x[2],x[3],x[4]))

# Plot the noisy data
plt.scatter(X, Y, label='Data Points')

# Plot the original ellipse from which the data was generated
phi = np.linspace(0, 2*np.pi, 1000).reshape((1000,1))
c = np.hstack([np.cos(phi), np.sin(phi)])
ground_truth_ellipse = c.dot(B)
plt.plot(ground_truth_ellipse[:,0], ground_truth_ellipse[:,1], 'k--', label='Generating Ellipse')

# Plot the least squares ellipse
x_coord = np.linspace(-5,5,300)
y_coord = np.linspace(-5,5,300)
X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
"""
# Plot data points
plt.scatter(x, y, color='blue', label='Data points')

# Plot ellipse
ellipse = Ellipse(result['center'], result['A']*2, result['B']*2, angle=np.degrees(result['angle']),
                  edgecolor='red', fill=False, label='Fitted Ellipse')
plt.gca().add_patch(ellipse)
print('result is')
print(result)


plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()