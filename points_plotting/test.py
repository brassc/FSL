from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

y_coords = [0, 1, 2, 3]
x_coords = [-1, 1, 3, 5]
order = 1

coefficients = np.polyfit(y_coords, x_coords, order)

# Create a polynomial function using the coefficients
polynomial_function = np.poly1d(coefficients)

# Integrate the polynomial function to find the area between curve and y-axis
area, _ = quad(polynomial_function, 0, 4)

print("Numerical value of the area between curve and y-axis:", area)



# Plotting
y_values = np.linspace(0, 4, 1000)  # Generate y values
x_values = polynomial_function(y_values)  # Compute corresponding x values

plt.plot(x_values, y_values, label='Polynomial Curve')
plt.fill_betweenx(y_values, x_values, color='lightblue', alpha=0.5, label='Area between curve and y-axis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Area between Curve and Y-axis')
plt.legend()
plt.grid(True)
plt.show()
