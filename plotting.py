import numpy as np
import matplotlib.pyplot as plt

# Define the function: y = x^2 - 2x + 1
def func(x):
    return x**2 - 2*x + 1

# Generate x values
x = np.linspace(-5, 5, 100)  # 100 points from -5 to 5
y = func(x)
print(y)

# Coordinates to highlight (as red points)
points_x = [-2, 0, 1, 3]
points_y = func(np.array(points_x))  # Calculate y-values for these x-coordinates

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="y = x² - 2x + 1", color="blue")  # Plot the function
plt.scatter(points_x, points_y, color="red", s=100, label="Points", zorder=5)  # Red points

# Add labels and legend
plt.title("Function Plot with Highlighted Points", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
