
import numpy as np
import matplotlib.pyplot as plt

def divided_differences(x, y):
    """
    Compute the divided differences table for Newton interpolation.

    Args:
        x: List of x-coordinates (nodes).
        y: List of y-coordinates (function values at nodes).

    Returns:
        A list containing the coefficients for the Newton polynomial.
    """
    n = len(x)
    coef = y.copy()  # Initial coefficients are the y-values

    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])

    return coef

def newton_interpolation(x_data, y_data, x_eval):
    """
    Evaluate the Newton interpolating polynomial at a given point x_eval.

    Args:
        x_data: List of x-coordinates (nodes).
        y_data: List of y-coordinates (function values at nodes).
        x_eval: The point at which to evaluate the polynomial.

    Returns:
        The interpolated value at x_eval.
    """
    coef = divided_differences(x_data, y_data)
    n = len(x_data)
    result = coef[-1]  # Start with the highest-order coefficient

    for i in range(n-2, -1, -1):
        result = result * (x_eval - x_data[i]) + coef[i]

    return result

# Example usage:
x_points = [911.3, 636.0, 451.1]  # Example x-coordinates
y_points = [30.131, 40.12, 50.128]  # Example y-coordinates
x_to_evaluate = 3      # Point to evaluate

# Compute and print the interpolated value
interpolated_value = newton_interpolation(x_points, y_points, x_to_evaluate)

print(f"Interpolated value at x = {x_to_evaluate}: {interpolated_value}")

b_list = divided_differences(x_points, y_points)
print(f'b0, b1, b2, ... = {b_list}')

# Plotting
x_linspace = np.linspace(x_points[-1], x_points[0], 100)

def func(b_list, x_points, R):
    val = b_list[0]

    for i in range(1, len(b_list)):
        wspolczynnik = 1
        for j in range(0, i):
            wspolczynnik *= (R - x_points[j])

        print(f'wspolczynnik {i} = {wspolczynnik}')
        val += b_list[i] * wspolczynnik

    return val


# y = lambda x: b_list[0] + [b_list[i] * x_points[i+1] for i in range(1, len(b_list)+1)]


y1 = lambda x: func(b_list, x_points, x)
y = y1(x_linspace)
# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_linspace, y, label="f", color="blue")
plt.scatter(x_points, y_points, color="red", s=100, label="Points", zorder=5)  # Red points

# Add labels and legend
plt.title("Function Plot with Highlighted Points", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
