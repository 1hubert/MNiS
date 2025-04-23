from matplotlib import pyplot as plt
from numpy import pi
import numpy as np

def roznica_dzielona_centralna(f, x, h):
    return (f(x+h) - f(x-h)) / (2*h)

def f(x):
    if x < 0 or x > 5:
        return 0
        # raise ValueError('f(x) przyjmuje wartości w dziedzinie <0,5>')

    if 0 <= x < 1:
        return x * 10
    elif 1 <= x < 2:
        return 10
    elif 2 <= x < 3:
        return 10 + (x-2) * 10
    elif 3 <= x < 4:
        return 20
    else:  # 4 <= x <= 5
        return 20 - (x-4) * 20

def integrate_function_trap(x_values, func):
    integral = np.zeros_like(x_values)
    dx = x_values[1] - x_values[0]

    for i in range(1, len(x_values)):
        integral[i] = integral[i-1] + dx * (func(x_values[i]) + func(x_values[i-1])) / 2

    return integral

x_linspace = np.linspace(
    0,
    5,
    100
)

y_oryginal = []
y_derivative = []
for x in x_linspace:
    y_oryginal.append(f(x))
    y_derivative.append(roznica_dzielona_centralna(f, x, 0.01))

y_integral = integrate_function_trap(x_linspace, f)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_linspace, y_integral, label="$\int$ f(x) - funkcja drogi", color="blue")
plt.plot(x_linspace, y_oryginal, label="f(x) - prędkość chwilowa", color="red")
plt.plot(x_linspace, y_derivative, label="f'(x) - przyspieszenie chwilowe", color="purple")


# Show the plot
plt.title(f'Wykres drogi/prędkości/przyspieszenia robota mobilnego', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
