import numpy as np
from numpy import pi, sin
import matplotlib.pyplot as plt


def q_rect(a, b, n, func):
    area = 0
    h = (b-a) / n
    a_rect = a
    b_rect = a + h
    for _ in range(n):
        area += h * func((a_rect+b_rect) / 2)

        a_rect += h
        b_rect += h
    return area


def q_trap(a, b, n, func):
    area = 0
    h = (b-a) / n
    a_rect = a
    b_rect = a + h
    for _ in range(n):
        area += h * (func(a_rect)+func(b_rect)) / 2

        a_rect += h
        b_rect += h
    return area

def integrate_function_trap(x_values, func):
    integral = np.zeros_like(x_values)
    dx = x_values[1] - x_values[0]

    for i in range(1, len(x_values)):
        integral[i] = integral[i-1] + dx * (func(x_values[i]) + func(x_values[i-1])) / 2

    return integral


def f(x, omega, max_k):
    """
    Compute the Fourier series approximation of a square wave.

    Parameters:
    t : array_like
        Time points where to evaluate the function
    omega : float
        Angular frequency of the wave
    n_terms : int
        Number of terms to include in the approximation

    Returns:
    array_like
        The approximated square wave values
    """
    result = 0
    for k in range(1, max_k + 1):
        harmonic = 2*k - 1
        result += sin(2 * pi * harmonic * omega * x) / harmonic

    result *= 4 / pi

    return result * 4 / pi


y1 = lambda x: f(x, 1/(2*pi), 1000)

# Porównanie metod całkowania
print(f'Całka metodą prostokątów: {q_rect(0, 4*pi, 100, y1)}')
print(f'Całka metodą trapezów: {q_trap(0, 4*pi, 100, y1)}')

x_linspace = np.linspace(
    0,
    4*pi,
    100
)

y_oryginal = []
for x in x_linspace:
    y_oryginal.append(y1(x))

y_integral = integrate_function_trap(x_linspace, y1)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_linspace, y_oryginal, label=f"f(x)", color="red")
plt.plot(x_linspace, y_integral, label=f"całka f(x) dla dwóch okresów", color="blue")


# Show the plot
plt.title(f'f(x)', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
