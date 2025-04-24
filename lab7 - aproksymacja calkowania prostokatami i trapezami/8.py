from numpy import sin, pi, cos
import numpy as np
from matplotlib import pyplot as plt

def q_rect(a, b, n, func):
    area = 0
    h = (b-a) / n
    a_rect = a
    b_rect = a + h
    for _ in range(n):
        area += h * func((a_rect+b_rect)/2)

        a_rect += h
        b_rect += h
    return area


def q_trap(a, b, n, func):
    area = 0
    h = (b-a) / n
    a_rect = a
    b_rect = a + h
    for _ in range(n):
        area += h * ((func(a_rect)+func(b_rect))/2)

        a_rect += h
        b_rect += h
    return area


def integrate_function_trap(x_values, func):
    integral = np.zeros_like(x_values)
    dx = x_values[1] - x_values[0]

    for i in range(1, len(x_values)):
        integral[i] = integral[i-1] + dx * (func(x_values[i]) + func(x_values[i-1])) / 2

    return integral

def integrate_function_rect(x_values, func):
    integral = np.zeros_like(x_values)
    dx = x_values[1] - x_values[0]

    for i in range(1, len(x_values)):
        integral[i] = integral[i-1] + dx * func((x_values[i] + x_values[-1]/2))

    return integral

def f_pulse(t, T, tau, max_n):
    """
    t - argument
    T - okres funkcji
    tau - czas trwania impulsu
    max_n - maksymalna wartość n w sumowaniu / maks. wartość iteracji
    """
    sum1 = 0
    for n in range(1, max_n + 1):
        sum1 += (2 / (n * pi)) * sin((pi * n * tau) / T) * cos(((pi * n * tau) / T) * t)
    return tau/T + sum1

y1 = lambda x: f_pulse(x, 1, 0.75, 100)

# Porównanie metod całkowania
print(f'Całka f_pulse metodą prostokątów: {q_rect(0, 4*pi, 100, y1)}')
print(f'Całka f_pulse metodą trapezów: {q_trap(0, 4*pi, 100, y1)}')

x_linspace = np.linspace(
    0,
    10,
    100
)

y_oryginal = []
for x in x_linspace:
    y_oryginal.append(y1(x))

y_integral_trap = integrate_function_trap(x_linspace, y1)
y_integral_rect = integrate_function_rect(x_linspace, y1)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_linspace, y_oryginal, label=f"f(x)", color="red")
plt.plot(x_linspace, y_integral_trap, label=f"całka f(x) metodą trapezów", color="blue")
plt.plot(x_linspace, y_integral_rect, label=f"całka f(x) metodą prostokątów", color="purple")


# Show the plot
plt.title(f'f(x)', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
