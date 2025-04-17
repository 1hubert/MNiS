"""różnica dzielona w tył"""
from math import pi, sin
import numpy as np
import matplotlib.pyplot as plt


def f(x, freq, max_k):
    omega = 2 * pi * freq
    result = (8 / (pi ** 2))
    sum1 = 0
    for k in range(0, max_k + 1):
        sum1 += ((-1)**k) * sin((2*k+1)*omega*x) / ((2*k+1)**2)

    return result * sum1

y1 = lambda x: f(x, 0.25, 500)

def roznica_dzielona_w_tyl(f, x, h):
    """Różnica dzielona centralna"""
    return (f(x) - f(x-h)) / (h)

# Plotting
x_linspace = np.linspace(
    -5,
    5,
    100
)

y_pochodna = []
for x in x_linspace:
    y_pochodna.append(roznica_dzielona_w_tyl(y1, x, 0.02))

y_oryginal = []
for x in x_linspace:
    y_oryginal.append(y1(x))


# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_linspace, y_oryginal, label=f"f(x)", color="red")
plt.plot(x_linspace, y_pochodna, label=f"f'(x)", color="blue")

# Add labels and legend
plt.title(f'Różnica dzielona w tył', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
