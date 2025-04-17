from math import pi, sin
import numpy as np
import matplotlib.pyplot as plt


def f(x, omega, max_k):
    # omega = 2 * pi * freq
    sum1 = 0
    for k in range(0, max_k + 1):
        sum1 += sin(2*pi*(2*k-1)*omega*x) / (2*k-1)

    return (4/pi) * sum1


y1 = lambda x: f(x, 0.1, 500)

# Plotting
x_linspace = np.linspace(
    0,
    20,
    100
)

y_oryginal = []
for x in x_linspace:
    y_oryginal.append(y1(x))


# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_linspace, y_oryginal, label=f"f(x)", color="red")


# Show the plot
plt.title(f'f(x)', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
