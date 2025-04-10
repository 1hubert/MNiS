from math import pi, sin
import numpy as np
import matplotlib.pyplot as plt


# f = lambda x, omega, max_k:  * sum([(-1)**k * sin((2*k+1)*omega*x) / ((2*k+1)**2) for k in range(0, max_k+1)])

def f(x, freq, max_k):
    omega = 2 * pi * freq
    result = (8 / (pi ** 2))
    sum1 = 0
    for k in range(0, max_k + 1):
        sum1 += ((-1)**k) * sin((2*k+1)*omega*x) / ((2*k+1)**2)

    return result * sum1

# roznica_dzielona_w_przod = lambda x:


# Plotting
x_linspace = np.linspace(
    -500,
    500,
    100
)



# y1 = lambda x: f(x_linspace, 0.25, 5)
y = []
for x in x_linspace:
    y.append(f(x, 1, 500))

# y = y1(x_linspace)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_linspace, y, label=f"wielomian stopnia", color="blue")
# plt.scatter(x_points, y_points, color="red", s=75, label="dane", zorder=5)  # Red points

# Add labels and legend
plt.title(f'Interpolacja wielomianowa', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
