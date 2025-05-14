# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def position(t, initial_vel, g):
    return initial_vel*t + 0.5*g*t**2

# ilość czasów dodawanych do t_linspace z każdym odbiciem piłki
resolution = 100

# ilość kolejnych rzutów
n = 10

# grawitacja działa w dół
g = -9.81

# początkowa prędkość [m/s]
initial_vel = 10

# y0=[y1, y2]
# y1 -- początkowe położenie to 0, rzucamy piłkę w górę z ziemii
# y2 -- początkowa prędkość to 10 skierowane w górę
t_linspace = np.empty(0)
y_positions = np.empty(0)
y0 = [0, initial_vel]
offset = 0
for i in range(n):
    t_approx = 2 * y0[1] / (-g)
    t_return = fsolve(position, t_approx, args=(y0[1], g))

    t_sample = np.linspace(y0[0], t_return, resolution)

    t_linspace = np.append(t_linspace, t_sample + offset)

    y_positions = np.append(y_positions, np.array([position(t, y0[1], g) for t in t_sample]))

    y0[1] *= 0.8
    offset += t_return


print(t_linspace)
print(y_positions)
# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(t_linspace, y_positions, label=f"położenie piłki", color="blue")

# Add labels and legend
plt.title(f'Trajektoria lotu piłki - wykres $y_1(t)$, n={n}', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
