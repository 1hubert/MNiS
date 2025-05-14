# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def system_of_equations(t, y):
    y1, y2 = y
    dy1dt = y2
    dy2dt = -9.81
    return dy1dt, dy2dt

def zero_crossing(t, y):
    """This function returns 0 when y = 0"""
    if t < 0.1:
        return 1.0  # Return non-zero value to avoid event detection
    return y[0]

# Setting terminal = True tells solve_ivp to stop integration when the event occurs
zero_crossing.terminal = True

# Setting direction = 0 detects zero crossings in both directions
zero_crossing.direction = 0

# ilość kolejnych rzutów
n = 10

t_linspace = np.empty(0)
y_positions = np.empty(0)
velocity = 10
offset = 0
for i in range(n):
    solution = solve_ivp(
        system_of_equations,
        [0, 5],
        [0, velocity],
        method='RK45',
        events=zero_crossing,
        t_eval=np.linspace(0,5,1000)
    )

    t_linspace = np.append(t_linspace, solution.t + offset)
    y_positions = np.append(y_positions, solution.y[0])

    velocity *= 0.9  # 10% straty prędkości
    offset += solution.t[-1]


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
