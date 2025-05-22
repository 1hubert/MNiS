# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
from scipy.integrate import solve_ivp
from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt

def system_of_equations(t, theta):
    omega = 2
    gamma = 0.5
    A = 1
    omega_d = 30
    g = -9.81
    L = 10

    theta1, theta2 = theta
    dtheta1_dt = omega
    dtheta2_dt = -gamma * omega - (g/L) * sin(theta2) + A * cos(omega_d*t)
    return dtheta1_dt, dtheta2_dt

def zero_crossing(t, y):
    """This function returns 0 when y1 crosses 0"""
    if t < 0.1:
        return 1.0  # Return non-zero value to avoid event detection
    return y[0]  # y1 to położenie

# Setting terminal = True tells solve_ivp to stop integration when the event occurs
zero_crossing.terminal = False

# Setting direction = 0 detects zero crossings in both directions
zero_crossing.direction = 0

# ilość kolejnych rzutów
n = 10

t_linspace = np.empty(0)
theta_positions = np.empty(0)
offset = 0
theta = 45
gamma = 0.5  # temp
# for i in range(n):
#     solution = solve_ivp(
#         system_of_equations,
#         [0, 5],  # zakres calkowania dla tej iteracji
#         [0, theta],  # polozenie i kat wychylenia
#         method='RK45',
#         events=zero_crossing,
#         t_eval=np.linspace(0,5,1000)
#     )

#     t_linspace = np.append(t_linspace, solution.t + offset)
#     theta_positions = np.append(theta_positions, solution.y[0])

#     theta *= gamma  # gamma% straty prędkości
#     offset += solution.t[-1]


solution = solve_ivp(
    system_of_equations,
    [0, 15],  # zakres calkowania dla tej iteracji
    [0, theta],  # polozenie i kat wychylenia
    method='RK45',
    events=zero_crossing,
    t_eval=np.linspace(0,15,1000)
)

t_linspace = np.append(t_linspace, solution.t)
theta_positions = np.append(theta_positions, solution.y[0])


print(t_linspace)
print(theta_positions)
# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(t_linspace, theta_positions, label=f"położenie piłki", color="blue")

# Add labels and legend
plt.title(f'Położenie w osi pionowej wahadła - wykres $\\theta(t)$, n={n}', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
