import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
def model(t, y):
    y1, y2 = y
    dy1dt = y2
    dy2dt = -9.81
    return dy1dt, dy2dt
intialvalue = 0



def zero_crossing(t, y):
    """This function returns 0 when y = 0"""
    if t < 0.1:
        return 1.0  # Return non-zero value to avoid event detection
    return y[0]

# Setting terminal = True tells solve_ivp to stop integration when the event occurs
zero_crossing.terminal = True

# Setting direction = 0 detects zero crossings in both directions
zero_crossing.direction = 0



solution= solve_ivp(
    model,
    [0, 5],
    [0, 10],
    method='RK45',
    events=zero_crossing,
    t_eval=np.linspace(0,5,1000)
)

print(solution)
plt.plot(solution.t, solution.y[0])
plt.show()
