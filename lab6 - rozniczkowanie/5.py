from math import pi, sin
import numpy as np
import matplotlib.pyplot as plt


def euler():
    """Rozwiązuje równanie różniczkowe y'(x)=y^2"""
    x_list = []
    y_euler_list = []
    y_exact_list = []

    h = 0.002
    x_i = 0
    y_i = 3
    y_exact = lambda x: -3 / (3*x - 1)

    print('-----------------------------------------------------')
    print(f'index\tx_i\ty_i\t\ty(x)\t\ty(x)-y_1')
    print(f'{0}\t{x_i}\t{y_i}\t\t{y_exact(x_i)}\t\t{y_exact(x_i)-y_i}')

    for i in range(1, 18):
        x_i = x_i + h
        y_i = y_i + h * y_i ** 2

        x_list.append(x_i)
        y_euler_list.append(y_i)
        y_exact_list.append(y_exact(x_i))

        print(f'{i}\t{x_i:.3f}\t{y_i:.6f}\t{y_exact(x_i):.6f}\t{(y_exact(x_i)-y_i):.6f}')
    print('-----------------------------------------------------')

    return x_list, y_euler_list, y_exact_list

def f(x, freq, max_k):
    omega = 2 * pi * freq
    result = (8 / (pi ** 2))
    sum1 = 0
    for k in range(0, max_k + 1):
        sum1 += ((-1)**k) * sin((2*k+1)*omega*x) / ((2*k+1)**2)

    return result * sum1

# porównaj dla h=[1, 0.1, 0.02]
# omega = 0.25

y1 = lambda x: f(x, 0.25, 500)
roznica_dzielona_w_przod = lambda x, h: (y1(x+h)-y1(x))/(h)


# Plotting
x_linspace = np.linspace(
    -5,
    5,
    100
)

y_pochodna = []
for x in x_linspace:
    y_pochodna.append(roznica_dzielona_w_przod(x, 0.02))

y_oryginal = []
for x in x_linspace:
    y_oryginal.append(y1(x))


# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_linspace, y_oryginal, label=f"f(x)", color="red")
plt.plot(x_linspace, y_pochodna, label=f"f'(x)", color="blue")

# Add labels and legend
plt.title(f'Różniczkowanie', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
