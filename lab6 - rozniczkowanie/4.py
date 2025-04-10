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


x_list, y_euler_list, y_exact_list = euler()



# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_list, y_euler_list, label=f"rozwiązanie metodą Eulera", color="red")
plt.plot(x_list, y_exact_list, label=f"rozwiązanie rzeczywiste", color="blue")

# Add labels and legend
plt.title(f'Różniczkowanie', fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# Show the plot
plt.show()
