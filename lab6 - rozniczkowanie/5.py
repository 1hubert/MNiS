import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import pi, sin

def triangular_func_template(x, freq, max_k):
    """Funkcja trójkątna dana wzorem"""
    omega = 2 * pi * freq
    result = (8 / (pi ** 2))
    sum1 = 0
    for k in range(0, max_k + 1):
        sum1 += ((-1)**k) * sin((2*k+1)*omega*x) / ((2*k+1)**2)
    return result * sum1

def triangular_func_omega_0_25(x):
    """Funkcja trójkątna dla częstotliwości 0.25"""
    return triangular_func_template(x, 0.25, 500)

def roznica_dzielona_w_przod(f, x, h):
    """Różnica dzielona w przód"""
    return (f(x+h) - f(x)) / h

def roznica_dzielona_w_tyl(f, x, h):
    """Różnica dzielona centralna"""
    return (f(x) - f(x-h)) / (h)

def roznica_dzielona_centralna(f, x, h):
    """Różnica dzielona centralna"""
    return (f(x+h) - f(x-h)) / (2*h)



def main():
    h_list = [1, 0.1, 0.02]

    x_linspace = np.linspace(0, 5, 100)

    forward_history = []
    backward_history = []
    central_history = []

    # Testujemy różne wartości h
    for i, h in enumerate(h_list):
        forward = []
        backward = []
        central = []

        for x in x_linspace:
            forward.append(roznica_dzielona_w_przod(triangular_func_omega_0_25, x, h))

        for x in x_linspace:
            backward.append(roznica_dzielona_w_tyl(triangular_func_omega_0_25, x, h))

        for x in x_linspace:
            central.append(roznica_dzielona_centralna(triangular_func_omega_0_25, x, h))

        forward_history.append(forward)
        backward_history.append(backward)
        central_history.append(central)


    # Wizualizacja wyników - raozwiązania równania różniczkowego
    plt.figure(figsize=(12, 8))

    # Rysujemy wyniki metody Eulera w przód
    plt.plot(x_linspace, forward_history[0],label='Różnica dzielona w przód, h=1')
    plt.plot(x_linspace, forward_history[1],label='Różnica dzielona w przód, h=0.1')
    plt.plot(x_linspace, forward_history[2],label='Różnica dzielona w przód, h=0.02')

    plt.plot(x_linspace, backward_history[0], label='Różnica dzielona w tył, h=1')
    plt.plot(x_linspace, backward_history[1], label='Różnica dzielona w tył, h=0.1')
    plt.plot(x_linspace, backward_history[2], label='Różnica dzielona w tył, h=0.02')

    plt.plot(x_linspace, central_history[0], label='Różnica dzielona centralna, h=1')
    plt.plot(x_linspace, central_history[1], label='Różnica dzielona centralna, h=0.1')
    plt.plot(x_linspace, central_history[2], label='Różnica dzielona centralna, h=0.02')


    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Porównanie metod różnic dzielonych')
    plt.grid(True)
    plt.legend()

    # Obliczamy maksymalną różnicę między metodami dla różnych h

    forward_history = [np.array(x) for x in forward_history]
    backward_history = [np.array(x) for x in backward_history]
    central_history = [np.array(x) for x in central_history]

    # print(f'forward_history 3 {forward_history[0][:3]}')
    # print(f'backward_history 3 {backward_history[0][:3]}')
    # print(f'abs {np.abs(forward_history[0][:3] - backward_history[0][:3])}')
    # print(f'max {np.max(np.abs(forward_history[0][:3] - backward_history[0][:3]))}')

    for i, h in enumerate([1, 0.1, 0.02]):
        max_diff = np.max(np.abs(forward_history[i] - backward_history[i]))
        print(f"Maksymalna różnica między metodami dla h={h}: {max_diff:.16f}")

    plt.show()


if __name__ == '__main__':
    main()
