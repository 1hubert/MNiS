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
    x_start = 0
    x_end = 0

    x_linspace = np.linspace(x_start, x_end, 100)

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
    plt.plot(x_linspace, forward_history[0], 'b-', label='Różnica dzielona w przód, h=1')
    plt.plot(x_linspace, forward_history[1], 'g-', label='Różnica dzielona w przód, h=0.1')
    plt.plot(x_linspace, forward_history[2], 'c-', label='Różnica dzielona w przód, h=0.02')

    plt.plot(x_linspace, backward_history[0], 'b-', label='Różnica dzielona w tył, h=1')
    plt.plot(x_linspace, backward_history[1], 'g-', label='Różnica dzielona w tył, h=0.1')
    plt.plot(x_linspace, backward_history[2], 'c-', label='Różnica dzielona w tył, h=0.02')

    plt.plot(x_linspace, central_history[0], 'b-', label='Różnica dzielona centralna, h=1')
    plt.plot(x_linspace, central_history[1], 'g-', label='Różnica dzielona centralna, h=0.1')
    plt.plot(x_linspace, central_history[2], 'c-', label='Różnica dzielona centralna, h=0.02')

    # Obliczamy wartości oryginalnej funkcji dla porównania
    x_fine = np.linspace(x_start, x_end, 1000)

    y_original = [triangular_func_omega_0_25(x) for x in x_fine]

    # Obliczamy "dokładną" pochodną za pomocą bardzo małego h dla porównania
    y_derivative_exact = [roznica_dzielona_centralna(triangular_func_omega_0_25, x, 0.0001) for x in x_fine]

    # plt.plot(x_fine, y_derivative_exact, 'k-', label='Dokładna pochodna f\'(x)')
    # plt.plot(x_fine, y_original, 'k-', label='f(x)')


    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Porównanie metod różniczkowania')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Obliczamy maksymalną różnicę między metodami dla różnych h
    # for i, h in enumerate([1, 0.1, 0.02]):
    #     max_diff = np.max(np.abs(y_eul_forward_history[i][1] - y_eul_backward_history[i][1]))
    #     print(f"Maksymalna różnica między metodami dla h={h}: {max_diff:.6f}")

if __name__ == '__main__':
    main()
