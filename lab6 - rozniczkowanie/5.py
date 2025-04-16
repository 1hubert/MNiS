import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import pi, sin

roznica_dzielona_w_przod = lambda func, x, h: (func(x+h)-func(x))/(h)

def f(x, freq, max_k):
    omega = 2 * pi * freq
    result = (8 / (pi ** 2))
    sum1 = 0
    for k in range(0, max_k + 1):
        sum1 += ((-1)**k) * sin((2*k+1)*omega*x) / ((2*k+1)**2)

    return result * sum1

original_func = lambda x: f(x, 0.25, 500)

def backward_euler_step(y_n, h):
    """
    Wykonuje jeden krok metody Eulera wstecz.
    y_{n+1} = y_n + h*f(y_{n+1})

    Parametry:
    y_n - wartość funkcji w punkcie x_n
    h - krok całkowania
    f - prawa strona równania różniczkowego y' = f(y)

    Zwraca:
    y_{n+1} - wartość funkcji w punkcie x_{n+1}
    """
    # Definiujemy równanie, które musimy rozwiązać: y_{n+1} - y_n - h*f(y_{n+1}) = 0
    def equation(y_next):
        return y_next - y_n - h * roznica_dzielona_w_przod(original_func, y_next, h)

    # Początkowe przybliżenie - używamy wyniku z metody Eulera w przód
    initial_guess = y_n + h * roznica_dzielona_w_przod(original_func, y_n, h)

    # Rozwiązujemy równanie nieliniowe
    y_next = fsolve(equation, initial_guess)[0]

    return y_next

def solve_diff_eq_using_backward_euler(y0, x0, x_end, h):
    """
    Rozwiązuje równanie różniczkowe y' = f(y) metodą Eulera wstecz.

    Parametry:
    f - funkcja definiująca prawą stronę równania różniczkowego
    y0 - wartość początkowa y(x0)
    x0 - początkowy punkt x
    x_end - końcowy punkt x
    h - krok całkowania

    Zwraca:
    x - tablica punktów x
    y - tablica wartości y w punktach x
    """
    # Obliczamy liczbę kroków
    n_steps = int((x_end - x0) / h) + 1

    # Inicjalizujemy tablice wyników
    x = np.linspace(x0, x_end, n_steps)
    y = np.zeros(n_steps)

    # Ustawiamy wartość początkową
    y[0] = y0

    print('-----------------------------------------------------')
    print(f'index\tx_i\ty_i')
    print(f'{0}\t{x[0]}\t{y[0]}')

    # Wykonujemy kroki metody
    for i in range(1, n_steps):
        y[i] = backward_euler_step(y[i-1], h)

        # print(f'{i}\t{x[i]:.3f}\t{y[i]:.6f}')
    print('-----------------------------------------------------')

    return x, y

def solve_diff_eq_using_forward_euler(y0, x0, x_end, h):
    # Obliczamy liczbę kroków
    n_steps = int((x_end - x0) / h) + 1

    # Inicjalizujemy tablice wyników
    x = np.linspace(x0, x_end, n_steps)
    y = np.zeros(n_steps)

    # Ustawiamy wartość początkową
    y[0] = y0

    print('-----------------------------------------------------')
    print(f'index\tx_i\ty_i\t\ty(x)\t\ty(x)-y_1')
    print(f'{0}\t{x[0]}\t{y[0]}')

    for i in range(1, n_steps):
        y[i] = y[i-1] + h * roznica_dzielona_w_przod(original_func, y[i-1], h)

        # print(f'{i}\t{x[i]:.3f}\t{y[i]:.6f}')
    print('-----------------------------------------------------')

    return x, y


def main():


    # 1) zróżniczkowanie funkcji z zadania 1 metodą różniczkowania z zadania 2


    # 2) zcałkowanie obydwoma eulerami uzyskanej pochodnej używając trzech różnych h
    x_end = 3  # x end
    x0 = 0  # x start
    y0 = 0  # war. pocz.
    y_eul_forward_history = []
    y_eul_backward_history = []
    for h in [1, 0.1, 0.02]:
        x_linspace, y_eul_forward = solve_diff_eq_using_forward_euler(y0, x0, x_end, h)
        x_linspace, y_eul_backward = solve_diff_eq_using_backward_euler(y0, x0, x_end, h)

        y_eul_forward_history.append([x_linspace, y_eul_forward])
        y_eul_backward_history.append([x_linspace, y_eul_backward])

    # na potrzeby wykresu oryginalnej, trójkątnej funkcji
    y_original = []
    for x in y_eul_backward_history[2][0]:
        y_original.append(original_func(x))

    # Wizualizacja wyników
    plt.figure(figsize=(10, 6))
    plt.plot(y_eul_forward_history[0][0], y_eul_forward_history[0][1], label='forward, h=1')
    plt.plot(y_eul_forward_history[1][0], y_eul_forward_history[1][1], label='forward, h=0.1')
    plt.plot(y_eul_forward_history[2][0], y_eul_forward_history[2][1], label='forward, h=0.02')

    plt.plot(y_eul_backward_history[0][0], y_eul_backward_history[0][1], label='backward, h=1')
    plt.plot(y_eul_backward_history[1][0], y_eul_backward_history[1][1], label='backward, h=0.1')
    plt.plot(y_eul_backward_history[2][0], y_eul_backward_history[2][1], label='backward, h=0.02')

    plt.plot(y_eul_backward_history[2][0], y_original, label='f(x)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Porównanie m. Eulera w przód z m. Eulera w tył')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3) porównanie wpływu różnych h na wynik
    # 4) porównanie obu metod


if __name__ == '__main__':
    main()
