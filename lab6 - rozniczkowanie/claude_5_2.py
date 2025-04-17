import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import pi, sin

def f(x, freq, max_k):
    """Funkcja trójkątna dana wzorem"""
    omega = 2 * pi * freq
    result = (8 / (pi ** 2))
    sum1 = 0
    for k in range(0, max_k + 1):
        sum1 += ((-1)**k) * sin((2*k+1)*omega*x) / ((2*k+1)**2)
    return result * sum1

def roznica_dzielona_w_przod(func, x, h):
    """Różnica dzielona w przód"""
    return (func(x+h) - func(x)) / h

def roznica_dzielona_centralna(func, x, h):
    """Różnica dzielona centralna"""
    return (func(x+h) - func(x-h)) / (2*h)

def original_func(x):
    """Funkcja trójkątna dla częstotliwości 0.25"""
    return f(x, 0.25, 500)

def derivative_forward(x):
    """Przybliżenie pochodnej funkcji original_func za pomocą różnicy dzielonej w przód z małym h"""
    return roznica_dzielona_w_przod(original_func, x, 0.0001)

def solve_diff_eq_using_forward_euler(y0, x0, x_end, h):
    """
    Rozwiązuje równanie różniczkowe y' = f'(x) metodą Eulera w przód.

    Parametry:
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
    print(f'Forward Euler method with h = {h}')
    print(f'index\tx_i\ty_i')
    print(f'{0}\t{x[0]:.4f}\t{y[0]:.6f}')

    # Wykonujemy kroki metody Eulera w przód: y_{i+1} = y_i + h * f'(x_i)
    for i in range(1, n_steps):
        y[i] = y[i-1] + h * derivative_forward(x[i-1])
        if i % 10 == 0 or i == n_steps - 1:
            print(f'{i}\t{x[i]:.4f}\t{y[i]:.6f}')

    print('-----------------------------------------------------')

    return x, y

def backward_euler_step(y_n, x_n, x_next, h):
    """
    Wykonuje jeden krok metody Eulera wstecz.
    Równanie: y_{n+1} = y_n + h * f'(x_{n+1})

    Parametry:
    y_n - wartość funkcji w punkcie x_n
    x_n - obecny punkt x
    x_next - następny punkt x
    h - krok całkowania

    Zwraca:
    y_{n+1} - wartość funkcji w punkcie x_{n+1}
    """
    # Definiujemy równanie, które musimy rozwiązać: y_{n+1} - y_n - h*f'(x_{n+1}) = 0
    def equation(y_next):
        return y_next - y_n - h * derivative_forward(x_next)

    # Początkowe przybliżenie - używamy wyniku z metody Eulera w przód
    initial_guess = y_n + h * derivative_forward(x_n)

    # Rozwiązujemy równanie
    y_next = fsolve(equation, initial_guess)[0]

    return y_next

def solve_diff_eq_using_backward_euler(y0, x0, x_end, h):
    """
    Rozwiązuje równanie różniczkowe y' = f'(x) metodą Eulera wstecz.

    Parametry:
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
    print(f'Backward Euler method with h = {h}')
    print(f'index\tx_i\ty_i')
    print(f'{0}\t{x[0]:.4f}\t{y[0]:.6f}')

    # Wykonujemy kroki metody Eulera wstecz: y_{i+1} = y_i + h * f'(x_{i+1})
    for i in range(1, n_steps):
        y[i] = backward_euler_step(y[i-1], x[i-1], x[i], h)
        if i % 10 == 0 or i == n_steps - 1:
            print(f'{i}\t{x[i]:.4f}\t{y[i]:.6f}')

    print('-----------------------------------------------------')

    return x, y

def main():
    # Parametry
    x0 = 0      # x początkowe
    x_end = 6   # x końcowe
    y0 = 0      # wartość początkowa całkowanej funkcji

    # Historia wyników dla różnych h
    y_eul_forward_history = []
    y_eul_backward_history = []

    # Testujemy różne wartości h
    for h in [1, 0.1, 0.02]:
        x_linspace, y_eul_forward = solve_diff_eq_using_forward_euler(y0, x0, x_end, h)
        x_linspace, y_eul_backward = solve_diff_eq_using_backward_euler(y0, x0, x_end, h)

        y_eul_forward_history.append([x_linspace, y_eul_forward])
        y_eul_backward_history.append([x_linspace, y_eul_backward])

    # Obliczamy wartości oryginalnej funkcji dla porównania
    x_fine = np.linspace(x0, x_end, 1000)
    y_original = [original_func(x) for x in x_fine]

    # Obliczamy "dokładną" pochodną za pomocą bardzo małego h dla porównania
    y_derivative_exact = [derivative_forward(x) for x in x_fine]

    # Wizualizacja wyników - rozwiązania równania różniczkowego
    plt.figure(figsize=(12, 8))

    # Rysujemy wyniki metody Eulera w przód
    plt.plot(y_eul_forward_history[0][0], y_eul_forward_history[0][1], 'b-', label='Forward Euler, h=1')
    plt.plot(y_eul_forward_history[1][0], y_eul_forward_history[1][1], 'g-', label='Forward Euler, h=0.1')
    plt.plot(y_eul_forward_history[2][0], y_eul_forward_history[2][1], 'c-', label='Forward Euler, h=0.02')

    # Rysujemy wyniki metody Eulera w tył
    plt.plot(y_eul_backward_history[0][0], y_eul_backward_history[0][1], 'r-', label='Backward Euler, h=1')
    plt.plot(y_eul_backward_history[1][0], y_eul_backward_history[1][1], 'm-', label='Backward Euler, h=0.1')
    plt.plot(y_eul_backward_history[2][0], y_eul_backward_history[2][1], 'y-', label='Backward Euler, h=0.02')

    # Rysujemy oryginalną funkcję
    plt.plot(x_fine, y_original, 'k--', label='Oryginalna funkcja f(x)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Porównanie metody Eulera w przód z metodą Eulera w tył')
    plt.grid(True)
    plt.legend()

    # Obliczamy maksymalną różnicę między metodami dla różnych h
    for i, h in enumerate([1, 0.1, 0.02]):
        max_diff = np.max(np.abs(y_eul_forward_history[i][1] - y_eul_backward_history[i][1]))
        print(f"Maksymalna różnica między metodami dla h={h}: {max_diff:.6f}")

    plt.figure(figsize=(12, 8))
    plt.plot(x_fine, y_derivative_exact, 'k-', label='Dokładna pochodna f\'(x)')
    plt.xlabel('x')
    plt.ylabel('f\'(x)')
    plt.title('Pochodna funkcji trójkątnej')
    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
