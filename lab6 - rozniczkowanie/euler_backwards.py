import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve



def backward_euler_step(y_n, h, f):
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
        return y_next - y_n - h * f(y_next)

    # Początkowe przybliżenie - używamy wyniku z metody Eulera w przód
    initial_guess = y_n + h * f(y_n)

    # Rozwiązujemy równanie nieliniowe
    y_next = fsolve(equation, initial_guess)[0]

    return y_next

def solve_diff_eq_using_backward_euler(f, y0, x0, x_end, h):
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

    # Wykonujemy kroki metody
    for i in range(1, n_steps):
        y[i] = backward_euler_step(y[i-1], h, f)

    return x, y

y_exact_solution = lambda x: -3 / (3*x - 1)

x0 = 0      # Punkt początkowy
y0 = 3      # Wartość początkowa y(0) = 1
x_end = 0.9 # Punkt końcowy (uwaga: dla y0=1 rozwiązanie dokładne ma osobliwość w x=1)
h = 0.002    # Krok całkowania


# Rozwiązujemy równanie metodą Eulera wstecz
x_numerical, y_numerical = solve_diff_eq_using_backward_euler(
    lambda y: y ** 2,  # Funkcja definiująca równanie różniczkowe y'(x) = y²
    y0,
    x0,
    x_end,
    h
)

# Obliczamy rozwiązanie dokładne
x_exact = np.linspace(x0, x_end, 1000)
y_exact = y_exact_solution(x_exact)

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.plot(x_exact, y_exact, 'b-', label='Analityczne')
plt.plot(x_numerical, y_numerical, 'ro', label='m. Eulera wstecz')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rozwiązanie równania y\' = y² metodą Eulera wstecz')
plt.grid(True)
plt.legend()
plt.show()

# Wypisujemy wartości numeryczne
print("Wyniki metody Eulera wstecz:")
for i in range(len(x_numerical)):
    if i % 4 == 0 or i == len(x_numerical)-1:  # Wypisujemy co 4-ty punkt dla przejrzystości
        y_exact_val = y_exact_solution(x_numerical[i], y0, x0)
        error = abs(y_numerical[i] - y_exact_val)
        print(f"x = {x_numerical[i]:.4f}, y_num = {y_numerical[i]:.6f}, y_exact = {y_exact_val:.6f}, błąd = {error:.6e}")
