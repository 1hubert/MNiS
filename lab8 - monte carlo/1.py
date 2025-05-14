from random import uniform
import numpy as np


def monte_carlo(a, b, n, func):
    """
    <a, b> - range
    n - number of random points
    func - a function to integrate
    """
    random_values = []
    for _ in range(n):
        random_values.append(func(uniform(a, b)))
    return sum(random_values) / n * (b-a)


def simpson(a, b, n, func):
    """
    <a, b> - range
    n - number of points
    func - a function to integrate
    """
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = func(x)

    # print(x[0])
    # print(x[1:-1:2])
    # print(x[2:-1:2])
    # print(x[-1])

    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])


def q_rect(a, b, n, func):
    area = 0
    h = (b-a) / n
    a_rect = a
    b_rect = a + h
    for _ in range(n):
        area += h * func((a_rect+b_rect) / 2)

        a_rect += h
        b_rect += h
    return area


def q_trap(a, b, n, func):
    area = 0
    h = (b-a) / n
    a_rect = a
    b_rect = a + h
    for _ in range(n):
        area += h * (func(a_rect)+func(b_rect)) / 2

        a_rect += h
        b_rect += h
    return area


x_l = 1
x_p = 10
n = 10
f = lambda x: 1 / (2 * (x ** 2)) + 2 * x

precise_solution = 99.45

methods = {
    "monte carlo": monte_carlo,
    "simpson": simpson,
    "m. kwadratów": q_rect,
    "m. trapezów": q_trap
}

for name, method in methods.items():
    result = method(x_l, x_p, n, f)
    error = abs(precise_solution - result)
    print(f'{name}:\t{result}\terror = {error}')

# W (a) wyliczyć asymptoty na kartce i ustalić w jakim przedziale zcałkowana funkcja będzie miała wartości
# W (b) wyliczyć na kartce całkę z f(x) i wyplotować wartości obliczone w sposób analityczny
# W (c) wyplotować wszystko: wartości analitycznie i 4 metodami
