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
    # a = x_0
    # b = x_n
    # x_i = (x_0 + x_n) /2

    if n < 1000:
        h = (b - a) / (n)
    elif n >= 1000:
        h = (b - a) / (n-1)

    sum1 = 0
    sum2 = 0

    x_values = np.linspace(a, b, n)

    przedzial1 = range(1, n-1, 2)
    # print(list(przedzial1))
    for i in przedzial1:  # <1, n-1> czyli nieparzyste indeksy
        sum1 += func(x_values[i])

    przedzial2 = range(2, n-1, 2)
    # print(list(przedzial2))
    for i in przedzial2:  # <2, n-2> czyli parzyste indeksy
        sum2 += func(x_values[i])

    # print(f'sum1: {sum1}')
    # print(f'sum2: {sum2}')

    # return (h / 3) * (func(a) + 4*func((b + a) / 2) + func(b))
    return (h / 3) * (func(a) + 4*sum1 + 2*sum2 + func(b))


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
n = 100_000
f = lambda x: 1 / (2 * x ** 2) + 2 * x

print(f'monte carlo: {monte_carlo(x_l, x_p, n, f)}')
print(f'simpson: {simpson(x_l, x_p, n, f)}')
print(f'metoda kwadratów: {q_rect(x_l, x_p, n, f)}')
print(f'metoda trapezów: {q_trap(x_l, x_p, n, f)}')

# W (a) wyliczyć asymptoty na kartce i ustalić w jakim przedziale zcałkowana funkcja będzie miała wartości
# W (b) wyliczyć na kartce całkę z f(x) i wyplotować wartości obliczone w sposób analityczny
# W (c) wyplotować wszystko: wartości analitycznie i 4 metodami
