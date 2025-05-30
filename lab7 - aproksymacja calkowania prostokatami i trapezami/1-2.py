from math import sin, cos, pi


def q_rect(a, b, n, func):
    area = 0
    h = (b-a) / n
    a_rect = a
    b_rect = a + h
    for _ in range(n):
        area += h * func((a_rect+b_rect)/2)

        a_rect += h
        b_rect += h
    return area


def q_trap(a, b, n, func):
    area = 0
    h = (b-a) / n
    a_rect = a
    b_rect = a + h
    for _ in range(n):
        area += h * ((func(a_rect)+func(b_rect))/2)

        a_rect += h
        b_rect += h
    return area


print(f'Całka funkcji sin metodą prostokątów: {q_rect(0, 2*pi, 1000, sin)}')
print(f'Całka funkcji cos metodą trapzów: {q_trap(0, 2*pi, 1000, cos)}')
