from math import e

def q_rect(a, b, n, func):
    area = 0
    h = (b-a) / n
    a_rect = a
    b_rect = a + h
    for i in range(n):
        area += (b_rect-a_rect) * func((a_rect+b_rect)/2)

        a_rect += h
        b_rect += h
    return area

def q_trap(a, b, n, func):
    pass

f = lambda x: 1/x

print(q_rect(1,e, 100, f))
