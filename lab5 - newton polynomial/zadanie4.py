import numpy as np
import matplotlib.pyplot as plt


def divided_differences(x, y, n):
    """
    pl. różnice dzielone
    Compute the divided differences table for Newton interpolation.

    Args:
        x: List of x-coordinates (nodes).
        y: List of y-coordinates (function values at nodes).

    Returns:
        A list containing the coefficients for the Newton polynomial.
    """
    coef = y[:n+1].copy()  # Initial coefficients are the y-values

    for j in range(1, n+1):
        for i in range(n, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])

    return coef




def collect_and_validate_input():
    # zakładany rząd interpolowanej funkcji
    while True:
        try:
            polynomial_degree = int(input('Podaj rząd funkcji: '))
            if polynomial_degree > 0:  # valid
                break
            else:
                print('Rząd funkcji musi być większy od zera!')
        except ValueError:
            print('Podana wartość musi być liczbą całkowitą!')

    # ilość punktów interpolacji
    while True:
        try:
            point_count = int(input('Podaj ilość punktów: '))
            if point_count > 0:  # valid
                break
            else:
                print('Ilość punktów musi być większa od zera!')
        except ValueError:
            print('Podana wartość musi być liczbą całkowitą!')

    # zestaw punktów (x, y).
    x_points = []
    y_points = []
    for i in range(1, point_count + 1):
        while True:
            try:
                x, y = input(f'Podaj punkt {i}/{point_count}: ').split()
                x_points.append(float(x))
                y_points.append(float(y))
                break  # valid
            except ValueError:
                 print('Błąd: Wprowadź dwie liczby oddzielone spacją (np. "2.5 3.7")')

    return polynomial_degree, x_points, y_points, point_count


def example_input():
    point_count = 5
    x_points = [-2, 0, 1, 2.5, 4]
    y_points = [-14, 9, 4, 9.625, 175]
    return x_points, y_points, point_count


def main():
    def func(coefficients, x_points, R):
        val = 0
        for i in range(0, len(coefficients)):
            product = 1
            for j in range(0, i):
                product *= (R - x_points[j])

            val += coefficients[i] * product

        return val

    poly_data = {}
    x_points, y_points, point_count = example_input()
    for polynomial_degree in [2]:
        coefficients = divided_differences(x_points, y_points, polynomial_degree)
        y1 = lambda x: func(coefficients, x_points, x)

        # Oblicz błędy dla każdego punktu jako suma warości absolutnych różnicy
        errors = []
        for i in range(point_count):
            err = abs( (y1(x_points[i]) - y_points[i]) / y1(x_points[i]) ) * 100  # %
            errors.append(err)

        print(f'Suma błędów dla wielomianu stopnia {polynomial_degree}: {sum(errors):.2f}%')
        poly_data[polynomial_degree] = [coefficients, sum(errors)]

    min_err = 1_000_000_000_000
    for key, value in poly_data.items():
        if value[1] < min_err:
            min_err = value[1]
            print(value[0])
            coefficients = value[0]
            y1 = lambda x: func(value[0], x_points, x)
            x_linspace = np.linspace(
                x_points[0] - 0.15 * (x_points[-1] - x_points[0]),
                x_points[-1] + 0.15 * (x_points[-1] - x_points[0]),
                100
            )
            y = y1(x_linspace)


            polynomial_degree = key
    print(coefficients)
    # Plotting



    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_linspace, y, label=f"wielomian stopnia {polynomial_degree}", color="blue")
    plt.scatter(x_points, y_points, color="red", s=75, label="dane", zorder=5)  # Red points

    # Add labels and legend
    plt.title(f'Interpolacja wielomianowa', fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()
