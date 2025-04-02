import numpy as np
import matplotlib.pyplot as plt

def divided_differences(x, y):
    """
    Compute the divided differences table for Newton interpolation.

    Args:
        x: List of x-coordinates (nodes).
        y: List of y-coordinates (function values at nodes).

    Returns:
        A list containing the coefficients for the Newton polynomial.
    """
    n = len(x)
    coef = y.copy()  # Initial coefficients are the y-values

    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])

    return coef

def newton_interpolation(x_data, y_data, x_eval):
    """
    Evaluate the Newton interpolating polynomial at a given point x_eval.

    Args:
        x_data: List of x-coordinates (nodes).
        y_data: List of y-coordinates (function values at nodes).
        x_eval: The point at which to evaluate the polynomial.

    Returns:
        The interpolated value at x_eval.
    """
    coef = divided_differences(x_data, y_data)
    n = len(x_data)
    result = coef[-1]  # Start with the highest-order coefficient

    for i in range(n-2, -1, -1):
        result = result * (x_eval - x_data[i]) + coef[i]

    return result



def main():
    # Take in input data and validate

    # Danymi wejściowymi algorytmu są:
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
            except ValueError as e:
                print(e)


    # Compute and print the interpolated value
    interpolated_value = newton_interpolation(x_points, y_points, point_count)

    print(f"Interpolated value at x = {point_count}: {interpolated_value}")

    b_list = divided_differences(x_points, y_points)
    print(f'b0, b1, b2, ... = {b_list}')

    # Plotting
    x_linspace = np.linspace(
        x_points[0] - 0.1 * (x_points[-1] - x_points[0]),
        x_points[-1] + 0.1 * (x_points[-1] - x_points[0]),
        100
    )

    def func(b_list, x_points, R):
        val = b_list[0]

        for i in range(1, len(b_list)):
            wspolczynnik = 1
            for j in range(0, i):
                wspolczynnik *= (R - x_points[j])

            print(f'wspolczynnik {i} = {wspolczynnik}')
            val += b_list[i] * wspolczynnik

        return val


    y1 = lambda x: func(b_list, x_points, x)
    y = y1(x_linspace)
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_linspace, y, label="f", color="blue")
    plt.scatter(x_points, y_points, color="red", s=100, label="Points", zorder=5)  # Red points

    # Add labels and legend
    plt.title("Function Plot with Highlighted Points", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
