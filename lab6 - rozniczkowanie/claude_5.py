def euler_derivative(func, x_start=0, steps=18, h=0.002):
    """
    Calculates derivative using Euler's method by solving y'(x) = f(x)

    Parameters:
    - func: The function f(x) whose derivative we want to find
    - x_start: Starting x value (default 0)
    - steps: Number of steps (default 18)
    - h: Step size (default 0.002)

    Returns:
    - x_list: List of x values
    - derivative_list: List of y' values (derivatives)
    """
    x_list = []
    derivative_list = []

    # Initial values
    x_i = x_start
    y_i = 0  # y(0) = 0 is a reasonable starting point for computing derivatives

    print('-----------------------------------------------------')
    print(f'index\tx_i\ty_i\t\tf(x_i)\t\tderivative')
    print(f'{0}\t{x_i}\t{y_i}\t\t{func(x_i)}\t\t{func(x_i)}')

    for i in range(1, steps + 1):
        # Calculate derivative at current point
        derivative = func(x_i)
        derivative_list.append(derivative)

        # Euler's method: y_{i+1} = y_i + h * y'(x_i)
        # Where y'(x_i) = f(x_i) in our case
        y_i = y_i + h * derivative
        x_i = x_i + h

        x_list.append(x_i)

        print(f'{i}\t{x_i:.3f}\t{y_i:.6f}\t{func(x_i):.6f}\t{func(x_i):.6f}')

    print('-----------------------------------------------------')

    return x_list, derivative_list

# Example usage
if __name__ == "__main__":
    # Define function to differentiate
    def function_to_differentiate(x):
        return x**2  # Example: f(x) = x^2

    # Calculate derivatives using Euler's method
    x_values, derivatives = euler_derivative(function_to_differentiate)

    # Compare with exact derivatives
    print("\nComparison with exact derivatives:")
    print('-----------------------------------------------------')
    print(f'x\tnumerical\texact\t\terror')

    for i in range(len(x_values)):
        x = x_values[i]
        exact = 2 * x  # Exact derivative of x^2 is 2x
        error = derivatives[i] - exact
        print(f'{x:.3f}\t{derivatives[i]:.6f}\t{exact:.6f}\t{error:.6f}')

    print('-----------------------------------------------------')
