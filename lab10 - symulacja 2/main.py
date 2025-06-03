import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def pendulum_system(t, y):
    """
    System równań różniczkowych dla wahadła odwróconego
    y[0] = theta (kąt wychylenia)
    y[1] = omega (prędkość kątowa)
    """
    gamma = 0.3      # tłumienie
    g = 9.81
    L = 1.0          # długość wahadła
    A = 1.0          # amplituda siły wymuszającej
    omega_d = 1.0    # częstotliwość siły wymuszającej

    theta, omega = y

    dtheta_dt = omega
    domega_dt = -gamma * omega - (g/L) * np.sin(theta) + A * np.cos(omega_d * t)

    return [dtheta_dt, domega_dt]

def simulate_pendulum():
    """Główna funkcja symulacji"""

    # Warunki początkowe
    theta0 = 2.8      # Kąt początkowy
    omega0 = 0.0      # początkowa prędkość kątowa
    y0 = [theta0, omega0]

    # Parametry czasowe
    t_span = (0, 25)  # czas symulacji
    t_eval = np.linspace(0, 25, 2500)  # punkty do ewaluacji

    # Rozwiązanie układu równań
    solution = solve_ivp(pendulum_system, t_span, y0, t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10)

    return solution

def calculate_coordinates(theta, L=1.0):
    """Oblicza współrzędne kartezjańskie na podstawie kąta"""
    x = L * np.sin(theta)
    y = -L * np.cos(theta)  # ujemne, bo y=0 jest na górze
    return x, y

def plot_results(solution):
    """Tworzy wszystkie wykresy wymagane w zadaniu"""

    t = solution.t
    theta = solution.y[0]

    # Obliczenie współrzędnych kartezjańskich
    L = 1.0
    x, y = calculate_coordinates(theta, L)

    # Utworzenie siatki wykresów 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Symulacja wahadła odwróconego', fontsize=16)

    # a) Wykres kąta θ(t)
    ax1.plot(t, theta, 'b-', linewidth=1.5)
    ax1.set_xlabel('Czas [s]')
    ax1.set_ylabel('Kąt θ [rad]')
    ax1.set_title('a) Kąt wychylenia θ(t)')
    ax1.grid(True, alpha=0.3)

    # b) Wykres x(t)
    ax2.plot(t, x, 'r-', linewidth=1.5)
    ax2.set_xlabel('Czas [s]')
    ax2.set_ylabel('Pozycja x [m]')
    ax2.set_title('b) Pozycja pozioma x(t)')
    ax2.grid(True, alpha=0.3)

    # c) Wykres y(t)
    ax3.plot(t, y, 'g-', linewidth=1.5)
    ax3.set_xlabel('Czas [s]')
    ax3.set_ylabel('Pozycja y [m]')
    ax3.set_title('c) Pozycja pionowa y(t)')
    ax3.grid(True, alpha=0.3)

    # d) Wykres trajektorii y(x)
    ax4.plot(x, y, 'purple', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Pozycja x [m]')
    ax4.set_ylabel('Pozycja y [m]')
    ax4.set_title('d) Trajektoria y(x)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.axis('equal')

    plt.tight_layout()
    plt.show()

    # Dodatkowy wykres z informacjami o zakresach
    print(f"Zakres kąta θ: {np.min(theta):.2f} do {np.max(theta):.2f} rad")
    print(f"Zakres kąta θ: {np.min(theta)*180/np.pi:.1f}° do {np.max(theta)*180/np.pi:.1f}°")
    print(f"Zakres pozycji x: {np.min(x):.3f} do {np.max(x):.3f} m")
    print(f"Zakres pozycji y: {np.min(y):.3f} do {np.max(y):.3f} m")

def create_animation(solution):
    """Tworzy animację ruchu wahadła"""

    t = solution.t
    theta = solution.y[0]

    L = 1.0
    x, y = calculate_coordinates(theta, L)

    # Konfiguracja animacji
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Animacja wahadła odwróconego')

    # Elementy do animacji
    line, = ax.plot([], [], 'b-', linewidth=3, label='Wahadło')
    point, = ax.plot([], [], 'ro', markersize=10, label='Masa')
    trail, = ax.plot([], [], 'r--', alpha=0.5, linewidth=1, label='Trajektoria')

    ax.plot([0], [0], 'ko', markersize=8, label='Oś obrotu')

    # Dodanie okręgu pokazującego pełny zakres ruchu
    circle = plt.Circle((0, 0), L, fill=False, linestyle=':', alpha=0.3, color='gray')
    ax.add_patch(circle)

    ax.legend()

    # Przechowywanie trajektorii
    trail_x, trail_y = [], []

    def animate(frame):
        idx = min(frame, len(x) - 1)

        # Aktualizacja pozycji wahadła
        current_x = x[idx]
        current_y = y[idx]

        # Linia wahadła (od osi do masy)
        line.set_data([0, current_x], [0, current_y])

        # Pozycja masy
        point.set_data([current_x], [current_y])

        # Dodanie punktu do trajektorii
        trail_x.append(current_x)
        trail_y.append(current_y)

        # Ograniczenie długości trajektorii dla lepszej czytelności
        if len(trail_x) > 150:
            trail_x.pop(0)
            trail_y.pop(0)

        trail.set_data(trail_x, trail_y)

        return line, point, trail

    # Utworzenie animacji
    anim = FuncAnimation(fig, animate, frames=len(t), interval=5, blit=True, repeat=False)

    plt.show()
    return anim

def main():
    """Główna funkcja programu"""

    print("Symulacja wahadła odwróconego")
    print("====================================")

    # Wykonanie symulacji
    print("Wykonywanie symulacji...")
    solution = simulate_pendulum()

    if solution.success:
        print("Symulacja zakończona pomyślnie!")

        # Wyświetlenie podstawowych wykresów
        print("\nGenerowanie wykresów podstawowych...")
        plot_results(solution)

        # Utworzenie animacji
        print("\nTworzenie animacji...")
        anim = create_animation(solution)

    else:
        print("Błąd podczas symulacji:", solution.message)

if __name__ == "__main__":
    main()
