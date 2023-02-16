import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return -2*y

def sol_exata(t):
    return pow(np.e,-2*t)

def backward_euler_successive(f, y0, t, max_iterations=100, tolerance=1e-12):
    """
    Uses the backward Euler method with successive approximations to solve an ordinary differential equation of the form
    y' = f(t, y), where y0 is the initial value of y at time t[0].

    :param f: a function representing the right-hand side of the differential equation
    :param y0: the initial value of y at time t[0]
    :param t: an array of times at which to compute the solution
    :param max_iterations: the maximum number of iterations to use for the successive approximation
    :param tolerance: the convergence tolerance for the successive approximation
    :return: an array containing the values of y at each time in t
    """
    n = len(t)
    y = [y0] * n
    for i in range(1, n):
        dt = t[i] - t[i-1]
        y_guess = y[i-1] + dt * f(t[i], y[i-1])
        for j in range(max_iterations):
            y_next_guess = y[i-1] + dt * f(t[i], y_guess)
            if abs(y_next_guess - y_guess) < tolerance:
                break
            y_guess = y_next_guess
        y[i] = y_guess
    return y

def gerar_graf(t_n, y_1, y_e):
    plt.plot(t_n, y_1, ':', color='black', label = 'presas')
    plt.plot(t_n, y_e, color='black', label = 'predadores')
    plt.xlabel('tempo [meses]')
    plt.ylabel('Populacão das espécies [indivíduos]')
    plt.title('Modelo populacional de Lotka-Volterra')
    plt.legend()
    plt.show()

t_euler = np.linspace(0, 1, 33)
y_euler = backward_euler_successive(f,1,t_euler)
y_exact = [sol_exata(t) for t in t_euler]

gerar_graf(t_euler, y_euler, y_exact)