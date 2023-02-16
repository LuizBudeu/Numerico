import math
import numpy as np
import matplotlib.pyplot as plt

# Differential equation
def f(X, t):
    x, y = X
    a = 0.08
    b = 0.001
    c = 0.02
    d = 0.00006
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
    return np.array([dx, dy])

# Exact solution
def exact(t):
    return np.exp(t)*np.sin(t)

# function to compute the error
def error(y, t):
    return abs(y - exact(t))

# Euler's method
def euler_implicito(y, t, h):

    for i in range(1, len(t)):
        y[i] = succesive_aprox(y[i-1], t[i], h)
    return y


def succesive_aprox(y_ant, t, h, max_iter = 100, precisao = 1E-12):
   
    y_guess = y_ant

    for i in range(max_iter):
        # Phi de Newton
        y_next_guess = y_ant + h*f(y_guess, t)
        if abs(np.linalg.norm(y_next_guess) - np.linalg.norm(y_guess)) < precisao:
            break
        y_guess = y_next_guess

    return y_next_guess

def gerar_graf(t_n, y_1, y_2):
    plt.plot(t_n, y_1, ':', color='black', label = 'presas')
    plt.plot(t_n, y_2, color='black', label = 'predadores')
    plt.xlabel('tempo [meses]')
    plt.ylabel('Populacão das espécies [indivíduos]')
    plt.title('Modelo populacional de Lotka-Volterra')
    plt.legend()
    plt.show()

# Initial condition
X0 = np.array([200, 50])
t0 = 0
T = 300

selector = 1

n = 8192
h = (T-t0)/n
t = np.linspace(t0, T, num=n+1)
y_n = [X0] * (n+1)  # initial condition

euler_implicito(y_n, t, h)
yn = np.array(y_n)
gerar_graf(t, yn[:,0], yn[:,1])