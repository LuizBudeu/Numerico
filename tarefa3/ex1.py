import math
import numpy as np
import matplotlib.pyplot as plt

# Differential equation
def f(y, t):
    return y + np.exp(t)*np.cos(t)

# Exact solution
def exact(t):
    return np.exp(t)*np.sin(t)

# function to compute the error
def error(y, t):
    return abs(y - exact(t))

# Euler's method
def euler_implicito(y, t, h):
    
    y_aprox = y
    for i in range(1, len(t)):
        y[i] = succesive_aprox(y[i-1], t[i], h)
    return y


def succesive_aprox(y_ant, t, h, max_iter = 100, precisao = 1E-12):
   
    y_guess = y_ant

    for i in range(max_iter):
        # Phi de Newton
        y_next_guess = y_ant + h*f(y_guess, t)
        if abs(y_next_guess - y_guess) < precisao:
            break
        y_guess = y_next_guess

    return y_next_guess

def error(y, t):
    return abs(y - exact(t))


def tabela(n):

    y_T = [] # Armazena a aprox de y(T) para os n casos distintos
    
    for i in range(len(n)):
        p = 0
        
        # Time points
        t = np.linspace(t0, T, num=n[i]+1) # Para incluir T
        y = np.zeros(len(t))
        y[0] = y0

        euler_implicito(y, t, h[i])
        y_T.append(y[-1]) # Pega as estimativas de y(T) para cada n  
        
        if i > 0:
            q = abs((exact(T)-y_T[i-1])/(exact(T)-y_T[i]))
            r = h[i-1]/h[i]
            p = math.log(q)/math.log(r)
            
        i_solution = (n[i], h[i], error(y[-1], t[-1]), p)
        solution.append(i_solution)
        # gerar_graf(t, y, [exact(ti) for ti in t])
    return solution

def gerar_graf(t_n, y_1, y_e):
    plt.plot(t_n, y_1, ':', color='black', label = 'presas')
    plt.plot(t_n, y_e, color='black', label = 'predadores')
    plt.xlabel('tempo [meses]')
    plt.ylabel('Populacão das espécies [indivíduos]')
    plt.title('Modelo populacional de Lotka-Volterra')
    plt.legend()
    plt.show()

# Initial condition
y0 = 0
T = 2
t0 = -np.pi

selector = 1

n = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
h = [(T-t0)/i for i in n]


t_euler = np.linspace(t0, T, num=n[selector]+1)
y_euler = [y0] * len(t_euler)

solution = []
# euler_implicito(y_euler, t_euler, h[selector])



a = tabela(n)
print(a)
with open('tarefa1/ex2.1.txt', 'w') as f:

    for i in range(len(a)):
        print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))        
        f.write("%5d & %9.3e & %9.3e & %9.3e \\\\ \n" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))

# gerar_graf(t_euler, y_euler, solution)