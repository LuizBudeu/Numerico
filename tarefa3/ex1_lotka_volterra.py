import math
import numpy as np
import matplotlib.pyplot as plt

# Initial condition
X0 = np.array([100, 50])
t0 = 0
T = 150

n = [pow(2, i) for i in range(4,16)]
h = [(T - t0)/i for i in n]

solution = []

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


# Euler's method
def euler_implicito(y, t, h):

    for i in range(1, len(t)):
        y[i] = succesive_aprox(y[i-1], t[i], h)
    return y


def succesive_aprox(y_ant, t, h, max_iter = 100, precisao = 1E-10):
   
    y_guess = y_ant

    for i in range(max_iter):
        y_next_guess = y_ant + h*f(y_guess, t)
        if abs(np.linalg.norm(y_next_guess) - np.linalg.norm(y_guess)) < precisao:
            break
        y_guess = y_next_guess

    return y_next_guess

def tabela(n):

    y_T = [] # Armazena a aprox de y(T) para os n casos distintos
    
    for i in range(len(n)):
        p = e = q = 0
        
        # Time points
        t = np.linspace(t0, T, num=n[i]+1) # Para incluir T
        y = np.zeros((len(t),2))
        y[0,:] = X0

        euler_implicito(y, t, h[i])
        y_T.append(y[-1]) # Pega as estimativas de y(T) para cada n  
        
        if i > 0:
            q0 = abs((y_T[i-2][0]-y_T[i-1][0])/(y_T[i-1][0]-y_T[i][0]))
            q1 = abs((y_T[i-2][1]-y_T[i-1][1])/(y_T[i-1][1]-y_T[i][1]))
            q = max(q0, q1)
            r = h[i-1]/h[i]
            p0 = math.log(q0)/math.log(r)
            p1 = math.log(q1)/math.log(r)
            p = max(p0, p1)
            
            e0 = abs((y_T[i-1][0]-y_T[i][0]))
            e1 = abs((y_T[i-1][1]-y_T[i][1]))
            e = math.sqrt(e0**2 + e1**2)
            
        i_solution = (n[i], h[i], e, q, p)
        solution.append(i_solution)
        
        if i == len(n) - 1:
            gerar_graf(t, y[:,0], y[:,1])

    return solution


def gerar_graf(t_n, y_1, y_2):
    # Presas
    plt.plot(t_n, y_1, ':', color='black', label = 'presas')
    plt.xlabel('tempo [meses]')
    plt.ylabel('Populacão de presas [indivíduos]')
    plt.title('Modelo populacional de Lotka-Volterra')
    plt.legend()
    plt.show()
    # Predadores
    plt.plot(t_n, y_2, '-.',color='black', label = 'predadores')
    plt.xlabel('tempo [meses]')
    plt.ylabel('Populacão de predadores [indivíduos]')
    plt.title('Modelo populacional de Lotka-Volterra')
    plt.legend()
    plt.show()


a = tabela(n)

with open('tarefa3/ex1_presapredador.txt', 'w') as f:

    for i in range(len(a)):
        print("%5d & %9.3e & %9.3e & %9.3e & %9.3e \\\\" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3], solution[i][4]))
        f.write("%5d & %9.3e & %9.3e & %9.3e & %9.3e \\\\ \n" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3], solution[i][4]))