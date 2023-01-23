import matplotlib.pyplot as plt
import numpy as np
import math

# Differential equations
def f(X, t):
    x, y = X
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
    return np.array([dx, dy])

# Initial conditions
X0 = np.array([100, 50])
t0 = 0
T = 20

# Constants
a = 0.08
b = 0.001
c = 0.02
d = 0.00002


n = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
h = [(T-t0)/i for i in n]

solution = []

def tabela(n):
    # Array to store the solution
    y_T = []
    
    for i in range(len(n)):
        p = e = 0
        
        t = np.linspace(t0, T, num=n[i]+1)
        X = np.zeros((len(t), 2))
        X[0,:] = X0
        
        euler(X, t, h[i])
        y_T.append(X[-1]) 
        
        if i > 0:
            q0 = abs((y_T[i-2][0]-y_T[i-1][0])/(y_T[i-1][0]-y_T[i][0]))
            q1 = abs((y_T[i-2][1]-y_T[i-1][1])/(y_T[i-1][1]-y_T[i][1]))
            r = h[i-1]/h[i]
            p0 = math.log(q0)/math.log(r)
            p1 = math.log(q1)/math.log(r)
            p = max(p0, p1)
            
            e0 = abs((y_T[i-1][0]-y_T[i][0]))
            e1 = abs((y_T[i-1][1]-y_T[i][1]))
            e = math.sqrt(e0**2 + e1**2)
            
        i_solution = (n[i], h[i], e, p)
        solution.append(i_solution)
    gerar_graf(t, X)
    return solution

# Euler's method
def euler(X, t, h):
    for i in range(1, len(t)):
        X[i,:] = X[i-1,:] + h * f(X[i-1,:], t[i-1])

def gerar_graf(t_n, y_n):
    plt.plot(t_n, y_n[:,0], ':', color='black', label = 'presas')
    plt.plot(t_n, y_n[:,1], '-.', color='black', label = 'predadores')
    plt.xlabel('tempo [meses]')
    plt.ylabel('Populacão das espécies [indivíduos]')
    plt.title('Modelo populacional de Lotka-Volterra')
    plt.legend()
    plt.show()

a = tabela(n)
with open('tarefa1/ex2.3.txt', 'w') as f:

    for i in range(len(a)):
        print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))
        
        f.write("%5d & %9.3e & %9.3e & %9.3e \\\\ \n" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))