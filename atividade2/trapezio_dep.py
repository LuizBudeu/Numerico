import math
import numpy as np
import matplotlib.pyplot as plt

# Initial condition
X0 = np.array([0.3, 0.5, 0.2])
t0 = 0
T = 2 

solution = []

n = [pow(2,i) for i in range(4,16)]
h = [(T-t0)/i for i in n]



# Differential equation
def f(X, t):
    y1, y2, y3 = X
    return np.array([
        2*y1 + y2 + 3*y3,
        y2 + 2*y3,
        3*y3
    ])

# Exact solution
def exact(t):
    return np.array([
        -0.3*np.e**t + 0.5*np.e**(2*t) + 0.8*np.e**(3*t),
        0.3*np.e**t + 0.2*np.e**(3*t),
        0.2*np.e**(3*t)
    ])

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

    for _ in range(max_iter):
        # Phi de Newton
        y_next_guess = y_ant + h*(f(y_guess, t) + f(y_ant, t-h))/2
        if abs(np.linalg.norm(y_next_guess) - np.linalg.norm(y_guess)) < precisao:
            break
        y_guess = y_next_guess

    return y_next_guess

def tabela(n):

    y_T = [] # Armazena a aprox de y(T) para os n casos distintos
    
    for i in range(len(n)):
        p = e = 0
        
        # Time points
        t = np.linspace(t0, T, num=n[i]+1) # Para incluir T
        y = np.zeros((len(t),3))
        y[0,:] = X0

        euler_implicito(y, t, h[i])
        y_T.append(y[-1]) # Pega as estimativas de y(T) para cada n  
        
        if i > 0:
            q0 = abs((exact(T)[0]-y_T[i-1][0])/(exact(T)[0]-y_T[i][0]))
            q1 = abs((exact(T)[1]-y_T[i-1][1])/(exact(T)[1]-y_T[i][1]))
            q2 = abs((exact(T)[2]-y_T[i-1][2])/(exact(T)[2]-y_T[i][2]))
            r = h[i-1]/h[i]
            p0 = math.log(q0)/math.log(r)
            p1 = math.log(q1)/math.log(r)
            p2 = math.log(q2)/math.log(r)
            p = max(p0, p1, p2)
            
            e0 = abs((exact(T)[0]-y_T[i][0]))
            e1 = abs((exact(T)[1]-y_T[i][1]))
            e2 = abs((exact(T)[2]-y_T[i][2]))
            e = math.sqrt(e0**2 + e1**2 + e2**2)
            
        i_solution = (n[i], h[i], e, p)
        solution.append(i_solution)
        
        if i == len(n) - 1:
            print(p0,p1,p2)
            print(e0,e1,e2)
            gerar_graf(t, y[:,0], y[:,1], y[:,2])

    return solution

def gerar_graf(t_n, y_1, y_2, y_3):
    plt.plot(t_n, y_1, ':', color='black', label = r'$y_1(t) = x(t)$')
    plt.xlabel('tempo [s]')
    plt.ylabel(r'$Valores\ \ \ y(t) = (y_1(t), y_2(t))$')
    plt.title(r'Depuração por Solução Manufaturada:  $\frac{d^2}{dx^2}x(t) = -x(t)$' ,)
    plt.legend()
    plt.show()
    plt.plot(t_n, y_2, '-.',color='black', label = r'$y_2(t) = \frac{d}{dx}x(t)$')
    plt.xlabel('tempo [s]')
    plt.ylabel(r'$Valores\ \ \ y(t) = (y_1(t), y_2(t))$')
    plt.title(r'Depuração por Solução Manufaturada:  $\frac{d^2}{dx^2}x(t) = -x(t)$' ,)
    plt.legend()
    plt.show()
    plt.plot(t_n, y_3, '.',color='black', label = r'$y_2(t) = \frac{d}{dx}x(t)$')
    plt.xlabel('tempo [s]')
    plt.ylabel(r'$Valores\ \ \ y(t) = (y_1(t), y_2(t))$')
    plt.title(r'Depuração por Solução Manufaturada:  $\frac{d^2}{dx^2}x(t) = -x(t)$' ,)
    plt.legend()
    plt.show()

a = tabela(n)

with open('tarefa3/ex1_depurado.txt', 'w') as f:

    for i in range(len(a)):
        print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))
        f.write("%5d & %9.3e & %9.3e & %9.3e \\\\ \n" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))