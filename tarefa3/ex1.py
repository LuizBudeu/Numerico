
import numpy as np
import math

def sol_exata(t):
    return np.power(np.e, -2*t)

# Differential equation
def f(y, t):
    return -2*y

def f_prime(y, t):
    return 4*y

# Initial condition
y0 = [1]
T = 2
t0 = 0

n = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
h = [(T-t0)/i for i in n]
t = np.linspace(t0, T, num=n[1]+1)

# Exact solution
def exact(t):
    return np.exp(-2*t)


# function to compute the error
# def error(y, t):
#     return abs(y - exact(t))

# Euler's method
def euler_implicito(y: list, t: list, h):
    print(f'Método de Euler Implícito para y0={y}, t0={t[0]} e h={h}')

    for i in range(1, len(t)):
        y_new = succesive_aprox(y[-1], t[i-1:i+1], h)
        y.append(y_new)
        


def succesive_aprox(y_ant, t, h):

    MAX_ITER = 500    
    PRECISAO = 10E-12
    y_old = y_ant
    y_next = y_ant

    """
    phi_newton(x) = x - g(x)/g'(x), x = y_k+1

    g(x) -> g(y_k+1) = y_k+1 - y_k - h*f(t_k+1, y_k+1)
    g'(x) -> g'(y_k+1) = 1 - h*f'(t_k+1, y_k+1)
    """
    def g():
        return (y_old - y_ant - h*f(y_old, t[1]))
    def g_prime():
        return ((1 - h*f_prime(y_old, t[1])))

    for i in range(MAX_ITER):
        # Phi de Newton
        y_next = y_old - g()/g_prime()
        if abs(y_next - y_old) < PRECISAO:
            break
        
        y_old = y_next

    print(f'\t {y_next:10.5f} na iteração {i}, exato = {sol_exata(t[1]):10.5f}')
    return y_next


solution = []

euler_implicito(y0, t, h[1])
# def tabela(n):

    # y_T = [] # Armazena a aprox de y(T) para os n casos distintos
    
    # for i in range(2):
        # p = 0
        
        # # Time points
        # t = np.linspace(t0, T, num=n[i]+1) # Para incluir T
        # y = np.zeros(len(t))
        # y[0] = y0

        # euler_implicito(y, t, h[i])
        # y_T.append(y[-1]) # Pega as estimativas de y(T) para cada n  
        
        # if i > 0:
            # q = abs((exact(T)-y_T[i-1])/(exact(T)-y_T[i]))
            # r = h[i-1]/h[i]
            # p = math.log(q)/math.log(r)
        
        # i_solution = (n[i], h[i], p)
        # solution.append(i_solution)
    # return solution

# a = tabela(n)
# with open('tarefa1/ex2.1.txt', 'w') as f:
# 
#     for i in range(len(a)):
#         print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))
#         
#         f.write("%5d & %9.3e & %9.3e & %9.3e \\\\ \n" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))