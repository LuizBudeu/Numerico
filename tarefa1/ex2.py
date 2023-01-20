import numpy as np
import math

# Differential equation
def f(y, t):
    return -2*y

# Initial condition
y0 = 1
T = 1
t0 = 0

n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
h = [(T-t0)/i for i in n]


solution = []
def tabela(n):
    
    for i in range(len(n)):
        p = 0
        
        # Time points
        t = np.arange(t0, T, h[i])
        y = np.zeros(len(t))
        y[0] = y0
        
        euler(y, t, h[i])
        
        if i > 0:
            q = abs((exact(T)-y[i-1])/(exact(T)-y[i]))
            r = h[i-1]/h[i]
            p = math.log(q)/math.log(r)
        
        i_solution = (n[i], h[i], error(y[-1], t[-1]), p)
        solution.append(i_solution)
    return solution

# Exact solution
def exact(t):
    return np.exp(-2*t)

# function to compute the error
def error(y, t):
    return abs(y - exact(t))

# Euler's method
def euler(y, t, h):
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(y[i-1], t[i-1])  
            
            


a = tabela(n)
with open('ex2.txt', 'w') as f:

    for i in range(len(a)):
        print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))
        
        f.write("%5d & %9.3e & %9.3e & %9.3e \\\\ \n" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))