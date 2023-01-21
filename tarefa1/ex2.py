
import numpy as np
import math

# Differential equation
def f(y, t):
    return -2*y

# Initial condition
y0 = 1
T = 1
t0 = 0

n = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
h = [(T-t0)/i for i in n]

solution = []

def tabela(n):

    y_T = [] # Armazenara a aprox de y(T) para os n casos distintos
    
    for i in range(len(n)):
        p = 0
        
        # Time points
        t = np.linspace(t0, T, num=n[i]+1) # Para incluir T
        y = np.zeros(len(t))
        y[0] = y0

        euler(y, t, h[i])
        y_T.append(y[-1]) # Pega as estimativas de y(T) para cada n  
        
        if i > 0:
            q = abs((exact(T)-y_T[i-1])/(exact(T)-y_T[i]))
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


"""
import numpy as np
import math

# Differential equation
def f(y, t):
    return -2*y

# n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# h = [(T-t0)/i for i in n]



def main():
    # Initial condition
    y0 = 1
    T = 1
    t0 = 0
    
    m = 9
    h = [0]*m
    yn = [y0]*m
    
    for i in range(1, m+1):
        n=16*2**(i-1)
        p = 0
        
        h[i-1], yn[-1] = euler(t0, y0, T, n)
        
        if i > 1:
            q = abs((exact(T)-yn[i-2])/(exact(T)-yn[i-1]))
            r = h[i-2]/h[i-1]
            p = math.log(q)/math.log(r)
        
        with open('ex2.txt', 'w') as f:
            print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (n, h[-1], error(yn[-1], T), p))
            
            f.write("%5d & %9.3e & %9.3e & %9.3e \\\\ \n" % (n, h[-1], error(yn[-1], T), p))

# Exact solution
def exact(t):
    return np.exp(-2*t)

# function to compute the error
def error(y, t):
    return abs(y - exact(t))

# Euler's method
def euler(t0, y0, T, n):

    tn = [t0]
    yn = [np.array(y0)]

    h = (T-t0)/n
    
    while tn[-1] < T:
        yn.append(yn[-1] + h * f(yn[-1], tn[-1]))
        tn.append(tn[-1] + h)
        h = min(h, T - tn[-1]) 
            
    yn = np.array(yn)
    return (T-t0)/n, yn[-1]


if __name__ == "__main__":
    main()
    
"""