import numpy as np
import matplotlib.pyplot as plt
import math
from constants import *


def SIR_model(Y, t):
    S, I, R = Y
    dotS = -Lambda * S * I / N
    dotI = Lambda * S * I / N - gamma * I
    dotR = gamma * I
    return np.array([dotS, dotI, dotR])


# Método dos Trapézios
def trapezios(Y0, t, h):
    nt = len(t)
    Y  = np.zeros([nt, len(Y0)])
    Y[0] = Y0
    for i in range(1, len(t)):
        Y[i] = succesive_aprox(Y[i-1], t[i], h)
    return Y


def succesive_aprox(y_ant, t, h, max_iter = 100, precisao = 1E-12):
   
    y_guess = y_ant

    for _ in range(max_iter):
        # Phi de Newton
        y_next_guess = y_ant + h*(SIR_model(y_guess, t) + SIR_model(y_ant, t-h))/2
        if abs(np.linalg.norm(y_next_guess) - np.linalg.norm(y_guess)) < precisao:
            break
        y_guess = y_next_guess

    return y_next_guess

solution= []

def tabela(n):

    y_T = [] # Armazena a aprox de y(T) para os n casos distintos
    
    for i in range(len(n)):        
        # Time points
        t = np.linspace(t0, T, num=n[i]+1) # Para incluir T

        y = trapezios(Y0, t, h[i])
        y_T.append(y[-1])
        
        p = e = 0
        if i > 1:
            q0 = abs((y_T[i-2][0] - y_T[i-1][0]) / (y_T[i-1][0] - y_T[i][0]))
            q1 = abs((y_T[i-2][1] - y_T[i-1][1]) / (y_T[i-1][1] - y_T[i][1]))
            q2 = abs((y_T[i-2][2] - y_T[i-1][2]) / (y_T[i-1][2] - y_T[i][2]))
            r = h[i-1]/h[i]
      
            p0 = math.log(q0)/math.log(r)
            p1 = math.log(q1)/math.log(r)
            p2 = math.log(q2)/math.log(r)
            p = min(p0, p1, p2)
      
            e0 = abs((y_T[i-1][0]-y_T[i][0]))
            e1 = abs((y_T[i-1][1]-y_T[i][1]))
            e2 = abs((y_T[i-1][2]-y_T[i][2]))
            e = math.sqrt(e0**2 + e1**2 + e2**2)

        i_solution = (n[i], h[i], e, p)
        solution.append(i_solution)

        if i == len(n) - 1:
            print(e0,e1,e2)
            gerar_grafico_indiv(t, y)
            gerar_grafico_conj(t, y)



def gerar_grafico_indiv(t_n, y_n):
    state_var = ['Suscetiveis', 'Infectados', 'Recuperados']
    for index, var in enumerate(state_var):
        plt.title(f"Modelo SIR - {var} - Método do Trapézio")
        plt.plot(t_n, y_n[:,index], '--', color="black", label=f"{var}")
        plt.grid()
        plt.xlabel("Tempo, $t$ [dias]")
        plt.ylim(-0.02E7,1.02E7)
        plt.ylabel("População")
        plt.legend(loc = "best")
        plt.show()

def gerar_grafico_conj(t_n, y_n):
    plt.title(f"Modelo SIR - Método do Trapézio")
    plt.plot(t_n, y_n[:,0], ':', color="black", label='Suscetiveis')
    plt.plot(t_n, y_n[:,1], '-.', color="black", label='Infectados')
    plt.plot(t_n, y_n[:,2], '--', color="black", label='Recuperados')
    plt.grid()
    plt.xlabel("Tempo, $t$ [dias]")
    plt.ylim(-0.02E7,1.02E7)
    plt.ylabel("População")
    plt.legend(loc = "best")
    plt.show()

a = tabela(n)

with open('atividade2/sir.txt', 'w') as f:

    for i in range(len(solution)):
        print("%5d & %9.3e & %9.3e & %9.3e \\\\" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))
        f.write("%5d & %9.3e & %9.3e & %9.3e \\\\ \n" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3]))

