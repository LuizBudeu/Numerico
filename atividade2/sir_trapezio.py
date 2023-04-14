import numpy as np
import matplotlib.pyplot as plt
import math
from constants import *
import json


def SIR_model(Y, t):
    S, I, R = Y
    dotS = -beta * S * I / N
    dotI = beta * S * I / N - gamma * I
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
    y_all = []
    t_all = []
    for i in range(len(n)):        
        # Time points
        t = np.linspace(t0, T, num=n[i]+1) # Para incluir T            

        y = trapezios(Y0, t, h[i])
        y_T.append(y[-1])
        
        if i == 1:  # Salvar os pontos para n=32 para ser usado no spline
            with open('atividade2/splines_pontos/sir_trapezio_32.json', 'w') as f:
                xs = (np.linspace(t0, T, 32))
                f.write(json.dumps({x: list(yy) for x, yy in zip(xs, y)}, indent=4))
        
        if i == 4:  # Salvar os pontos para n=256 para ser usado no spline
            with open('atividade2/splines_pontos/sir_trapezio_256.json', 'w') as f:
                xs = (np.linspace(t0, T, 256))
                f.write(json.dumps({x: list(yy) for x, yy in zip(xs, y)}, indent=4))
                
            # ATENCAO: se quiser ver o valor de y(60) para n=526, descomente a linha abaixo
            # print(y[list(t).index(min(t, key=lambda x:abs(x-60)))])  # Mostra os valores de y(60) para n=526
    
        p = e = q =  0
        if i > 0:
            q0 = abs((y_T[i-2][0] - y_T[i-1][0]) / (y_T[i-1][0] - y_T[i][0]))
            q1 = abs((y_T[i-2][1] - y_T[i-1][1]) / (y_T[i-1][1] - y_T[i][1]))
            q2 = abs((y_T[i-2][2] - y_T[i-1][2]) / (y_T[i-1][2] - y_T[i][2]))
            q = min(q0,q1,q2)
            r = h[i-1]/h[i]
      
            p0 = math.log(q0)/math.log(r)
            p1 = math.log(q1)/math.log(r)
            p2 = math.log(q2)/math.log(r)
            p = min(p0, p1, p2)
      
            e0 = abs((y_T[i-1][0]-y_T[i][0]))
            e1 = abs((y_T[i-1][1]-y_T[i][1]))
            e2 = abs((y_T[i-1][2]-y_T[i][2]))
            e = math.sqrt(e0**2 + e1**2 + e2**2)

        i_solution = (n[i], h[i], e, q, p)
        solution.append(i_solution)

        if i == len(n) - 1:
            print('Estimativa - Erros finais')
            print('{:.5e} {:.5e} {:.5e}'.format(e0, e1, e2), end ='\n\n')
            print('Estimativas - Ordens de convergência finais')
            print('{:.5f} {:.5f} {:.5f}'.format(p0, p1, p2), end ='\n\n')
            # gerar_grafico_indiv(t, y)
            gerar_grafico_conj(t, y)
        if i < 6:
            t_all.append(t)
            y_all.append(y)
    return y_all, t_all


def gerar_grafico_indiv(t_n, y_n):
    for index, var in enumerate(STATE_VAR):
        plt.title(f"Modelo SIR - {var} - Método do Trapézio")
        plt.plot(t_n, y_n[:,index], '--', color="black", label=f"{var}")
        plt.grid()
        plt.xlabel("Tempo, $t$ [dias]")
        plt.ylim(-0.02*N, 1.02*N)
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
    plt.ylabel("População")
    plt.legend(loc = "best")
    plt.show()

def gerar_grafico_aproxs(t_all, y, var_index):
    
    for i, t in enumerate(t_all):
        plt.plot(t, y[i], '-.', color='black',label=f"{2**(i+4)} partições")
    plt.title(f"Convergência da curva de {STATE_VAR[var_index]} - 16 a 512 partições")
    plt.grid()
    plt.xlabel("Tempo, $t$ [dias]")
    plt.ylim(-0.02*N, 1.02*N)
    plt.ylabel("População")
    plt.show()

yal, tal = tabela(n)
varsolutions = [[],[],[]]

for j in range(len(yal[0][0])):
    for i in range(len(yal)):
        varsolutions[j].append(yal[i][:,j])
    gerar_grafico_aproxs(tal, varsolutions[j], j)

with open('atividade2/sir.txt', 'w') as f:

    for i in range(len(solution)):
        print("%5d & %9.3e & %9.3e & %9.3e & %9.3e \\\\" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3], solution[i][4]))
        f.write("%5d & %9.3e & %9.3e & %9.3e & %9.3e \\\\ \n" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3], solution[i][4]))

