import numpy as np
import matplotlib.pyplot as plt
from constants import *


def SIR_model(Y, t):
    S, I, R = Y
    dotS = -beta * S * I / N
    dotI = beta * S * I / N - gamma * I
    dotR = gamma * I
    return np.array([dotS, dotI, dotR])


# Método de Runge-Kutta de 4ª ordem
def RK4(Y0, t, h):
    nt = len(t)
    Y = np.zeros([nt, len(Y0)])
    Y[0] = Y0
    for i in range(1, nt):
        k1 = h * SIR_model(Y[i-1], t[i-1])
        k2 = h * SIR_model(Y[i-1] + k1/2, t[i-1] + h/2)
        k3 = h * SIR_model(Y[i-1] + k2/2, t[i-1] + h/2)
        k4 = h * SIR_model(Y[i-1] + k3, t[i-1] + h)
        Y[i] = Y[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
    return Y


def tabela(n):

    y_T = [] # Armazena a aprox de y(T) para os n casos distintos
    
    for i in range(len(n)):        
        # Time points
        t = np.linspace(t0, T, num=n[i]+1) # Para incluir T

        y = RK4(Y0, t, h[i])
        y_T.append(y)
        
        # if i > 0:
        #     print(exact(T)[0], y_T[i-1][:,0])
        #     q0 = abs((exact(T)[0]-y_T[i-1][:,0])/(exact(T)[0]-y_T[i][:,0]))
        #     q1 = abs((exact(T)[1]-y_T[i-1][:,1])/(exact(T)[1]-y_T[i][:,1]))
        #     q2 = abs((exact(T)[2]-y_T[i-1][:,2])/(exact(T)[2]-y_T[i][:,2]))
        #     r = h[i-1]/h[i]
            
        #     print(q0)
        #     p0 = math.log(q0)/math.log(r)
        #     p1 = math.log(q1)/math.log(r)
        #     p2 = math.log(q2)/math.log(r)
        #     p = max(p0, p1, p2)
            
        #     e0 = abs((exact(T)[0]-y_T[i][0]))
        #     e1 = abs((exact(T)[1]-y_T[i][1]))
        #     e2 = abs((exact(T)[2]-y_T[i][2]))
        #     e = math.sqrt(e0**2 + e1**2 + e2**2)
        
        # if i == len(n)-1:
        #     # print(p0,p1,p2)
        #     # print(e0,e1,e2)
        #     gerar_grafico(t, y)
        
        gerar_grafico(t, y)



def gerar_grafico(t_n, y_n):
    plt.title("Método de Euler")
    plt.plot(t_n, y_n[:,0], color='orange', label='Suscetíveis')
    plt.plot(t_n, y_n[:,1], color='green', label='Infectados')
    plt.plot(t_n, y_n[:,2], color='lightblue', label='Recuperados')
    plt.grid()
    plt.xlabel("Tempo, $t$ [dias]")
    plt.ylabel("População")
    plt.legend(loc = "best")

    plt.show();


a = tabela(n)

with open('atividade2/sir.txt', 'w') as f:

    for i in range(len(solution)):
        print("%5d & %9.3e & %9.3e & %9.3e & %9.3e \\\\" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3], solution[i][4]))
        f.write("%5d & %9.3e & %9.3e & %9.3e & %9.3e \\\\ \n" % (solution[i][0], solution[i][1], solution[i][2], solution[i][3], solution[i][4]))

