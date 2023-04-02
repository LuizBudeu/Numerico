import numpy as np
import matplotlib.pyplot as plt
from constants import *


# # Condições iniciais
# S0 = 10E6 - 1
# I0 = 1
# R0 = 0

# # Equações diferenciais
# def SIR_model(t, y, beta, gamma):
#     S, I, R = y
#     dSdt = -beta*S*I
#     dIdt = beta*S*I - gamma*I
#     dRdt = gamma*I
#     return np.array([dSdt, dIdt, dRdt])

# # Parâmetros
# beta = 0.4
# gamma = 0.1

# # Intervalo de tempo
# t = np.linspace(0, 100, 1)

# # Número de pontos
# n = len(t)

# # Função que calcula os valores dos pontos usando o método dos trapézios
# def trapezoidal(f, a, b, n, y0):
#     h = (b - a) / float(n)
#     w = np.zeros((n, len(y0)))
#     w[0] = y0
#     for i in range(1, n):
#         w[i] = w[i-1] + h/2 * (f(t[i-1], w[i-1], beta, gamma) + f(t[i], w[i-1] + h*f(t[i-1], w[i-1], beta, gamma), beta, gamma))
#     return  w

# # Implementando o método dos trapézios para resolver o modelo SIR
# y = trapezoidal(SIR_model, 0, 100, n, np.array([S0, I0, R0]))

# print(y)

# # Plotando os resultados
# plt.plot(range(0, 100), y[:,0], label='Suscetíveis')
# plt.plot(range(0, 100), y[:,1], label='Infectados')
# plt.plot(range(0, 100), y[:,2], label='Recuperados')
# plt.xlabel('Tempo')
# plt.ylabel('Número de Indivíduos')
# plt.title('Modelo SIR com o Método dos Trapézios')
# plt.legend()
# plt.show()


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


def tabela(n):

    y_T = [] # Armazena a aprox de y(T) para os n casos distintos
    
    for i in range(len(n)):        
        # Time points
        t = np.linspace(t0, T, num=n[i]+1) # Para incluir T

        y = trapezios(Y0, t, h[i])
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
        
        if i == len(n)-1:
            # print(p0,p1,p2)
            # print(e0,e1,e2)
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


