import numpy as np
import matplotlib.pyplot as plt
import json


def spline(x, y):
    n = len(x)
    a = y.copy()
    h = np.diff(x)
    alpha = np.zeros(n-2)
    for i in range(1, n-1):
        alpha[i-1] = (3/h[i-1])*(a[i+1]-a[i]) - (3/h[i])*(a[i]-a[i-1])
    
    l = np.zeros(n)
    u = np.zeros(n)
    z = np.zeros(n)
    l[0] = 1
    u[0] = 0
    z[0] = 0
    for i in range(1, n-1):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*u[i-1]
        u[i] = h[i]/l[i]
        z[i] = (alpha[i-1]-h[i-1]*z[i-1])/l[i]
    l[n-1] = 1
    z[n-1] = 0
    
    c = np.zeros(n)
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    for j in range(n-2, -1, -1):
        c[j] = z[j] - u[j]*c[j+1]
        b[j] = (a[j+1]-a[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
        
    return a, b, c, d


def eval_spline(x, a, b, c, d, xp):
    n = len(x)
    yp = np.zeros_like(xp)
    for i in range(n-1):
        idx = np.where((xp >= x[i]) & (xp <= x[i+1]))
        yp[idx] = a[i] + b[i]*(xp[idx]-x[i]) + c[i]*(xp[idx]-x[i])**2 + d[i]*(xp[idx]-x[i])**3
    return yp

def main(n):
    data = json.load(open(f'atividade2/splines_pontos/sir_trapezio_{n}.json', 'r'))

    x = np.array(list(map(float, data.keys())))
    y1 = np.array([i[0] for i in data.values()])
    y2 = np.array([i[1] for i in data.values()])
    y3 = np.array([i[2] for i in data.values()])
    xp = np.linspace(0, 160, 1000)
    
    if n == 32:
        plt.plot(x, y1, 'o', color='black', label='Pontos - Suscetíveis')
        plt.plot(x, y2, 'o', color='black', label='Pontos - Infectados')
        plt.plot(x, y3, 'o', color='black', label='Pontos - Recuperados')
        
    a, b, c, d = spline(x, y1)
    yp = eval_spline(x, a, b, c, d, xp)
    plt.plot(xp, yp, ':', color="black", label='Splines - Suscetíveis')
    
    print_y_60(x, a, b, c, d)

    a, b, c, d = spline(x, y2)
    yp = eval_spline(x, a, b, c, d, xp)
    plt.plot(xp, yp, '-.', color="black", label='Splines - Infectados')
    
    print_y_60(x, a, b, c, d)

    a, b, c, d = spline(x, y3)
    yp = eval_spline(x, a, b, c, d, xp)
    plt.plot(xp, yp, '--', color="black", label='Splines - Recuperados')
    
    print_y_60(x, a, b, c, d)
    
    plt.grid()
    plt.legend()
    plt.xlabel("Tempo, $t$ [dias]")
    plt.ylabel("População")
    plt.show()
    
    
def print_y_60(x, a, b, c, d):
    y_60 = eval_spline(x, a, b, c, d, np.linspace(60, 60, 1))
    print(y_60)


# ATENCAO: É possível rodar a 'main' para n = 32 ou n = 256; comentar/descomentar
if __name__ == "__main__":
    # main(32)
    main(256)
