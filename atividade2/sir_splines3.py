import numpy as np
import matplotlib.pyplot as plt


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


# Dados de exemplo
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 4, 2, 3, 0])

# Calcular os coeficientes de splines
a, b, c, d = spline(x, y)

for i in range(len(x)-1):
    print(f"S{i}(x) = {a[i]} + {b[i]}(x-{x[i]}) + {c[i]}(x-{x[i]})^2 + {d[i]}(x-{x[i]})^3")


# Avaliar o polinômio de splines em um conjunto de pontos
xp = np.linspace(0, len(x)-1, 100)
yp = eval_spline(x, a, b, c, d, xp)

# Plotar os resultados
plt.plot(x, y, 'o', label='Dados')
plt.plot(xp, yp, label='Splines')
plt.legend()
plt.show()




# import numpy as np

# def spline_interpolation(x, y):
#     n = len(x)

#     # Step 1: Define the matrix A and vector b
#     A = np.zeros((n, n))
#     b = np.zeros(n)
#     for i in range(1, n-1):
#         h1 = x[i] - x[i-1]
#         h2 = x[i+1] - x[i]
#         A[i, i-1:i+2] = [h1, 2*(h1+h2), h2]
#         b[i] = 3*((y[i+1]-y[i])/h2 - (y[i]-y[i-1])/h1)

#     # Step 2: Solve the system of equations for the spline coefficients
#     c = np.linalg.solve(A, b)
#     a = y[:-1]
#     b = (y[1:]-y[:-1])/(x[1:]-x[:-1]) - (2*c[:-1]+c[1:])*np.diff(x)/6
#     d = (c[1:]-c[:-1])/(6*np.diff(x))

#     # Step 3: Construct the spline function
#     def spline_function(x_new):
#         i = np.searchsorted(x, x_new) - 1
#         dx = x_new - x[i]
#         return a[i] + b[i]*dx + c[i]*dx**2/2 + d[i]*dx**3

#     return spline_function


# import matplotlib.pyplot as plt

# # Dados de entrada
# x = np.array([0, 1, 2, 3, 4])
# y = np.array([0, 1, 4, 9, 16])

# # Interpolação por splines
# f = spline_interpolation(x, y)

# # Plot da função interpolada
# x_new = np.linspace(0, 4, 100)
# y_new = f(x_new)

# plt.plot(x, y, 'o', label='Dados')
# plt.plot(x_new, y_new, label='Spline')
# plt.legend()
# plt.show()
