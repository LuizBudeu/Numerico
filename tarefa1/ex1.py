import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-np.pi, np.pi, 0.1)

y1 = np.cos(x) * 1
y2 = np.cos(x) * 2
y3 = np.cos(x) * 3

plt.plot(x, y1, 'g-', x, y2, 'g--', x, y3, 'g-.')
plt.xlabel('x (radianos), entre [-pi, pi]') 
plt.ylabel('y(x)')
plt.title('Tarefa 1 - Gráfico de diferentes funções cossenos')
plt.legend(['cos(x)', 'cos(2x)', 'cos(3x)'])      
plt.show()