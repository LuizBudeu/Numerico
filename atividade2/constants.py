# N: número total de indivíduos
N = 10E6  

# Número inicial de indivíduos infectados e recuperados
I0, R0 = 1, 0  

# Número inicial de indivíduos suscetíveis a infecção
S0 = N - I0 - R0 

# Força de infecção e taxa de recuperação
Lambda, gamma = 0.4, 0.1  

# Vetor de condições iniciais
Y0 = S0, I0, R0  

# Intervalo de tempo
T = 160
t0 = 0
n = [pow(2,i) for i in range(4,16)]
h = [(T-t0)/i for i in n]