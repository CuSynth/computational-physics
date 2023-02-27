import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np

L = 1
T =  5

N = 100     # Шагов по x и y
middle = int(np.floor(N/2)) # Положение центра
M = 1000    # Шагов по t

h = 2*L/N 
tau = T/M / 2   # Полушаг по t

r = tau / h**2 # Замена

x = np.linspace(-L, L, N + 1)
y = np.linspace(-L, L, N + 1)

X, Y = np.meshgrid(x, y)

def plot_temp(temp):    # Строит распределение температуры
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, temp, rstride=1, cstride=1, 
                    cmap='viridis', edgecolor='none')

    ax.set_title('Initial temperature distribution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    
    if np.min(temp) != np.max(temp):
        ax.set_zlim((np.min(temp), np.max(temp)))


def animate(aX, aY, aZ, fps, filename):     # Записывает результат расчетов как видео с распределением
   plot_args = {'rstride' : 1, 'cstride' : 1,
             'cmap' : 'viridis', 'edgecolor' : 'none'}
   frn = np.shape(aZ)[0]

   zs = np.zeros((np.shape(aZ)[1], np.shape(aZ)[2], frn))
   for i in range(frn):
      zs[:, :, i] = aZ[i]


   fig = plt.figure(figsize=(10,8))
   ax = fig.add_subplot(111, projection='3d')
   
   ax.set_zlim((np.min(aZ), np.max(aZ)))

   ax.set_xlabel('x')
   ax.set_ylabel('y')
   ax.set_zlabel('u')


   def change_plot(frame_number, zarray, plot):
      plot[0].remove()
      plot[0] = ax.plot_surface(aX, aY, zarray[:, :, frame_number], **plot_args)

   plot = [ax.plot_surface(aX, aY, zs[:, :, 0], **plot_args)]

   anim = animation.FuncAnimation(fig, change_plot, frn, fargs=(zs, plot), interval=1000/fps, save_count=frn)
   writervideo = animation.FFMpegWriter(fps=fps) 
   anim.save(filename, writer=writervideo)


def plot_t_middle(temps, times):   # Строит температуры от времени
    plt.figure()
    plt.plot(times, temps)
    plt.xlabel('time')
    plt.ylabel('T')
    plt.title('Temperature in the middle')
    plt.grid()
    plt.show()


def matrix_X(w, t): # Матрица для Х
    A = (-r/2) * np.ones(N+1)
    A[0]=0
    A[-1]=0
    
    B = (1+r) * np.ones(N+1)
    B[0]=1
    B[-1]=1
    
    C = (-r/2) * np.ones(N+1)
    C[0]=0
    C[-1]=0
    
    D = (r/2)*np.roll(w, 1) +  (1-r)*w + (r/2)*np.roll(w,-1)
    D[0] = np.sin(2*np.pi*t)    # Гранусловия на функцию
    D[-1] = np.sin(2*np.pi*t)

    return A, B, C, D


def matrix_Y(w, t): # Матрица для У
    A = (-r/4) * np.ones(N+1)
    A[0]=0
    A[-1]=-1
    
    B = (1+r/2) * np.ones(N+1)
    B[0]=1
    B[-1]=1
    
    C = (-r/4) * np.ones(N+1)
    C[0]=-1
    C[-1]=0
    
    D = (r/4)*np.roll(w, 1) +  (1-r/2)*w + (r/4)*np.roll(w,-1)
    D[0]=0
    D[-1]=0

    return A, B, C, D



def TMA(w, t, mtx):
    A, B, C, D = mtx(w, t)

    for i in range(N):
        xi = A[i+1] / B[i]
        B[i+1] -= xi * C[i]
        D[i+1] -= xi * D[i]
    
    W = np.zeros(N + 1)
    W[-1] = D[-1] / B[-1]
    
    for i in range(N - 1, -1, -1):
        W[i] = (D[i] - C[i] * W[i + 1]) / B[i]
    
    return W


# Здесь храним М матриц размера (N+1, N+1) (так как точки у нас u0..uN, то точек N+1 штука)
# В них будут лежать итоговые температуры
W = [ [np.zeros(N + 1), np.zeros(N + 1)] for _ in range(M)] 

# Здесь будут температуры, которые были в центре на каждом шаге 
T_middle = np.zeros(M) 

# Пусть в самом начале будет функция с производной по у на границе = 0
# и значением на другой границе 0 (так как рассматриваем начальный момент времени, то sin(2*pi*t=0) = 0)
u0 = lambda x, y: (1 - x**2 / L**2) * (2*y/L - y**2 / L**2)   
# u0 = lambda x, y: (x*0) * (y*0) # Хотя можно взять тождественный 0, он тоже подходит

Z0 = u0(X, Y) 
W[0] = Z0

# Можем посмотреть на начальное распределение:
# plot_temp(W[0])
# plt.show()

T_middle[0] = W[0][middle][middle]
for i in range(1, M):
    prev_Ts = W[i-1] # Массив (N+1)x(N+1)

    half_step = []

    for along_x in prev_Ts:
       half_step.append(TMA(along_x, tau*(2*i-1), matrix_X))

    half_step = np.array(half_step).transpose()

    full_step = []

    for along_y in half_step:
        full_step.append(TMA(along_y, tau*(2*i), matrix_Y)) 

    W[i] = np.array(full_step).transpose()
    
    T_middle[i] = W[i][middle][middle]


# Этой штукой можем строить анимацию. Просто задаем название файла и получаем красивое видео.
# Считается долго, поэтому можно взять только первую треть от всего времени: W[:(len(W)//3)]
# animate(X, Y, W[:], fps=50, filename=r"./zero.mp4");

plot_t_middle(T_middle, np.linspace(0, T, M))