import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --- 1. Параметри моделі ---
L = 2000.0   # м (Довжина ділянки)
W = 50.0     # м (Ширина річки)
U = 0.5      # м/с (Швидкість течії)
Dx = 5.0     # м2/с (Поздовжня дисперсія)
Dy = 0.2     # м2/с (Поперечна дисперсія - КЛЮЧОВИЙ параметр для ширини шлейфу)
k1 = 0.0     # 1/с (Для наочності приберемо розпад, дивимось тільки на змішування)

# Сітка
dx = 20.0    # м
dy = 1.0     # м (Робимо дрібну сітку по Y, щоб гарно бачити профіль)
Nx = int(L / dx) + 1
Ny = int(W / dy) + 1
N_nodes = Nx * Ny
print(f"Сітка: {Nx}x{Ny} вузлів")

x_grid = np.linspace(0, L, Nx)
y_grid = np.linspace(0, W, Ny)

# --- 2. Функція побудови матриці (Stable Upwind Scheme) ---
def build_matrix(Nx, Ny, Dx, Dy, U, k, dx, dy, S_source, C_inlet):
    A = lil_matrix((N_nodes, N_nodes))
    B = np.zeros(N_nodes)

    CD_x = Dx / dx**2
    CD_y = Dy / dy**2
    C_adv = U / dx  # Upwind
    CR = k

    for i in range(Nx):
        for j in range(Ny):
            p = i * Ny + j

            # Вхід (x=0)
            if i == 0:
                A[p, p] = 1.0
                B[p] = C_inlet
            # Вихід (x=L)
            elif i == Nx - 1:
                A[p, p] = 1.0
                A[p, p - Ny] = -1.0
                B[p] = 0.0
            # Внутрішня область
            else:
                # Upwind по X: (C[i] - C[i-1])/dx
                # Дифузія по X та Y

                # Коефіцієнти
                val_p = (2*CD_x + 2*CD_y) + C_adv + CR # Діагональ
                val_im1 = -(CD_x + C_adv)              # i-1
                val_ip1 = -CD_x                        # i+1

                A[p, p] = val_p
                A[p, p - Ny] = val_im1
                A[p, p + Ny] = val_ip1

                # Y-сусіді (Граничні умови Неймана на берегах)
                if j == 0: # Лівий берег
                    A[p, p + 1] = -2 * CD_y
                elif j == Ny - 1: # Правий берег
                    A[p, p - 1] = -2 * CD_y
                else:
                    A[p, p + 1] = -CD_y
                    A[p, p - 1] = -CD_y

                B[p] = S_source[p]

    return A.tocsr(), B

# --- 3. Створення Сценарію Скиду ---
S_array = np.zeros(N_nodes)
C_inlet = 0.0 # Чиста вода на вході

# Точковий скид (Труба) на відстані 200м від входу, на лівому березі (y=0)
source_x_idx = int(200 / dx) # Індекс по X
source_strength = 0.1 # Інтенсивність джерела (мг/м3/с)

# Додаємо джерело в одну точку (або декілька сусідніх для плавності)
p_source = source_x_idx * Ny + 0 # j=0 (лівий берег)
S_array[p_source] = source_strength
# S_array[p_source + 1] = source_strength # Можна трохи розширити трубу

# --- 4. Розв'язок ---
A_mat, B_vec = build_matrix(Nx, Ny, Dx, Dy, U, k1, dx, dy, S_array, C_inlet)
C_flat = spsolve(A_mat, B_vec)
C_2D = C_flat.reshape(Nx, Ny).T # Транспонуємо для (Ny, Nx) - (Y, X)

# --- 5. Візуалізація та Аналіз Поперечного Профілю ---

fig = plt.figure(figsize=(14, 10))

# Графік 1: Карта забруднення (вид зверху)
ax1 = plt.subplot(2, 1, 1)
extent = [0, L, 0, W]
im = ax1.imshow(C_2D, extent=extent, origin='lower', aspect='auto', cmap='jet')
ax1.set_title('Карта шлейфу забруднення (Вид зверху)')
ax1.set_ylabel('Ширина річки (м)')
ax1.set_xlabel('Відстань по річці (м)')
plt.colorbar(im, ax=ax1, label='Концентрація')

# Додамо лінії, де ми будемо "різати" поперечні профілі
cuts_x = [300, 600, 1000, 1800] # В метрах
colors = ['white', 'cyan', 'yellow', 'orange']
for xc, col in zip(cuts_x, colors):
    ax1.axvline(x=xc, color=col, linestyle='--', alpha=0.8, label=f'x={xc}м')
ax1.legend()

# Графік 2: Поперечні профілі (Концентрація vs Ширина)
ax2 = plt.subplot(2, 1, 2)

for xc, col in zip(cuts_x, colors):
    # Знаходимо індекс сітки, найближчий до відстані xc
    idx = int(xc / dx)
    if idx < Nx:
        # Беремо зріз даних: стовпчик idx з матриці (до транспонування це був рядок)
        # У C_2D (після transpose) це стовпчик: C_2D[:, idx]
        profile = C_2D[:, idx]

        # Замінимо колір white на black для видимості на білому фоні графіку
        plot_col = 'black' if col == 'white' else col

        ax2.plot(y_grid, profile, color=plot_col, linewidth=2, label=f'Профіль на {xc} м')

ax2.set_title('Поперечні профілі концентрації (Розмивання шлейфу)')
ax2.set_xlabel('Відстань від лівого берега (м)')
ax2.set_ylabel('Концентрація (мг/м$^3$)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()