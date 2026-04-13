import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --- 1. Параметри моделі (в одиницях СІ)
L = 15000.0   # м (Довжина річки)
W = 100.0    # м (Ширина річки)
U = 0.5      # м/с (Швидкість потоку)
Dx = 10.0    # м2/с (Поздовжня дисперсія)
Dy = 0.1     # м2/с (Поперечна дисперсія)
k1 = 1.0e-5  # 1/с (Константа споживання NH4+)

# Параметри сітки
dx = 50.0  # м
dy = 5.0   # м (Зменшив крок по Y для кращої точності поперек річки)
Nx = int(L / dx) + 1
Ny = int(W / dy) + 1
N_nodes = Nx * Ny

print(f"Grid: {Nx}x{Ny} ({N_nodes} nodes)")

# Початкові/Граничні умови
C_inlet = 0.5  # мг/м3
S_A_diffuse = 5.0e-5 # мг/м3/с (Трохи збільшив для наочності на графіку)

x_grid = np.linspace(0, L, Nx)
y_grid = np.linspace(0, W, Ny)

def build_2d_matrix_and_rhs(Nx, Ny, Dx, Dy, U, k, dx, dy, S_source_array, C_inlet):
    A = lil_matrix((N_nodes, N_nodes))
    B = np.zeros(N_nodes)

    # Коефіцієнти FDM
    CD_x = Dx / (dx**2)
    CD_y = Dy / (dy**2)

    # !!! ВИПРАВЛЕННЯ: Використовуємо Upwind (схема проти потоку) для адвекції
    # Це стабільніше ніж центральна різниця для рівнянь переносу
    # U * dC/dx approx U * (C[i] - C[i-1]) / dx
    C_adv = U / dx

    # Константа реакції
    CR = k

    for i in range(Nx):
        for j in range(Ny):
            p = i * Ny + j  # Глобальний індекс

            # --- 1. Вхід (x=0) - Dirichlet ---
            if i == 0:
                A[p, p] = 1.0
                B[p] = C_inlet

            # --- 2. Вихід (x=L) - Neumann dC/dx=0 ---
            elif i == Nx - 1:
                A[p, p] = 1.0
                A[p, p - Ny] = -1.0 # C[i] - C[i-1] = 0
                B[p] = 0.0

            # --- 3. Внутрішня область та бічні границі ---
            else:
                # Базова структура для внутрішніх вузлів (Upwind Scheme):
                # - D_x*d2C/dx2 - D_y*d2C/dy2 + U*dC/dx + k*C = S
                # Upwind для U*dC/dx -> U*(C_i - C_i-1)/dx

                # Коефіцієнти для X-напрямку (внутрішні):
                # C[i,j]:   2*CD_x + C_adv + CR
                # C[i-1,j]: -CD_x - C_adv
                # C[i+1,j]: -CD_x

                # Починаємо формувати рівняння: A*C = B -> переносимо все вліво, S вправо
                # Знак "-" бо ми формуємо -(Diffusion) + Advection + Reaction = Source
                # Але стандартний запис: Coeff_p * C_p + Sum(Coeff_n * C_n) = RHS

                # X-terms (Diffusion + Advection + Reaction)
                val_p = 2 * CD_x + C_adv + CR  # Діагональ
                val_im1 = -(CD_x + C_adv)      # i-1
                val_ip1 = -CD_x                # i+1

                A[p, p - Ny] = val_im1
                A[p, p + Ny] = val_ip1

                # Y-terms (Diffusion)
                # Розглядаємо границі по Y

                if j == 0:
                    # Лівий берег (Neumann dC/dy=0 -> C[j-1] = C[j+1])
                    # Апроксимація: -D_y * (C[j+1] - 2C[j] + C[j-1])/dy^2
                    # Заміна C[j-1] на C[j+1] -> -D_y * (2C[j+1] - 2C[j])/dy^2
                    # Це дає: -2*CD_y * C[j+1] + 2*CD_y * C[j]

                    A[p, p + 1] = -2 * CD_y     # Сусід (j+1)
                    val_p += -2 * CD_y          # Додаємо до діагоналі (зверніть увагу на знак)

                    # !!! Тут знак діагоналі +2*CD_y з рівняння дифузії,
                    # але ми перенесли все в одну сторону, де діагональ позитивна.
                    # Повна діагональ: (2*CD_x + C_adv + k) + (-2*CD_y ?) -> Ні.
                    # Рівняння: -D d2C - ... = S.
                    # Дискретизація: -CD*(C+ - 2C + C-) ...
                    # Діагональний внесок від дифузії завжди позитивний в матриці A (2*CD).
                    # Тому val_p вже містить X частину. Додаємо Y частину:
                    # При Neumann на границі член 2*C[j] залишається, отже додаємо 2*CD_y до діагоналі?
                    # Так, бо потік не виходить, але концентрація впливає на лапласіан.

                elif j == Ny - 1:
                    # Правий берег
                    A[p, p - 1] = -2 * CD_y
                else:
                    # Внутрішній вузол по Y
                    A[p, p + 1] = -CD_y
                    A[p, p - 1] = -CD_y
                    val_p += 2 * CD_y # Додаємо стандартні 2*CD_y до діагоналі

                # Записуємо фінальну діагональ
                A[p, p] = val_p

                # Права частина (Джерело)
                # Рівняння виду Matrix * C = Source.
                # Source array зазвичай позитивний (додавання маси).
                B[p] = S_source_array[p]

    return A.tocsr(), B

# --- 2. Створення масиву джерела
S_A_array = np.zeros(N_nodes)
# Джерело вздовж лівого берега (j=0) від x=1000м до x=3000м
for i in range(Nx):
    if 1000 <= x_grid[i] <= 3000:
        p = i * Ny + 0  # j=0
        S_A_array[p] = S_A_diffuse

# --- 3. Розв'язання
A_sparse, B_vector = build_2d_matrix_and_rhs(Nx, Ny, Dx, Dy, U, k1, dx, dy, S_A_array, C_inlet)
C_A_flat = spsolve(A_sparse, B_vector)
C_A_2D = C_A_flat.reshape(Nx, Ny)

# --- 4. Візуалізація
plt.figure(figsize=(10, 6))

# !!! ВИПРАВЛЕННЯ: Транспонуємо C_A_2D, щоб відповідати осям Meshgrid
# Meshgrid створює X shape (Ny, Nx) і Y shape (Ny, Nx)
# C_A_2D shape (Nx, Ny). Тому беремо .T -> (Ny, Nx)
X, Y = np.meshgrid(x_grid, y_grid)
contour = plt.contourf(X, Y, C_A_2D.T, levels=50, cmap='viridis')

plt.colorbar(contour, label='Концентрація NH4+, мг/м$^3$')
plt.title('2D Стаціонарний Розподіл Амонію (NH4+)')
plt.xlabel('Відстань вздовж річки (X), м')
plt.ylabel('Поперечна відстань (Y), м')
plt.show()

# Перевірка профілю
plt.figure(figsize=(8, 4))
mid_idx = Nx // 2
plt.plot(y_grid, C_A_2D[mid_idx, :], label=f'x={x_grid[mid_idx]} m')
plt.xlabel('Y, m')
plt.ylabel('C, mg/m3')
plt.title('Поперечний профіль посередині річки')
plt.grid()
plt.legend()
plt.show()