import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --- Параметри (ті самі) ---
L, W = 1000.0, 100.0
U = 0.5
Dx, Dy = 10.0, 0.1
k1 = 1.0e-5
dx, dy = 1.0, 1.0 # Крок
Nx, Ny = int(L/dx)+1, int(W/dy)+1
N_nodes = Nx * Ny
C_inlet = 0.5

x_grid = np.linspace(0, L, Nx)
y_grid = np.linspace(0, W, Ny)

def build_stable_matrix(Nx, Ny, Dx, Dy, U, k, dx, dy, S_source_array, C_inlet):
    A = lil_matrix((N_nodes, N_nodes))
    B = np.zeros(N_nodes)

    # Коефіцієнти дифузії
    CD_x = Dx / (dx**2)
    CD_y = Dy / (dy**2)

    # !!! ГОЛОВНЕ ВИПРАВЛЕННЯ: Коефіцієнт адвекції для Upwind схеми
    # Замість U/(2dx) ми використовуємо U/dx
    C_adv = U / dx

    # Реакція
    CR = k

    for i in range(Nx):
        for j in range(Ny):
            p = i * Ny + j

            # --- 1. Вхід (Dirichlet) ---
            if i == 0:
                A[p, p] = 1.0
                B[p] = C_inlet

            # --- 2. Вихід (Neumann) ---
            elif i == Nx - 1:
                A[p, p] = 1.0
                A[p, p - Ny] = -1.0
                B[p] = 0.0

            # --- 3. Внутрішні вузли та береги ---
            else:
                # Формуємо рівняння:
                # (Адвекція + Дифузія + Реакція) * C = Джерело
                # Використовуємо Upwind для першої похідної: (C[i] - C[i-1])/dx

                # Коефіцієнт при поточному вузлі C[i, j] (Діагональ)
                # Він складається з: витоку дифузії (2*Dx + 2*Dy), витоку адвекції (+U/dx) та реакції (+k)
                # Зверніть увагу: всі доданки діагоналі мають бути позитивними для стабільності
                val_p = (2 * CD_x + 2 * CD_y) + C_adv + CR

                # Коефіцієнт при C[i-1, j] (Сусід ззаду / Upstream)
                # Сюди приходить адвекція (-U/dx) і дифузія (-Dx)
                val_im1 = - (CD_x + C_adv)

                # Коефіцієнт при C[i+1, j] (Сусід спереду / Downstream)
                # Сюди діє ТІЛЬКИ дифузія (адвекція проти потоку не йде)
                val_ip1 = - CD_x

                # Заповнюємо матрицю по X
                A[p, p] = val_p
                A[p, p - Ny] = val_im1
                A[p, p + Ny] = val_ip1

                # Заповнюємо матрицю по Y (Береги - умови Неймана)
                if j == 0: # Лівий берег
                    A[p, p + 1] = -2 * CD_y # Подвійний внесок через "дзеркало"
                    # Діагональ не змінюємо, вона вже містить 2*CD_y
                elif j == Ny - 1: # Правий берег
                    A[p, p - 1] = -2 * CD_y
                else: # Середина річки
                    A[p, p + 1] = -CD_y
                    A[p, p - 1] = -CD_y

                # Права частина
                B[p] = S_source_array[p]

    return A.tocsr(), B

# --- Тестування ---
# Створюємо тестове джерело
S_arr = np.zeros(N_nodes)
# Додамо точкове джерело посередині для перевірки
S_arr[(Nx//4)*Ny + Ny//2] = 1.0e-4

A_mat, B_vec = build_stable_matrix(Nx, Ny, Dx, Dy, U, k1, dx, dy, S_arr, C_inlet)
C_res = spsolve(A_mat, B_vec)

# Перевірка на від'ємні значення
min_val = np.min(C_res)
print(f"Мінімальна концентрація: {min_val}")

if min_val < -1e-10:
    print("УВАГА: Все ще є значні від'ємні значення!")
else:
    print("УСПІХ: Від'ємні значення зникли (або в межах машинної похибки).")

# Примусове обнулення машинної похибки (наприклад -1e-20)
C_res[C_res < 0] = 0

# Візуалізація
C_2D = C_res.reshape(Nx, Ny).T
plt.figure(figsize=(10, 4))
plt.imshow(C_2D, extent=[0, L, 0, W], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='C')
plt.title('Розподіл (Upwind Scheme - Стабільний)')
plt.show()