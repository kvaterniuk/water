import numpy as np
from scipy.sparse import lil_matrix # Для створення розрідженої матриці
from scipy.sparse.linalg import spsolve # КОНФІГурація: spsolve знаходиться в підмодулі linalg
import matplotlib.pyplot as plt

# 1. Налаштування Середовища та Геометрія (FDM Grid)
L_x, L_y = 10.0, 1.0  # Довжина та ширина домену (м)
Nx, Ny = 100, 10      # Кількість внутрішніх точок (вузлів)
dx = L_x / (Nx + 1)   # Крок сітки по X
dy = L_y / (Ny + 1)   # Крок сітки по Y

# 2. Визначення Фізичних Параметрів
D_x = 0.05    # Коефіцієнт поздовжньої дисперсії (D_L, м^2/с)
D_y = 0.005   # Коефіцієнт поперечної дисперсії (D_T, м^2/с)
k_nit = 0.005 # Константа нітрифікації (1/с)
k_den = 0.002 # Константа денітрифікації (1/с)

# Початкові та Граничні Умови
C1_in = 5.0   # Вхідна концентрація NH4+ (мг/л)
C2_in = 1.0   # Вхідна концентрація NO3- (мг/л)
N_total = Nx * Ny # Загальна кількість вузлів у сітці
N_eq = 2 * N_total # Загальна кількість рівнянь (2 концентрації на вузол)

# 3. Ідеалізоване Поле Швидкості U (Параболічний Профіль)
# Створення сітки координат для обчислення поля швидкості
X = np.linspace(dx, L_x - dx, Nx)
Y = np.linspace(dy, L_y - dy, Ny)
X_grid, Y_grid = np.meshgrid(X, Y, indexing='ij')

# U_x: U_max * (1 - (2y/L_y - 1)^2)
U_max = 1.0
# Параболічний профіль U_x: U_max * (1 - (2y/Ly - 1)^2)
# Тут y - це Y_grid, що змінюється від dy до Ly - dy.
U_x = U_max * (1.0 - (2.0 * Y_grid / L_y - 1.0)**2)
U_y = np.zeros_like(U_x) # Припускаємо нульову поперечну швидкість

# 4. Складання Розрідженої Матриці A та Вектора B
# A - матриця коефіцієнтів (2*Nx*Ny x 2*Nx*Ny)
A = lil_matrix((N_eq, N_eq), dtype=np.float64)
B = np.zeros(N_eq, dtype=np.float64) # Вектор правого боку

def flat_index(i, j, component):
    """Мапує індекси сітки (i, j) та компонент (0 або 1) на плоский індекс вектора."""
    # component=0 для C1 (NH4+), component=1 для C2 (NO3-)
    # Стовпці ідуть по j (y), потім по i (x), потім по компоненту
    return (i * Ny + j) * 2 + component

# Ітерація по всіх внутрішніх вузлах (i: x-напрямок, j: y-напрямок)
for i in range(Nx):
    for j in range(Ny):
        idx_C1 = flat_index(i, j, 0) # Індекс рівняння для C1
        idx_C2 = flat_index(i, j, 1) # Індекс рівняння для C2

        # Коефіцієнти для поточного вузла (i, j)
        Ux_ij = U_x[i, j]

        # 4.1. Рівняння для C1 (NH4+)
        # -- Дискретизований оператор Адвекції-Дисперсії-Реакції (АДР) --

        # Діагональний член C1(i, j) (включає центральні різниці та реакцію)
        # Сума коефіцієнтів для C(i, j)
        A[idx_C1, idx_C1] += (2*D_x/dx**2 + 2*D_y/dy**2 + k_nit)

        # Внесок від Дифузії (центральні різниці)
        # Дифузія по X
        if i > 0: A[idx_C1, flat_index(i-1, j, 0)] -= D_x/dx**2 # i-1
        if i < Nx - 1: A[idx_C1, flat_index(i+1, j, 0)] -= D_x/dx**2 # i+1
        # Дифузія по Y
        if j > 0: A[idx_C1, flat_index(i, j-1, 0)] -= D_y/dy**2 # j-1
        if j < Ny - 1: A[idx_C1, flat_index(i, j+1, 0)] -= D_y/dy**2 # j+1

        # Внесок від Адвекції (схема проти потоку, припускаємо Ux > 0)
        # Term: Ux/dx * (C_i - C_{i-1})
        A[idx_C1, idx_C1] += Ux_ij / dx
        if i > 0: A[idx_C1, flat_index(i-1, j, 0)] -= Ux_ij / dx

        # -- Граничні Умови (ГУ) для C1 --
        if i == 0:
            # Фіксація C1 на Вході (i=0) -> C1(0, j) = C1_in
            A[idx_C1, :] = 0.0 # Обнуляємо рядок
            A[idx_C1, idx_C1] = 1.0 # Встановлюємо діагональ = 1
            B[idx_C1] = C1_in # Встановлюємо правий бік = C1_in

        # 4.2. Рівняння для C2 (NO3-)

        # Діагональний член C2(i, j) (включає центральні різниці та реакцію)
        A[idx_C2, idx_C2] += (2*D_x/dx**2 + 2*D_y/dy**2 + k_den)

        # Внесок від Дифузії (аналогічно C1)
        if i > 0: A[idx_C2, flat_index(i-1, j, 1)] -= D_x/dx**2
        if i < Nx - 1: A[idx_C2, flat_index(i+1, j, 1)] -= D_x/dx**2
        if j > 0: A[idx_C2, flat_index(i, j-1, 1)] -= D_y/dy**2
        if j < Ny - 1: A[idx_C2, flat_index(i, j+1, 1)] -= D_y/dy**2

        # Зв'язок C1 -> C2 (Джерело NO3- від нітрифікації, -k_nit * C1)
        # Переносимо член -k_nit * C1 на лівий бік: + k_nit * C1.
        # У PDE це: - k_nit * C1 (джерело), тому в A, після перенесення, має бути -k_nit
        A[idx_C2, flat_index(i, j, 0)] += -k_nit

        # Внесок від Адвекції (аналогічно C1)
        A[idx_C2, idx_C2] += Ux_ij / dx
        if i > 0: A[idx_C2, flat_index(i-1, j, 1)] -= Ux_ij / dx

        # -- Граничні Умови (ГУ) для C2 --
        if i == 0:
            # Фіксація C2 на Вході (i=0) -> C2(0, j) = C2_in
            A[idx_C2, :] = 0.0
            A[idx_C2, idx_C2] = 1.0
            B[idx_C2] = C2_in

# 5. Розв'язання Системи Рівнянь
A_sparse = A.tocsr() # Перетворення в CSR формат для ефективного розв'язання
# Використовуємо spsolve з scipy.sparse.linalg
C_solution_vec = spsolve(A_sparse, B)

# 6. Розділення розв'язку для постобробки
C1_sol_flat = C_solution_vec[::2]
C2_sol_flat = C_solution_vec[1::2]

# Зміна форми вектора на 2D сітку (Nx x Ny)
C1_sol_grid = C1_sol_flat.reshape((Nx, Ny))
C2_sol_grid = C2_sol_flat.reshape((Nx, Ny))

print("Чисельний розв'язок отримано за допомогою МСР (NumPy/SciPy).")

# Додатково: Візуалізація результатів (потрібен matplotlib)
plt.figure(figsize=(12, 5))

# Графік для NH4+
plt.subplot(1, 2, 1)
# Транспонуємо C1_sol_grid, щоб вісь Y (j) була вертикальною
plt.imshow(C1_sol_grid.T, aspect='auto', origin='lower',
           extent=[0, L_x, 0, L_y], cmap='viridis')
plt.colorbar(label='Концентрація $C_1$ (NH$_4^+$) [мг/л]')
plt.title('Розподіл NH$_4^+$ у річці')
plt.xlabel('Поздовжній напрямок, x [м]')
plt.ylabel('Поперечний напрямок, y [м]')

# Графік для NO3-
plt.subplot(1, 2, 2)
plt.imshow(C2_sol_grid.T, aspect='auto', origin='lower',
           extent=[0, L_x, 0, L_y], cmap='plasma')
plt.colorbar(label='Концентрація $C_2$ (NO$_3^-$) [мг/л]')
plt.title('Розподіл NO$_3^-$ у річці')
plt.xlabel('Поздовжній напрямок, x [м]')
plt.ylabel('Поперечний напрямок, y [м]')

plt.tight_layout()
plt.show()