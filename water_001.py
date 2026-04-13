import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve

def global_k(i, j, Nx):
    """Перетворення 2D індексу (i, j) на 1D глобальний індекс k."""
    return j * Nx + i

def build_system(Lx, Ly, Nx, Ny, Dx, Dy, Ux, Uy, k_decay, Cin):
    """
    Побудова розрідженої матриці A та вектора правої частини b.
    """

    # 1. Параметри сітки
    N = Nx * Ny
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    # 2. Коефіцієнти дискретизації
    Px = Dx / dx**2
    Py = Dy / dy**2
    Qx = Ux / dx
    Qy = Uy / dy
    K = k_decay

    # 3. Ініціалізація структур COO та вектора b
    # ВИПРАВЛЕНО: Коректна ініціалізація списків
    rows, cols, data = [], [], []
    b = np.zeros(N)

    # Цикли по вузлах (j - рядки, i - стовпці)
    for j in range(Ny):
        for i in range(Nx):
            k = global_k(i, j, Nx)

            # --- Обробка Вхідної Границі (Діріхле, i=0) ---
            if i == 0:
                # Фіксуємо рівняння: C_k = C_in
                rows.append(k); cols.append(k); data.append(1.0)
                b[k] = Cin
                continue # Перехід до наступного вузла

            # --- Обробка Внутрішніх та Неймана вузлів (i > 0) ---

            # Базові коефіцієнти (Upwind схема)
            # Ac (Central), Ae (East), Aw (West), An (North), As (South)
            Ac = -2*Px - 2*Py - Qx - Qy - K
            Ae, Aw, An, As = Px, Px + Qx, Py, Py + Qy

            # --- Модифікація для Умов Неймана (Боки та Вихід) ---
            # ВИПРАВЛЕНО: Видалено некоректне додавання до протилежних сусідів.
            # Умова dC/dn = 0 означає C_{i+1} = C_i, тому Ae додається тільки до Ac.

            # Вихідний Нейман (i = Nx - 1)
            if i == Nx - 1:
                Ac += Ae
                Ae = 0

            # Нижній Нейман (j = 0)
            if j == 0:
                Ac += As
                As = 0

            # Верхній Нейман (j = Ny - 1)
            if j == Ny - 1:
                Ac += An
                An = 0

            # --- Додавання коефіцієнтів до COO ---

            # 1. Центральний вузол (k, k)
            rows.append(k); cols.append(k); data.append(Ac)

            # 2. Схід (k, k+1)
            if i < Nx - 1 and Ae != 0:
                rows.append(k); cols.append(global_k(i+1, j, Nx)); data.append(Ae)

            # 3. Захід (k, k-1)
            if i > 0 and Aw != 0:
                # Оскільки ми вже обробили i=0 вище через continue,
                # тут ми просто перевіряємо чи не є сусід зліва граничним вузлом Діріхле.

                if i == 1:
                    # Сусід зліва (i=0) має фіксоване значення Cin.
                    # Переносимо відомий член (Aw * Cin) у праву частину рівняння.
                    b[k] -= Aw * Cin
                else:
                    # Сусід зліва - невідомий, додаємо в матрицю
                    rows.append(k); cols.append(global_k(i-1, j, Nx)); data.append(Aw)

            # 4. Північ (k, k+Nx)
            if j < Ny - 1 and An != 0:
                rows.append(k); cols.append(global_k(i, j+1, Nx)); data.append(An)

            # 5. Південь (k, k-Nx)
            if j > 0 and As != 0:
                rows.append(k); cols.append(global_k(i, j-1, Nx)); data.append(As)

    # 4. Створення та конвертація матриці
    A_coo = coo_array((data, (rows, cols)), shape=(N, N))
    A_csr = A_coo.tocsr()
    return A_csr, b

# --- Приклад Використання ---
Lx = 1000.0
Ly = 50.0
Nx = 100
Ny = 50
Dx = 1.0
Dy = 0.1
Ux = 0.5
Uy = 0.0
k_decay = 1e-4
Cin = 10.0

A, b = build_system(Lx, Ly, Nx, Ny, Dx, Dy, Ux, Uy, k_decay, Cin)

# 5. Розв'язання системи
C_1D = spsolve(A, b)

# 6. Перетворення результату на 2D масив
C_2D = C_1D.reshape((Ny, Nx))

print(f"Розрахунок завершено. Максимальна концентрація: {np.max(C_2D):.2f}")
print(f"Концентрація на виході (центр): {C_2D[Ny//2, -1]:.2f}")