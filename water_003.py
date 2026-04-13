# @title Моделювання забруднення річки (Точкове джерело)
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve

# --- 1. Функції розрахунку (Ядро) ---

def global_k(i, j, Nx):
    """Переведення 2D індексу в 1D"""
    return j * Nx + i

def build_system(Lx, Ly, Nx, Ny, Dx, Dy, Ux, Uy, k_decay, Cin, source_width_fraction=0.2):
    """
    Побудова системи рівнянь для стаціонарної задачі.
    source_width_fraction: яку частку ширини річки займає джерело (0.0 - 1.0)
    """
    N = Nx * Ny
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    # Числа Пекле та безрозмірні параметри
    Px = Dx / dx**2
    Py = Dy / dy**2
    Qx = Ux / dx
    Qy = Uy / dy
    K = k_decay

    rows, cols, data = [], [], []
    b = np.zeros(N)

    # Визначаємо межі джерела (по центру річки)
    j_center = Ny // 2
    j_width = int(Ny * source_width_fraction / 2)
    j_source_start = j_center - j_width
    j_source_end = j_center + j_width

    print(f"Джерело забруднення розташоване між вузлами Y: [{j_source_start}, {j_source_end}]")

    for j in range(Ny):
        for i in range(Nx):
            k = global_k(i, j, Nx)

            # --- ГРАНИЧНА УМОВА НА ВХОДІ (i=0) ---
            if i == 0:
                rows.append(k); cols.append(k); data.append(1.0)

                # ЛОГІКА ТОЧКОВОГО ДЖЕРЕЛА:
                # Якщо вузол потрапляє в зону скиду -> концентрація Cin
                # Якщо ні (чиста вода навколо) -> концентрація 0
                if j_source_start <= j <= j_source_end:
                    b[k] = Cin
                else:
                    b[k] = 0.0
                continue

            # --- ВНУТРІШНІ ВУЗЛИ ТА НЕЙМАН ---
            Ac = -2*Px - 2*Py - Qx - Qy - K
            Ae, Aw, An, As = Px, Px + Qx, Py, Py + Qy

            # Обробка границь (Нейман: похідна = 0)
            if i == Nx - 1: # Вихід
                Ac += Ae; Ae = 0
            if j == 0:      # Нижній берег
                Ac += As; As = 0
            if j == Ny - 1: # Верхній берег
                Ac += An; An = 0

            # Запис у матрицю
            rows.append(k); cols.append(k); data.append(Ac)

            if i < Nx - 1 and Ae != 0:
                rows.append(k); cols.append(global_k(i+1, j, Nx)); data.append(Ae)

            if i > 0 and Aw != 0:
                # Перевірка сусіда зліва (чи це вхідна границя?)
                if i == 1:
                    # Якщо сусід зліва - це вхід (i=0), перевіряємо, чи він у зоні забруднення
                    val_at_boundary = Cin if (j_source_start <= j <= j_source_end) else 0.0
                    b[k] -= Aw * val_at_boundary
                else:
                    rows.append(k); cols.append(global_k(i-1, j, Nx)); data.append(Aw)

            if j < Ny - 1 and An != 0:
                rows.append(k); cols.append(global_k(i, j+1, Nx)); data.append(An)
            if j > 0 and As != 0:
                rows.append(k); cols.append(global_k(i, j-1, Nx)); data.append(As)

    A_coo = coo_array((data, (rows, cols)), shape=(N, N))
    return A_coo.tocsr(), b

# --- 2. Параметри моделювання ---

# Геометрія
Lx = 1000.0   # Довжина (м)
Ly = 100.0    # Ширина (м) - зробив ширшою для наочності
Nx = 150      # Більше вузлів для гладкості
Ny = 60

# Фізика
Dx = 2.0      # Поздовжня дисперсія
Dy = 0.8      # Поперечна дисперсія (визначає ширину "факела")
Ux = 0.4      # Швидкість течії (м/с)
Uy = 0.0      # Поперечної течії немає
k_decay = 2e-4 # Коефіцієнт розпаду/осідання
Cin = 50.0    # Концентрація у джерелі

# --- 3. Розрахунок ---
print("1. Генерація системи рівнянь...")
A, b = build_system(Lx, Ly, Nx, Ny, Dx, Dy, Ux, Uy, k_decay, Cin, source_width_fraction=0.15)

print("2. Розв'язання розрідженої матриці...")
C_1D = spsolve(A, b)
C_2D = C_1D.reshape((Ny, Nx))

# --- 4. Візуалізація ---
print("3. Побудова графіків...")
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(15, 12), dpi=100) # dpi=100 для гарної якості в Colab

# Графік 1: Heatmap
plt.subplot(2, 1, 1)
# Використовуємо contourf для більш гладкої картинки + pcolormesh для точності
pcm = plt.pcolormesh(X, Y, C_2D, cmap='turbo', shading='auto')
plt.colorbar(pcm, label='Концентрація (мг/л)')
plt.title(f'Поширення забруднення від точкового джерела\nUx={Ux} м/с, Dy={Dy} м²/с, K={k_decay}')
plt.xlabel('Відстань (м)')
plt.ylabel('Ширина річки (м)')
plt.axhline(Ly/2, color='white', linestyle='--', alpha=0.3, label='Центр річки')

# Графік 2: Перерізи
plt.subplot(2, 1, 2)
# Беремо зрізи на відстанях: 50м, 200м, 500м, 900м
indices = [int(Nx*0.05), int(Nx*0.2), int(Nx*0.5), int(Nx*0.9)]
for idx in indices:
    dist = x[idx]
    plt.plot(y, C_2D[:, idx], label=f'Відстань X = {dist:.1f} м', linewidth=2)

plt.title('Поперечний розподіл забруднення (Профілі)')
plt.xlabel('Ширина річки (м)')
plt.ylabel('Концентрація (мг/л)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()