import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --- 1. Допоміжні функції та Солвер ---

def global_k(i, j, Nx):
    """Перетворення 2D індексу (i, j) на 1D глобальний індекс k."""
    return j * Nx + i

def build_sparse_system(Lx, Ly, Nx, Ny, Dx, Dy, Ux, Uy, k_decay, Cin):
    N = Nx * Ny
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    # Коефіцієнти
    Px = Dx / dx**2
    Py = Dy / dy**2
    Qx = Ux / dx
    Qy = Uy / dy
    K = k_decay

    rows, cols, data = [], [], []
    b = np.zeros(N)

    for j in range(Ny):
        for i in range(Nx):
            k = global_k(i, j, Nx)

            # --- Границя на вході (Dirichlet) ---
            if i == 0:
                rows.append(k); cols.append(k); data.append(1.0)
                b[k] = Cin
                continue

            # --- Внутрішні вузли ---
            Ac = -2*Px - 2*Py - Qx - Qy - K
            Ae, Aw, An, As = Px, Px + Qx, Py, Py + Qy

            # --- Граничні умови Неймана (dC/dn = 0) ---
            # Використовуємо метод Ghost Point (дзеркальне відображення)

            # Правий край (Outlet)
            if i == Nx - 1:
                Aw += Ae  # Ae повертається назад
                Ae = 0

            # Нижня стінка
            if j == 0:
                An += As  # As повертається назад
                As = 0

            # Верхня стінка
            if j == Ny - 1:
                As += An  # An повертається назад
                An = 0

            # --- Запис у матрицю ---
            rows.append(k); cols.append(k); data.append(Ac)

            if i < Nx - 1 and Ae != 0:
                rows.append(k); cols.append(global_k(i+1, j, Nx)); data.append(Ae)

            if i > 0 and Aw != 0:
                k_w = global_k(i-1, j, Nx)
                if i == 1:
                    b[k] -= Aw * Cin # Перенос відомого Dirichlet в праву частину
                else:
                    rows.append(k); cols.append(k_w); data.append(Aw)

            if j < Ny - 1 and An != 0:
                rows.append(k); cols.append(global_k(i, j+1, Nx)); data.append(An)

            if j > 0 and As != 0:
                rows.append(k); cols.append(global_k(i, j-1, Nx)); data.append(As)

    A_coo = coo_array((data, (rows, cols)), shape=(N, N))
    return A_coo.tocsr(), b

# --- 2. Параметри та Розрахунок ---

Lx, Ly = 33000.0, 12.0
Nx, Ny = 200, 5
Dx, Dy = 5.0, 0.5
Ux, Uy = 0.8, 0.0
k_decay = 12e-8 # Коефіцієнт розпаду
Cin = 2.49

print("Будуємо систему...")
A, b = build_sparse_system(Lx, Ly, Nx, Ny, Dx, Dy, Ux, Uy, k_decay, Cin)
print("Розв'язуємо систему...")
C_1D = spsolve(A, b)
C_2D = C_1D.reshape((Ny, Nx))

# --- 3. Візуалізація результатів ---

# Створюємо сітку координат для графіків (в метрах)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(14, 10))

# Графік 1: 2D Поле концентрації
ax1 = plt.subplot(2, 1, 1) # Верхня половина
# extent задає реальні розміри в метрах [xmin, xmax, ymin, ymax]
# aspect='auto' важливий, бо Lx >> Ly (5000 vs 100). Без нього графік буде сплюснутим.
im = ax1.imshow(C_2D, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis', aspect='auto')
ax1.set_title(f'Розподіл концентрації (Ux={Ux}, Dx={Dx}, Dy={Dy})')
ax1.set_xlabel('Довжина X (м)')
ax1.set_ylabel('Ширина Y (м)')
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Концентрація C')

# Графік 2: Профіль вздовж центру (потік)
ax2 = plt.subplot(2, 2, 3) # Лівий нижній
y_center_idx = Ny // 2
ax2.plot(x, C_2D[y_center_idx, :], 'r-', linewidth=2, label=f'Центр (Y={y[y_center_idx]:.1f})')
ax2.plot(x, C_2D[0, :], 'b--', alpha=0.6, label='Стінка (Y=0)')
ax2.set_title('Зміна концентрації вздовж потоку')
ax2.set_xlabel('Відстань X (м)')
ax2.set_ylabel('Концентрація')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

# Графік 3: Поперечні профілі (дифузія)
ax3 = plt.subplot(2, 2, 4) # Правий нижній
# Беремо зрізи на початку, в середині та в кінці каналу
indices = [10, Nx//4, Nx//2, Nx-1]
for idx in indices:
    ax3.plot(y, C_2D[:, idx], label=f'X={x[idx]:.0f} м')

ax3.set_title('Поперечні профілі (розмиття)')
ax3.set_xlabel('Ширина Y (м)')
ax3.set_ylabel('Концентрація')
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend()

plt.tight_layout()
plt.show()