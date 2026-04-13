import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

# --- 1. Параметри моделі ---
L = 10000.0  # м
T = 10.0     # дні
U_ms = 0.5   # м/с
D_m2s = 10.0 # м2/с

U = U_ms * 86400  # м/день
D = D_m2s * 86400 # м2/день

dx = 100.0   # м
dt = 0.05    # день
J = int(L / dx) + 1
N = int(T / dt) + 1

k1 = 0.2  # день^-1 (NH4 -> NO2)
k2 = 0.8  # день^-1 (NO2 -> NO3)
k3 = 0.05 # день^-1 (NO3 removal)

S_A_diffuse = 10.0 # мг/м3/день (дифузне джерело амонію)

# Початкові умови (НУ) та граничні умови (ГУ)
C_A_init = 1.0  # мг/м3
C_N_init = 0.1  # мг/м3
C_R_init = 2.0  # мг/м3
C_inlet = 0.5   # мг/м3 (ГУ Діріхле на вході)

x_grid = np.linspace(0, L, J)

def solve_cn_step(C_n, D, U, k, S_diff, dx, dt, J, boundary_inlet, generation_term=None):
    """
    Розв'язує один часовий крок методом Кранка-Ніколсона.
    """
    # ВИПРАВЛЕННЯ 1: Обробка generation_term, якщо це число або None
    if generation_term is None:
        generation_term = np.zeros(J)
    elif isinstance(generation_term, (float, int)):
        val = generation_term
        generation_term = np.full(J, val)

    # Коефіцієнти CN
    rD = D * dt / (2 * dx**2)
    rU = U * dt / (4 * dx)
    rK = k * dt / 2
    S_star = S_diff * dt

    # --- 2. Формування матриці A ---
    # (upper, main, lower)
    A_banded = np.zeros((3, J))

    # Головна діагональ (j=1 до J-2)
    A_banded[1, 1:J-1] = 1 + 2 * rD + rK

    # Верхня діагональ (stored at row 0).
    # Елемент A[i, i+1] зберігається в A_banded[0, i+1].
    A_banded[0, 2:J] = -rD - rU

    # Нижня діагональ (stored at row 2).
    # Елемент A[i, i-1] зберігається в A_banded[2, i-1].
    A_banded[2, 0:J-2] = -rD + rU

    # --- 3. Граничні умови (ГУ) ---

    # ГУ Вхід (j=0, Dirichlet: C = C_inlet) -> 1 * C[0] = C_inlet
    # ВИПРАВЛЕННЯ 2: Присвоєння конкретному елементу, а не перезапис змінної
    A_banded[1, 0] = 1.0

    # ГУ Вихід (j=J-1, Neumann: C[J-1] - C[J-2] = 0)
    # Рівняння: 1*C[J-1] + (-1)*C[J-2] = 0
    A_banded[1, J-1] = 1.0       # Головна діагональ для C[J-1]
    # Нижня діагональ для стовпця J-2 (який відповідає C[J-2] у рядку J-1)
    # У форматі solve_banded: row=2 (lower), col=J-2
    A_banded[2, J-2] = -1.0

    # --- 4. Формування вектора B ---
    B = np.zeros(J)

    # Внутрішні точки (j=1 до J-2)
    B[1:J-1] = (C_n[1:J-1] * (1 - 2 * rD - rK) +
                C_n[0:J-2] * (rD - rU) +
                C_n[2:J] * (rD + rU) +
                S_star + generation_term[1:J-1] * dt)

    # ГУ Вхід (j=0, Dirichlet)
    # ВИПРАВЛЕННЯ 3: Присвоєння елементу масиву
    B[0] = boundary_inlet

    # ГУ Вихід (j=J-1, Neumann) -> C[J-1] - C[J-2] = 0 -> RHS = 0
    B[J-1] = 0.0

    # --- 5. Розв'язання системи ---
    C_next = solve_banded((1, 1), A_banded, B)
    return C_next

# --- 6. Головний цикл інтеграції ---

C_A = np.full(J, C_A_init)
C_N = np.full(J, C_N_init)
C_R = np.full(J, C_R_init)

C_A_history = [C_A.copy()]
C_N_history = [C_N.copy()]
# ВИПРАВЛЕННЯ 4: Синтаксис списку
C_R_history = [C_R.copy()]

for n in range(N - 1):
    # --- A. NH4+ ---
    # ВИПРАВЛЕННЯ 5: generation_term=0.0 тепер обробляється коректно у функції
    C_A_next = solve_cn_step(C_A, D, U, k1, S_A_diffuse, dx, dt, J, C_inlet, generation_term=0.0)

    # --- B. NO2- ---
    R_gen_N = k1 * 0.5 * (C_A + C_A_next)
    C_N_next = solve_cn_step(C_N, D, U, k2, 0.0, dx, dt, J, C_inlet * 0.1, generation_term=R_gen_N)

    # --- C. NO3- ---
    R_gen_R = k2 * 0.5 * (C_N + C_N_next)
    C_R_next = solve_cn_step(C_R, D, U, k3, 0.0, dx, dt, J, C_inlet * 4.0, generation_term=R_gen_R)

    # Оновлення
    C_A = C_A_next
    C_N = C_N_next
    C_R = C_R_next

    if (n + 1) % 50 == 0:
        C_A_history.append(C_A.copy())
        C_N_history.append(C_N.copy())
        C_R_history.append(C_R.copy())

# Додаємо фінальний крок, якщо він не зберігся
if (N-1) % 50 != 0:
    C_A_history.append(C_A.copy())
    C_N_history.append(C_N.copy())
    C_R_history.append(C_R.copy())

# --- 7. Візуалізація результатів ---
plt.figure(figsize=(12, 6))

# Побудова графіків (трохи спрощена логіка циклу)
colors = ['red', 'orange', 'blue']
labels = ['NH4+', 'NO2-', 'NO3-']
histories = [C_A_history, C_N_history, C_R_history]

for idx, history in enumerate(histories):
    # Початковий стан
    plt.plot(x_grid / 1000, history[0], linestyle='--', color=colors[idx], alpha=0.5,
             label=f'{labels[idx]} (T=0)')
    # Кінцевий стан
    plt.plot(x_grid / 1000, history[-1], linestyle='-', color=colors[idx], linewidth=2,
             label=f'{labels[idx]} (T={T} дні)')

plt.title('Просторовий розподіл азотних сполук (NH4 -> NO2 -> NO3)')
plt.xlabel('Відстань, км')
plt.ylabel('Концентрація, мг/м$^3$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()