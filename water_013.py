import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import colors

# ==============================================================================
# 1. ІНІЦІАЛІЗАЦІЯ
# ==============================================================================
jax.config.update("jax_enable_x64", True)

try:
    DEVICE = jax.devices()[0]
    print(f"JAX backend: {DEVICE.platform.upper()}")
except Exception:
    print("Пристрій не знайдено. Використовується CPU.")

# --- Параметри ---
NX, NY = 500, 200
L_X, L_Y = 30000.0, 100.0
dx, dy = L_X / NX, L_Y / NY
N_NODES = NX * NY
N_TOT = 2 * N_NODES

U = 0.5
DL = 10.0
DT = 0.1
T_ENV = 298.15
T_REF = 293.15

# --- Кінетика ---
k_nitr_20 = 1.0e-5
k_denit_20 = 5.0e-6
THETA = 1.05

def calculate_kinetics(k_20, T, T_ref, theta):
    return k_20 * (theta ** ((T - T_ref) / 1.0))

K_NITR = calculate_kinetics(k_nitr_20, T_ENV, T_REF, THETA)
K_DENIT = calculate_kinetics(k_denit_20, T_ENV, T_REF, THETA)

# ==============================================================================
# 2. КОЕФІЦІЄНТИ ТА МАСКИ (!!! FIX: Додано маски для FDM)
# ==============================================================================
AW = DL / dx**2 + U / dx
AE = DL / dx**2
AS = DT / dy**2
AN = DT / dy**2
AP_TRANS = -(2*DL/dx**2 + 2*DT/dy**2 + U/dx)

# Створення масок для видалення періодичності (щоб річка не текла по колу)
# 1 - внутрішні вузли, 0 - границя, з якої не має братися значення (wrap-around)
mask_W = jnp.ones((NY, NX)).at[:, 0].set(0.0)   # Не брати з "останнього" стовпця при обчисленні першого
mask_E = jnp.ones((NY, NX)).at[:, -1].set(0.0)  # Не брати з "першого" стовпця при обчисленні останнього
mask_S = jnp.ones((NY, NX)).at[0, :].set(0.0)
mask_N = jnp.ones((NY, NX)).at[-1, :].set(0.0)

# ==============================================================================
# 3. JIT-ОПЕРАТОР (!!! FIX: Розділення векторів та Граничні умови)
# ==============================================================================

@jax.jit
def matvec_operator_coupled(C_1D: jnp.ndarray) -> jnp.ndarray:
    # !!! FIX 1: Правильне розділення вхідного вектора на компоненти
    C_A_1D = C_1D[:N_NODES]
    C_N_1D = C_1D[N_NODES:]

    C_A_2D = C_A_1D.reshape((NY, NX))
    C_N_2D = C_N_1D.reshape((NY, NX))

    def transport_stencil(C_2D):
        R_2D = AP_TRANS * C_2D
        # !!! FIX 2: Множення на маски, щоб прибрати циклічність jnp.roll
        R_2D += AW * (jnp.roll(C_2D, shift=1, axis=1) * mask_W) # i-1
        R_2D += AE * (jnp.roll(C_2D, shift=-1, axis=1) * mask_E) # i+1
        R_2D += AS * (jnp.roll(C_2D, shift=1, axis=0) * mask_S) # j-1
        R_2D += AN * (jnp.roll(C_2D, shift=-1, axis=0) * mask_N) # j+1
        return R_2D

    # Система рівнянь
    RHS_A_2D = transport_stencil(C_A_2D) - K_NITR * C_A_2D

    RHS_N_2D = (
        transport_stencil(C_N_2D)
        + K_NITR * C_A_2D
        - K_DENIT * C_N_2D
    )

    return jnp.concatenate((RHS_A_2D.ravel(), RHS_N_2D.ravel()))

# ==============================================================================
# 4. ВЕКТОР ПРАВОЇ ЧАСТИНИ B
# ==============================================================================
C_IN_A = 100.0
C_IN_N = 0.0
S_MAX_N = 1.0e-3 # !!! Збільшив для наочності на графіку (було e-6, це дуже мало)

X_START_SRC = int(200.0 / dx)
X_END_SRC = int(400.0 / dx)

# !!! FIX 3: Правильне використання .at[].set з індексами
S_TERM_N_2D = jnp.zeros((NY, NX))
S_TERM_N_2D = S_TERM_N_2D.at[:, X_START_SRC:X_END_SRC].set(S_MAX_N)

# Формування B
B_A_2D = jnp.zeros((NY, NX))
B_N_2D = -S_TERM_N_2D # B = -Source (перенесено вправо)

# Врахування Діріхле на вході (x=0)
# Вплив границі (x=-1, яка є C_IN) на вузол x=0 через коефіцієнт AW
B_A_2D = B_A_2D.at[:, 0].add(-AW * C_IN_A)
B_N_2D = B_N_2D.at[:, 0].add(-AW * C_IN_N)
# Примітка: Знаки залежать від того, як ми записали A*x = b.
# Тут ми перенесли A*x вліво, тому тут додаємо внесок відомих членів з протилежним знаком.
# Оскільки в Stencil ми використовуємо "+AW*Neighbor", при перенесенні відомого Neighbor вправо це стає "-AW*Val".

B_1D = jnp.concatenate((B_A_2D.ravel(), B_N_2D.ravel()))

# ==============================================================================
# 5. РОЗВ'ЯЗАННЯ
# ==============================================================================
C0 = jnp.zeros(N_TOT)
print(f"Система: {N_TOT} невідомих.")

# "Прогрів" JIT
_ = matvec_operator_coupled(C0).block_until_ready()
print("JIT компіляція завершена. Старт GMRES...")

start_time = time.time()
# Збільшив restart для складності, але зменшив maxiter для швидкості демо
solution_1d, info = gmres(
    A=matvec_operator_coupled,
    b=B_1D,
    x0=C0,
    tol=1e-5,
    restart=100,
    maxiter=2000
)
solution_1d.block_until_ready()
total_time = time.time() - start_time

print(f"Статус GMRES: {info} (0=OK). Час: {total_time:.4f} с.")

# ==============================================================================
# 6. ВІЗУАЛІЗАЦІЯ
# ==============================================================================
# !!! FIX 4: Розділення результату перед reshape
C_SOL_A_1D = solution_1d[:N_NODES]
C_SOL_N_1D = solution_1d[N_NODES:]

C_SOLUTION_A = C_SOL_A_1D.reshape((NY, NX))
C_SOLUTION_N = C_SOL_N_1D.reshape((NY, NX))

# Для візуалізації: JAX масиви -> Numpy
Z_A = np.array(C_SOLUTION_A)
Z_N = np.array(C_SOLUTION_N)

x_coords = np.linspace(0, L_X, NX)
y_coords = np.linspace(0, L_Y, NY)
X, Y = np.meshgrid(x_coords, y_coords)

plt.figure(figsize=(12, 5))

# Амоній
plt.subplot(1, 2, 1)
# !!! FIX 5: Прибрано .T (транспонування), воно ламало відповідність осей
plt.contourf(X, Y, Z_A, levels=50, cmap='viridis')
plt.colorbar(label='NH4 (мг/м3)')
plt.title('Амоній (Розпад + Адвекція)')
plt.xlabel('x (м)')
plt.ylabel('y (м)')

# Нітрат
plt.subplot(1, 2, 2)
plt.contourf(X, Y, Z_N, levels=50, cmap='inferno')
plt.colorbar(label='NO3 (мг/м3)')
plt.title('Нітрат (Утворення + Джерело)')
plt.xlabel('x (м)')

# Зона джерела
plt.axvline(x=X_START_SRC*dx, color='white', linestyle='--', alpha=0.5)
plt.axvline(x=X_END_SRC*dx, color='white', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()